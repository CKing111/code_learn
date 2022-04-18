import numpy as np

from uti import interface_check
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor

from .affine_grid import AffineGrid


class AffineGrid3D(AffineGrid):

    @staticmethod
    def _check_params(theta, grid):
        interface_check.check_param(theta, [5], ["float16"], ["NC1HWC0"])
        interface_check.check_param(grid, [6], ["float16"], ["NDC1HWC0"])

    @staticmethod
    def _check_params_shape(theta, grid, matrix_size, align_corners):
        if len(matrix_size) != 5:
            raise RuntimeError("[ERROR] matrix_size length is not supported")
        batch, channel, depth, height, width = matrix_size
        if theta.get("shape") != (batch, 1, 3, 4, 16):
            raise RuntimeError("[ERROR] shape of theta is not supported")
        if align_corners and (depth == 1 or height == 1 or width == 1):
            raise RuntimeError("[ERROR] matrix_size dims equal 1")
        max_data_num = 20000000
        max_dim_num = 10000
        if batch * depth * height * width > max_data_num or \
                depth > max_dim_num or height > max_dim_num or \
                width > max_dim_num:
            raise RuntimeError("[ERROR] matrix_size is too larger")
        if grid.get("shape") != (batch, depth, 1, height, width, 16):
            raise RuntimeError("[ERROR] shape of grid is not supported")

    @staticmethod
    def _split_shape(matrix_size_shape):
        """
        get shape info
        Args:
            matrix_size_shape: tuple(n, c, h, w)
        Returns: batch_num, matrix_size
        """
        theta_num = 12
        matrix_size_dict = dict()
        batch_num = matrix_size_shape[0]
        matrix_size_dict["d"] = matrix_size_shape[2]
        matrix_size_dict["h"] = matrix_size_shape[3]
        matrix_size_dict["w"] = matrix_size_shape[4]
        return batch_num, matrix_size_dict, theta_num

    def _mode_split_n_compute_each_core(self, batch_index_start,
                                        each_core_batch_num):
        width, height, depth = self.matrix_size.get("w"), self.matrix_size.get(
            "h"), self.matrix_size.get("d")
        grid_base_l1 = self._init_l1_tensor(
            self.fp32_type, self.fp32_block_data_num, [width, height, depth],
            "grid_base")

        grid_base_width_l1, grid_base_height_l1, grid_base_depth_l1 = \
            grid_base_l1
        self._get_base_grid(grid_base_width_l1, width, self.dim_start.get("w"),
                            self.dim_stride.get("w"))
        self._get_base_grid(grid_base_height_l1, height,
                            self.dim_start.get("h"), self.dim_stride.get("h"))
        self._get_base_grid(grid_base_depth_l1, depth, self.dim_start.get("d"),
                            self.dim_stride.get("d"))

        thread_num = 1 if each_core_batch_num == 1 else 2
        ub_size = self.cont.const_ub_max_byte // thread_num
        with self.tik_inst.for_range(0, each_core_batch_num,
                                     thread_num=thread_num) as batch_index:
            batch_index_now = batch_index_start + batch_index
            self._mode1_compute_each_loop(grid_base_l1, batch_index_now,
                                          ub_size)

    def _mode1_compute_each_loop(self, grid_base_l1, batch_index, ub_size):
        # init grid mul tensor
        width, height, depth = self.matrix_size.get("w"), self.matrix_size.get(
            "h"), self.matrix_size.get("d")
        l1_data_num_all = [width, width, width, height, height, height, depth,
                           depth, depth]
        grid_l1 = self._init_l1_tensor(
            self.fp32_type, self.fp32_block_data_num, l1_data_num_all,
            "grid_mul")
        # init theta
        theta_scalar = [self.tik_inst.Scalar(self.fp32_type) for _ in
                        range(self.theta_num)]
        self._get_theta_scalar(theta_scalar, batch_index)
        # width mul scalar[0] add scalar[3]; width mul scalar[4] add scalar[7];
        # width mul scalar[8] add scalar[11]
        # height mul scalar[1]; height mul scalar[5]; height mul scalar[9]
        # depth mul scalar[2]; depth mul scalar[6]; depth mul scalar[10]
        grid_base_width_l1, grid_base_height_l1, grid_base_depth_l1 = \
            grid_base_l1
        grid_width_mul_l1 = grid_l1[:3]
        grid_height_mul_l1 = grid_l1[3:6]
        grid_depth_mul_l1 = grid_l1[6:]
        weight_scalar_all = (theta_scalar[0], theta_scalar[4], theta_scalar[8])
        height_scalar_all = (theta_scalar[1], theta_scalar[5], theta_scalar[9])
        depth_scalar_all = (theta_scalar[2], theta_scalar[6], theta_scalar[10])
        bias_scalar_all = (theta_scalar[3], theta_scalar[7], theta_scalar[11])
        self._get_l1_grid(width, grid_base_width_l1, grid_width_mul_l1,
                          weight_scalar_all, bias_scalar_all)
        self._get_l1_grid(height, grid_base_height_l1, grid_height_mul_l1,
                          height_scalar_all)
        self._get_l1_grid(depth, grid_base_depth_l1, grid_depth_mul_l1,
                          depth_scalar_all)
        # init result
        self._transpose_split_weight(grid_l1, ub_size, batch_index)

    def _transpose_split_height(self, grid_l1, max_num_each_loop, batch_index,
                                w_start, width):
        depth, height = self.matrix_size.get("d"), self.matrix_size.get("h")
        each_loop_h_d_num_max = depth * height
        width_format = self._get_format_num(width, self.transpose_num)
        each_loop_h_d_num = max_num_each_loop // width_format
        each_loop_h_d_num = min(each_loop_h_d_num, each_loop_h_d_num_max)
        transpose_dst_ub, transpose_src_ub, add_grid_ub, h_d_ub = \
            self._init_transpose_tensor(each_loop_h_d_num, width_format)
        # pad width
        dim_num, l1_index_start, add_grid_index_start = 3, 0, 3
        self._pad_width(
            grid_l1, add_grid_ub, dim_num, l1_index_start,
            add_grid_index_start,
            each_loop_h_d_num, w_start, width)
        # split height
        each_loop_height_num = min(each_loop_h_d_num, height)
        last_loop_height_num, height_loop_num = self._get_loop_info(
            height, each_loop_height_num)
        with self.tik_inst.for_range(0, height_loop_num) as height_index:
            h_start = height_index * each_loop_height_num
            with self.tik_inst.if_scope(height_index != height_loop_num - 1):
                self._transpose_split_depth(
                    transpose_dst_ub, transpose_src_ub, add_grid_ub, h_d_ub,
                    grid_l1, each_loop_h_d_num, batch_index,
                    w_start, width, h_start, each_loop_height_num)
            with self.tik_inst.else_scope():
                self._transpose_split_depth(
                    transpose_dst_ub, transpose_src_ub, add_grid_ub, h_d_ub,
                    grid_l1, each_loop_h_d_num, batch_index,
                    w_start, width, h_start, last_loop_height_num)

    def _transpose_split_depth(self, transpose_dst_ub, transpose_src_ub,
                               add_grid_ub, h_d_ub, grid_l1,
                               each_loop_h_d_num, batch_index, w_start, width,
                               h_start, height):
        # pad height
        depth = self.matrix_size.get("d")
        each_loop_depth_num = min(each_loop_h_d_num // height, depth)
        dim_num, l1_index_start, add_grid_index_start = 3, 3, 6
        self._pad_height(
            grid_l1, add_grid_ub, h_d_ub, dim_num, l1_index_start,
            add_grid_index_start,
            each_loop_depth_num, h_start, height, width)
        # split depth
        last_loop_depth_num, depth_loop_num = self._get_loop_info(
            depth, each_loop_depth_num)
        with self.tik_inst.for_range(0, depth_loop_num) as depth_index:
            d_start = depth_index * each_loop_depth_num
            with self.tik_inst.if_scope(depth_index != depth_loop_num - 1):
                self._transpose_3d(
                    transpose_dst_ub, transpose_src_ub, add_grid_ub, h_d_ub,
                    grid_l1, batch_index, w_start, width,
                    h_start, height, d_start, each_loop_depth_num)
            with self.tik_inst.else_scope():
                self._transpose_3d(
                    transpose_dst_ub, transpose_src_ub, add_grid_ub, h_d_ub,
                    grid_l1, batch_index, w_start, width,
                    h_start, height, d_start, last_loop_depth_num)

    def _get_transpose_3d_buf(self, transpose_src_ub, add_grid_ub, grid_num,
                              num_per_cmd, each_batch_num):
        dims = ["width", "height", "depth"]
        grid_name_all = ["grid_{}".format(dim) for dim in dims]
        grid_name_all.extend(["grid_{}_{}".format(dim, index_i)
                              for dim in dims for index_i in range(3)])
        grid_result_name = ["grid_{}_result".format(dim) for dim in dims]
        buf_add_all = {name: AVecBuf(add_grid_ub, grid_num,
                                     each_batch_num * index_i, self.cont,
                                     False, num_per_cmd)
                       for index_i, name in enumerate(grid_name_all)}
        buff_add_result = {name: AVecBuf(transpose_src_ub, grid_num,
                                         each_batch_num * index_i,
                                         self.cont, False, num_per_cmd)
                           for index_i, name in enumerate(grid_result_name)}

        buf_add_all.update(buff_add_result)
        return buf_add_all

    def _transpose_3d(self, transpose_dst_ub, transpose_src_ub, add_grid_ub,
                      h_d_ub, grid_l1, batch_index,
                      w_start, width, h_start, height, d_start, depth):
        dim_num, l1_index_start, add_grid_index_start = 3, 6, 9
        self._pad_depth(
            grid_l1, add_grid_ub, h_d_ub, dim_num, l1_index_start,
            add_grid_index_start,
            d_start, depth, height, width)

        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        width_format = add_grid_ub.shape[2]
        grid_num = depth * height * width_format
        each_batch_num = add_grid_ub.shape[1] * add_grid_ub.shape[2]
        buf_add_all = self._get_transpose_3d_buf(
            transpose_src_ub, add_grid_ub, grid_num,
            num_per_cmd, each_batch_num)
        cmd_add_tensor = [
            VecGCmd(cmd_name="vadd", dst_name="grid_width",
                    src0_name="grid_width_0", src1_name="grid_height_0"),
            VecGCmd(cmd_name="vadd", dst_name="grid_width",
                    src0_name="grid_width", src1_name="grid_depth_0"),
            VecGCmd(cmd_name="vadd", dst_name="grid_height",
                    src0_name="grid_width_1", src1_name="grid_height_1"),
            VecGCmd(cmd_name="vadd", dst_name="grid_height",
                    src0_name="grid_height", src1_name="grid_depth_1"),
            VecGCmd(cmd_name="vadd", dst_name="grid_depth",
                    src0_name="grid_width_2", src1_name="grid_height_2"),
            VecGCmd(cmd_name="vadd", dst_name="grid_depth",
                    src0_name="grid_depth", src1_name="grid_depth_2"),
            VecGCmd(cmd_name="vconv", dst_name="grid_width_result",
                    src0_name="grid_width", round_mode=""),
            VecGCmd(cmd_name="vconv", dst_name="grid_height_result",
                    src0_name="grid_height", round_mode=""),
            VecGCmd(cmd_name="vconv", dst_name="grid_depth_result",
                    src0_name="grid_depth", round_mode="")]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_add_all, cmd_add_tensor,
                                    "grid_width")

        repeat_num = grid_num // 16
        self._vnchwconv(transpose_dst_ub, transpose_src_ub, repeat_num)
        with self.tik_inst.for_range(0, depth) as depth_index:
            self.tik_inst.data_move(
                self.grid[batch_index, d_start + depth_index,
                          0, h_start, w_start, 0],
                transpose_dst_ub[depth_index * height, 0, 0], 0,
                height, width, max(0, width_format - width),
                max(0, self.matrix_size.get("w") - width))

    def _pad_depth(self, grid_l1, add_grid_ub, h_d_ub, dim_num, l1_index_start,
                   add_grid_index_start,
                   d_start, depth, height, width):
        """
        pad depth
        """
        d_start_format, d_index_start, d_block_num = \
            self._get_format_start_end(d_start, depth,
                                       self.fp32_block_data_num)
        depth_scalar = self.tik_inst.Scalar(self.fp32_type)
        width_format = self._get_format_num(width, self.transpose_num)
        h_w_data_num = height * width_format
        each_batch_num = add_grid_ub.shape[1] * add_grid_ub.shape[2]
        # depth data move to ub
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        for dim_index in range(dim_num):
            l1_index = l1_index_start + dim_index
            add_grid_index = add_grid_index_start + dim_index
            self.tik_inst.data_move(h_d_ub, grid_l1[l1_index][d_start_format],
                                    0, 1, d_block_num, 0, 0)
            with self.tik_inst.for_range(0, depth) as depth_index:
                depth_scalar.set_as(h_d_ub[d_index_start + depth_index])
                buf_dup_all = {
                    "tensor_dst": AVecBuf(
                        add_grid_ub, h_w_data_num,
                        each_batch_num * add_grid_index +
                        depth_index * h_w_data_num,
                        self.cont, False, num_per_cmd)}
                cmd_dup_tensor = [
                    VecGCmd(cmd_name="vector_dup", dst_name="tensor_dst",
                            scalar=depth_scalar)]
                VecGExecutor.exec_vec_g_cmd(self.cont, buf_dup_all,
                                            cmd_dup_tensor, "tensor_dst")

    def tik_output_debug(self):
        theta_gm = np.zeros(self.theta_shape, np.float16)
        for index_batch in range(self.batch_num):
            radian = np.random.rand() * np.pi * 2 - np.pi
            bias_w = np.random.rand() * 2 - 1
            bias_h = np.random.rand() * 2 - 1
            bias_d = np.random.rand() * 2 - 1
            each_batch_theta = np.array(
                [[np.cos(radian), -np.sin(radian), np.sin(radian), bias_w],
                 [np.sin(radian), np.cos(radian), -np.sin(radian), bias_h],
                 [-np.sin(radian), np.sin(radian), np.cos(radian), bias_d]],
                np.float16)
            theta_gm[index_batch, :, 0] = each_batch_theta.reshape((-1))
        feed_dict = {"theta": theta_gm}
        self.tik_inst.tikdb.start_debug(feed_dict, False)


def affine_grid3d(theta, grid, matrix_size, align_corners,
                  kernel_name="affine_grid", test=False):
    obj = AffineGrid3D(theta, grid, matrix_size, align_corners, kernel_name)
    obj.tiling_mode_select()
    switch = {
        1: obj.mode_split_n_compute,
    }
    switch[obj.mode]()
    if test:
        obj.tik_output_debug()
    return 0
