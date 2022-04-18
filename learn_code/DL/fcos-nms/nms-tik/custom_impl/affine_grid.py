from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from uti import interface_check
from version import get_version


class AffineGrid:
    """
    2d affine grid
    """

    def __init__(self, theta, grid, matrix_size, align_corners, kernel_name):
        interface_check.check_kernelname(kernel_name)
        self._check_params(theta, grid)
        self._check_attrs(matrix_size, align_corners)
        self._check_params_shape(theta, grid, matrix_size, align_corners)
        self.cont = get_version.get_aicore_container(("Ascend610", ))

        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ai_core_use = self.cont.const_aicore_num
        self.mode = 0

        self.kernel_name = kernel_name
        self.batch_num, self.matrix_size, self.theta_num = self._split_shape(
            matrix_size)
        self.align_corners = align_corners
        self.dim_start, self.dim_stride = self._get_start_stride(
            self.align_corners, self.matrix_size)

        self.dtype = theta.get("dtype")
        self.size = self.cont.const_dtype_byte.get(self.dtype)
        self.transpose_num = self.cont.get_c0_num(self.dtype)
        self.fp32_type = "float32"
        self.fp32_size = self.cont.const_dtype_byte.get(self.fp32_type)
        self.fp32_block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(
            self.fp32_type)

        self.theta_shape = (self.batch_num, self.theta_num, 16)
        self.theta = self.tik_inst.Tensor(self.dtype, self.theta_shape,
                                          self.tik.scope_gm, "theta")
        self.grid = self.tik_inst.Tensor(self.dtype, grid.get("shape"),
                                         self.tik.scope_gm, "grid")

    @staticmethod
    def _check_params(theta, grid):
        interface_check.check_param(theta, [5], ["float16"], ["NC1HWC0"])
        interface_check.check_param(grid, [5], ["float16"], ["NC1HWC0"])

    @staticmethod
    def _check_attrs(matrix_size, align_corners):
        if not isinstance(matrix_size, tuple) and not isinstance(matrix_size,
                                                                 list):
            raise RuntimeError("[ERROR] matrix_size type is not supported")
        for dim_num in matrix_size:
            if not isinstance(dim_num, int) or dim_num <= 0:
                raise RuntimeError("[ERROR] matrix_size dims smaller than 1")
        if not isinstance(align_corners, bool):
            raise RuntimeError("[ERROR] align_corners type is not supported")

    @staticmethod
    def _check_params_shape(theta, grid, matrix_size, align_corners):
        if len(matrix_size) != 4:
            raise RuntimeError("[ERROR] matrix_size length is not supported")
        batch, channel, height, width = matrix_size
        if theta.get("shape") != (batch, 1, 2, 3, 16):
            raise RuntimeError("[ERROR] shape of theta is not supported")
        if align_corners and (height == 1 or width == 1):
            raise RuntimeError("[ERROR] matrix_size dims equal 1")
        if (batch * height * width > 20000000 or
                height > 10000 or width > 10000):
            raise RuntimeError("[ERROR] matrix_size is too larger")
        if grid.get("shape") != (batch, 1, height, width, 16):
            raise RuntimeError("[ERROR] shape of grid is not supported")

    @staticmethod
    def _split_shape(matrix_size_shape):
        """
        get shape info
        Args:
            matrix_size_shape: tuple(n, c, h, w)
        Returns: batch_num, matrix_size
        """
        theta_num = 6
        matrix_size_dict = dict()
        batch_num = matrix_size_shape[0]
        matrix_size_dict["h"] = matrix_size_shape[2]
        matrix_size_dict["w"] = matrix_size_shape[3]
        return batch_num, matrix_size_dict, theta_num

    @staticmethod
    def _get_start_stride(align_corners, matrix_size):
        """
        get start coord, stride length
        Args:
            align_corners: if corners align
            matrix_size: matrix info
        Returns: dim_start, dim_stride
        """
        dim_stride = dict()
        dim_start = dict()
        if align_corners:
            for dim_name, dim_num in matrix_size.items():
                dim_stride[dim_name] = 2.0 / (dim_num - 1)
                dim_start[dim_name] = -1.0
        else:
            for dim_name, dim_num in matrix_size.items():
                dim_stride[dim_name] = 2.0 / dim_num
                dim_start[dim_name] = -1.0 + dim_stride[dim_name] / 2.0
        return dim_start, dim_stride

    @staticmethod
    def _get_format_num(input_num, format_num, ceil=True):
        if ceil:
            result = (input_num + format_num - 1) // format_num * format_num
        else:
            result = input_num // format_num * format_num
        return result

    @staticmethod
    def _get_loop_info(all_data_num, each_loop_num):
        loop_times = (all_data_num + each_loop_num - 1) // each_loop_num
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return last_loop_num, loop_times

    @staticmethod
    def _ceil_div(dividend, divisor):
        return (dividend + divisor - 1) // divisor

    def tiling_mode_select(self):
        self.mode = 1

    def mode_split_n_compute(self):
        """
        each loop count one batch
        """
        each_core_batch_num = self._ceil_div(self.batch_num, self.ai_core_use)
        last_core_batch_num, self.ai_core_use = self._get_loop_info(
            self.batch_num, each_core_batch_num)
        with self.tik_inst.for_range(0, self.ai_core_use,
                                     block_num=self.ai_core_use) as core_index:
            batch_index = each_core_batch_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                self._mode_split_n_compute_each_core(
                    batch_index, each_core_batch_num)
            with self.tik_inst.else_scope():
                self._mode_split_n_compute_each_core(
                    batch_index, last_core_batch_num)
        self.tik_inst.BuildCCE(inputs=[self.theta], outputs=[self.grid],
                               kernel_name=self.kernel_name)

    def _mode_split_n_compute_each_core(self, batch_index_start,
                                        each_core_batch_num):
        """
        Args:
            batch_index_start: batch_index start
            each_core_batch_num: batch num
        """
        # each core count once base grid
        width, height = self.matrix_size.get("w"), self.matrix_size.get("h")
        grid_base_l1 = self._init_l1_tensor(
            self.fp32_type, self.fp32_block_data_num, [width, height],
            "grid_base")
        grid_base_width_l1, grid_base_height_l1 = grid_base_l1
        self._get_base_grid(grid_base_width_l1, width, self.dim_start.get("w"),
                            self.dim_stride.get("w"))
        self._get_base_grid(grid_base_height_l1, height,
                            self.dim_start.get("h"), self.dim_stride.get("h"))
        # each loop count one batch
        thread_num = 1 if each_core_batch_num == 1 else 2
        ub_size = self.cont.const_ub_max_byte // thread_num
        with self.tik_inst.for_range(0, each_core_batch_num,
                                     thread_num=thread_num) as batch_index:
            batch_index_now = batch_index_start + batch_index
            self._mode1_compute_each_loop(grid_base_l1, batch_index_now,
                                          ub_size)

    def _mode1_compute_each_loop(self, grid_base_l1, batch_index, ub_size):
        grid_base_width_l1, grid_base_height_l1 = grid_base_l1
        width, height = self.matrix_size.get("w"), self.matrix_size.get("h")
        # init l1 grid;
        l1_data_num_all = [width, width, height, height]
        grid_l1 = self._init_l1_tensor(
            self.fp32_type, self.fp32_block_data_num, l1_data_num_all,
            "grid_mul")

        theta_scalar = [self.tik_inst.Scalar(self.fp32_type) for _ in
                        range(self.theta_num)]
        self._get_theta_scalar(theta_scalar, batch_index)
        # width mul scalar[0] add scalar[2]; width mul scalar[3] add scalar[5]
        # height mul scalar[1]; height mul scalar[4]
        grid_width_mul_l1 = grid_l1[:2]
        grid_height_mul_l1 = grid_l1[2:]
        weight_scalar_all = (theta_scalar[0], theta_scalar[3])
        height_scalar_all = (theta_scalar[1], theta_scalar[4])
        bias_scalar_all = (theta_scalar[2], theta_scalar[5])

        self._get_l1_grid(width, grid_base_width_l1, grid_width_mul_l1,
                          weight_scalar_all, bias_scalar_all)
        self._get_l1_grid(height, grid_base_height_l1, grid_height_mul_l1,
                          height_scalar_all)
        self._transpose_split_weight(grid_l1, ub_size, batch_index)

    def _init_l1_tensor(self, data_type, format_num, data_num_all, name):
        """
        init len(data_num_all) l1 tensor
        """
        l1_tensor_all = [None] * len(data_num_all)
        for index_i, data_num in enumerate(data_num_all):
            data_num_format = self._get_format_num(data_num, format_num)
            tensor_shape = (data_num_format,)
            tensor_name = name + "_{}_{}".format(data_type, index_i)
            l1_tensor_all[index_i] = self.tik_inst.Tensor(data_type,
                                                          tensor_shape,
                                                          self.tik.scope_cbuf,
                                                          tensor_name)
        return l1_tensor_all

    def _get_base_grid(self, grid_base_l1, data_num, start, stride):
        """
        data type: fp32
        Args:
            grid_base_l1: l1 tensor, save result
            data_num: base grid num
            start: base grid start coord
            stride: each coord stride
        """
        with self.tik_inst.new_stmt_scope():
            block_num = self._ceil_div(data_num, self.fp32_block_data_num)
            data_num_format = self._get_format_num(data_num,
                                                   self.fp32_block_data_num)
            grid_base_ub = self.tik_inst.Tensor(
                self.fp32_type, (data_num_format,), self.tik.scope_ubuf,
                "grid_base_ub")
            self._get_range_num(grid_base_ub, start, stride, data_num)
            self.tik_inst.data_move(grid_base_l1, grid_base_ub, 0, 1,
                                    block_num, 0, 0)

    def _get_range_num(self, grid_base_ub, start, stride, data_num):
        """
        Args:
            grid_base_ub: ub tensor
            start: grid start
            stride: grid stride
            data_num: grid num
        """
        # init min num data
        padded_num = min(self.fp32_block_data_num, data_num)
        self.tik_inst.vector_dup(padded_num, grid_base_ub, start, 1, 1, 8)
        for pad_index in range(1, padded_num):
            mask_pad = 2 ** pad_index
            self.tik_inst.vadds([0, mask_pad], grid_base_ub, grid_base_ub,
                                stride * pad_index, 1, 1, 1, 8, 8)
        # pad other data
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        self._pad_num(grid_base_ub, padded_num, stride, data_num, num_per_cmd)

    def _pad_num(self, grid_base_ub, padded_num, stride, data_num,
                 num_per_cmd):
        """
        recursive pad grid_base_ub
        Args:
            grid_base_ub: ub tensor
            padded_num: already padded num
            stride: grid stride
            data_num: total amount to pad
            num_per_cmd: fp32 num_per_cmd
        """
        if padded_num < data_num:
            tensor_len = grid_base_ub.shape[0]
            last_num = tensor_len - padded_num
            if last_num > padded_num:
                padding_num = padded_num
            else:
                padding_num = last_num
            buf_pad_all = {
                "tensor_src": AVecBuf(grid_base_ub, padding_num, 0, self.cont,
                                      False, num_per_cmd),
                "tensor_dst": AVecBuf(grid_base_ub, padding_num, padded_num,
                                      self.cont, False, num_per_cmd), }
            cmd_pad_tensor = [VecGCmd("vadds", "tensor_dst", "tensor_src",
                                      scalar=stride * padded_num)]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_pad_all, cmd_pad_tensor,
                                        "tensor_src")
            self._pad_num(grid_base_ub, padded_num + padding_num, stride,
                          data_num, num_per_cmd)

    def _get_theta_scalar(self, theta_scalar, batch_index):
        """
        init theta scalar, fp32
        Args:
            theta_scalar: scalar list
            batch_index: batch index
        """
        with self.tik_inst.new_stmt_scope():
            theta_ub = self.tik_inst.Tensor(
                self.dtype, (self.theta_num, 16), self.tik.scope_ubuf,
                "theta_ub")
            theta_ub_fp32 = self.tik_inst.Tensor(
                self.fp32_type, (self.theta_num, 16), self.tik.scope_ubuf,
                "theta_ub_fp32")
            block_num = \
                self.theta_num * 16 * self.size // self.cont.const_block_byte
            self.tik_inst.data_move(theta_ub, self.theta[batch_index, 0, 0], 0,
                                    1, block_num, 0, 0)
            self.tik_inst.vconv(1, "", theta_ub_fp32, theta_ub, self.theta_num,
                                1, 1, 2, 1)
            for index_i in range(self.theta_num):
                theta_scalar[index_i].set_as(theta_ub_fp32[index_i, 0])

    def _get_l1_grid(self, data_num, grid_src_l1, grid_dst_l1_all,
                     weight_scalar_all, bias_scalar_all=None):
        """
        get result: grid_base * weight + bias
        Args:
            data_num: grid num
            grid_src_l1: l1 tensor, base grid
            grid_dst_l1_all: l1 tensor, grid result
            weight_scalar_all: scalars, weights
            bias_scalar_all: scalars, bias
        """
        with self.tik_inst.new_stmt_scope():
            mask_temp = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
            block_num_fp32 = self._ceil_div(data_num, self.fp32_block_data_num)
            repeat_num = self._ceil_div(data_num, mask_temp)
            data_num_format = repeat_num * mask_temp
            ub_shape = (data_num_format,)
            grid_src_ub = self.tik_inst.Tensor(self.fp32_type, ub_shape,
                                               self.tik.scope_ubuf,
                                               "grid_base_ub")
            grid_dst_ub = self.tik_inst.Tensor(self.fp32_type, ub_shape,
                                               self.tik.scope_ubuf,
                                               "grid_dst_ub")

            self.tik_inst.data_move(grid_src_ub, grid_src_l1, 0, 1,
                                    block_num_fp32, 0, 0)
            if bias_scalar_all is None:
                for grid_dst_l1, weight_scalar in zip(grid_dst_l1_all,
                                                      weight_scalar_all):
                    self.tik_inst.vmuls(mask_temp, grid_dst_ub, grid_src_ub,
                                        weight_scalar, repeat_num, 1, 1, 8, 8)
                    self.tik_inst.data_move(grid_dst_l1, grid_dst_ub, 0, 1,
                                            block_num_fp32, 0, 0)
            else:
                for grid_dst_l1, weight_scalar, bias_scalar in zip(
                        grid_dst_l1_all, weight_scalar_all, bias_scalar_all):
                    self.tik_inst.vmuls(mask_temp, grid_dst_ub, grid_src_ub,
                                        weight_scalar, repeat_num, 1, 1, 8, 8)
                    self.tik_inst.vadds(mask_temp, grid_dst_ub, grid_dst_ub,
                                        bias_scalar, repeat_num, 1, 1, 8, 8)
                    self.tik_inst.data_move(grid_dst_l1, grid_dst_ub, 0, 1,
                                            block_num_fp32, 0, 0)

    def _get_transpose_max_num(self, ub_size):
        data_size = float(self.size)
        fp32_size = float(self.fp32_size)

        ub_size_last = ub_size - self.cont.const_block_byte * 2
        each_data_ub_size = (
                    self.transpose_num * data_size * 2 +
                    fp32_size * self.theta_num +
                    fp32_size / self.transpose_num)
        data_num_max = int(ub_size_last / each_data_ub_size)
        data_num_format = self._get_format_num(data_num_max,
                                               self.transpose_num, False)
        return data_num_format

    def _transpose_split_weight(self, grid_l1, ub_size, batch_index):
        """
        split width
        """
        max_num_each_loop = self._get_transpose_max_num(ub_size)
        width = self.matrix_size.get("w")
        width_format = self._get_format_num(width, self.transpose_num)
        each_loop_width_num = min(width_format, max_num_each_loop)
        last_loop_width_num, width_loop_num = \
            self._get_loop_info(width, each_loop_width_num)
        with self.tik_inst.for_range(0, width_loop_num) as width_index:
            w_start = width_index * each_loop_width_num
            with self.tik_inst.if_scope(width_index != width_loop_num - 1):
                self._transpose_split_height(grid_l1, max_num_each_loop,
                                             batch_index, w_start,
                                             each_loop_width_num)
            with self.tik_inst.else_scope():
                self._transpose_split_height(grid_l1, max_num_each_loop,
                                             batch_index, w_start,
                                             last_loop_width_num)

    def _init_transpose_tensor(self, dims_one, dims_two):
        dims_one_format = self._get_format_num(dims_one,
                                               self.fp32_block_data_num)
        dims_one_format += self.fp32_block_data_num
        transpose_src_ub = self.tik_inst.Tensor(
            self.dtype, (16, dims_one, dims_two), self.tik.scope_ubuf,
            "transpose_src_ub")
        transpose_dst_ub = self.tik_inst.Tensor(
            self.dtype, (dims_one, dims_two, 16), self.tik.scope_ubuf,
            "transpose_dst_ub")
        add_grid_ub = self.tik_inst.Tensor(
            self.fp32_type, (self.theta_num, dims_one, dims_two),
            self.tik.scope_ubuf, "add_grid_ub")
        height_ub = self.tik_inst.Tensor(self.fp32_type, (dims_one_format,),
                                         self.tik.scope_ubuf, "height_ub")
        return transpose_dst_ub, transpose_src_ub, add_grid_ub, height_ub

    def _transpose_split_height(self, grid_l1, max_num_each_loop, batch_index,
                                w_start, width):
        """
        split height
        """
        height = self.matrix_size.get("h")
        width_format = self._get_format_num(width, self.transpose_num)
        each_loop_height_num = max_num_each_loop // width_format
        each_loop_height_num = min(each_loop_height_num, height)
        transpose_dst_ub, transpose_src_ub, add_grid_ub, height_ub = \
            self._init_transpose_tensor(each_loop_height_num, width_format)

        dim_num, l1_index_start, add_grid_index_start = 2, 0, 2
        self._pad_width(
            grid_l1, add_grid_ub, dim_num, l1_index_start,
            add_grid_index_start,
            each_loop_height_num, w_start, width)
        # split height
        last_loop_height_num, height_loop_num = \
            self._get_loop_info(height, each_loop_height_num)
        with self.tik_inst.for_range(0, height_loop_num) as height_index:
            h_start = height_index * each_loop_height_num
            with self.tik_inst.if_scope(height_index != height_loop_num - 1):
                self._transpose_2d(
                    transpose_dst_ub, transpose_src_ub, add_grid_ub, height_ub,
                    grid_l1, batch_index,
                    w_start, width, h_start, each_loop_height_num)
            with self.tik_inst.else_scope():
                self._transpose_2d(
                    transpose_dst_ub, transpose_src_ub, add_grid_ub, height_ub,
                    grid_l1, batch_index,
                    w_start, width, h_start, last_loop_height_num)

    def _transpose_2d(self, transpose_dst_ub, transpose_src_ub, add_grid_ub,
                      height_ub, grid_l1,
                      batch_index, w_start, width, h_start, height):
        each_loop_depth_num = 1
        dim_num, l1_index_start, add_grid_index_start = 2, 2, 4
        self._pad_height(
            grid_l1, add_grid_ub, height_ub, dim_num, l1_index_start,
            add_grid_index_start,
            each_loop_depth_num, h_start, height, width)

        width_format = add_grid_ub.shape[2]
        grid_num = height * width_format
        data_num = add_grid_ub.shape[1] * width_format
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        buf_add_all = {
            "grid_width_result": AVecBuf(transpose_src_ub, grid_num, 0,
                                         self.cont, False, num_per_cmd),
            "grid_height_result": AVecBuf(transpose_src_ub, grid_num, data_num,
                                          self.cont, False, num_per_cmd),
            "grid_width": AVecBuf(add_grid_ub, grid_num, 0, self.cont, False,
                                  num_per_cmd),
            "grid_height": AVecBuf(add_grid_ub, grid_num, data_num, self.cont,
                                   False, num_per_cmd),
            "grid_width_0": AVecBuf(add_grid_ub, grid_num, data_num * 2,
                                    self.cont, False, num_per_cmd),
            "grid_width_1": AVecBuf(add_grid_ub, grid_num, data_num * 3,
                                    self.cont, False, num_per_cmd),
            "grid_height_0": AVecBuf(add_grid_ub, grid_num, data_num * 4,
                                     self.cont, False, num_per_cmd),
            "grid_height_1": AVecBuf(add_grid_ub, grid_num, data_num * 5,
                                     self.cont, False, num_per_cmd), }
        cmd_add_tensor = [
            VecGCmd(cmd_name="vadd", dst_name="grid_width",
                    src0_name="grid_width_0", src1_name="grid_height_0"),
            VecGCmd(cmd_name="vadd", dst_name="grid_height",
                    src0_name="grid_width_1", src1_name="grid_height_1"),
            VecGCmd(cmd_name="vconv", dst_name="grid_width_result",
                    src0_name="grid_width", round_mode=""),
            VecGCmd(cmd_name="vconv", dst_name="grid_height_result",
                    src0_name="grid_height", round_mode="")]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_add_all, cmd_add_tensor,
                                    "grid_width")

        repeat_num = grid_num // 16
        self._vnchwconv(transpose_dst_ub, transpose_src_ub, repeat_num)
        self.tik_inst.data_move(
            self.grid[batch_index, 0, h_start, w_start, 0], transpose_dst_ub,
            0, height, width,
            max(0, width_format - width),
            max(0, self.matrix_size.get("w") - width))

    def _vnchwconv(self, transpose_dst_ub, transpose_src_ub, repeat_num):
        dst_list = [transpose_dst_ub[0, index_i, 0] for index_i in range(16)]
        src_list = [transpose_src_ub[index_i, 0, 0] for index_i in range(16)]
        if repeat_num == 1:
            dst_rep_stride = 0
            src_rep_stride = 0
        else:
            dst_rep_stride = 16
            src_rep_stride = 1

        repeat_num_max = 255
        repeat_num_last, transpose_loop_num = \
            self._get_loop_info(repeat_num, repeat_num_max)
        for repeat_index in range(transpose_loop_num):
            if_loop = (repeat_index != (transpose_loop_num - 1))
            repeat_num_loop = repeat_num_max if if_loop else repeat_num_last
            self.tik_inst.vnchwconv(True, True, dst_list, src_list,
                                    repeat_num_loop, dst_rep_stride,
                                    src_rep_stride)

    def _get_format_start_end(self, start_index, data_num, format_num):
        """
        get start, end align format_num
        """
        start_index_format = self._get_format_num(start_index, format_num,
                                                  False)
        start_index_ub = start_index - start_index_format
        end_index_format = self._get_format_num(start_index + data_num,
                                                format_num)
        data_num_format = end_index_format - start_index_format
        block_num = data_num_format // format_num
        return start_index_format, start_index_ub, block_num

    def _pad_width(self, grid_l1, add_grid_ub, dim_num, l1_index_start,
                   add_grid_index_start,
                   height, w_start, width):
        """
        pad width data
        """
        width_block_num = self._ceil_div(width, self.fp32_block_data_num)
        for dim_index in range(dim_num):
            l1_index = l1_index_start + dim_index
            add_grid_index = add_grid_index_start + dim_index
            self.tik_inst.data_move(
                add_grid_ub[add_grid_index, 0, 0], grid_l1[l1_index][w_start],
                0, 1, width_block_num, 0, 0)
            if height > 1:
                with self.tik_inst.for_range(1, height) as h_d_index:
                    self.tik_inst.data_move(
                        add_grid_ub[add_grid_index, h_d_index, 0],
                        add_grid_ub[add_grid_index, 0, 0],
                        0, 1, width_block_num, 0, 0)

    def _pad_height(self, grid_l1, add_grid_ub, height_ub, dim_num,
                    l1_index_start, add_grid_index_start,
                    depth, h_start, height, width):
        """
        pad height data
        """
        height_scalar = self.tik_inst.Scalar(self.fp32_type)
        h_start_format, h_index_start, height_block_num = \
            self._get_format_start_end(h_start, height,
                                       self.fp32_block_data_num)

        width_format = self._get_format_num(width, self.transpose_num)
        each_batch_data_num = add_grid_ub.shape[1] * add_grid_ub.shape[2]
        h_w_block_num = height * width_format // self.fp32_block_data_num
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        for dim_index in range(dim_num):
            l1_index = l1_index_start + dim_index
            add_grid_index = add_grid_index_start + dim_index
            self.tik_inst.data_move(height_ub,
                                    grid_l1[l1_index][h_start_format], 0, 1,
                                    height_block_num, 0, 0)
            with self.tik_inst.for_range(0, height) as height_index:
                height_scalar.set_as(height_ub[h_index_start + height_index])
                buf_dup_all = {
                    "tensor_dst": AVecBuf(
                        add_grid_ub, width_format,
                        (each_batch_data_num * add_grid_index +
                         height_index * width_format),
                        self.cont, False, num_per_cmd)}
                cmd_dup_tensor = [
                    VecGCmd(cmd_name="vector_dup", dst_name="tensor_dst",
                            scalar=height_scalar)]
                VecGExecutor.exec_vec_g_cmd(self.cont, buf_dup_all,
                                            cmd_dup_tensor, "tensor_dst")
            if depth > 1:
                with self.tik_inst.for_range(1, depth) as depth_index:
                    self.tik_inst.data_move(
                        add_grid_ub[add_grid_index, depth_index * height, 0],
                        add_grid_ub[add_grid_index, 0, 0],
                        0, 1, h_w_block_num, 0, 0)

    def tik_output_debug(self):
        return self.tik_inst


def affine_grid(theta, grid, matrix_size, align_corners,
                kernel_name="affine_grid", test=False):
    obj = AffineGrid(theta, grid, matrix_size, align_corners, kernel_name)
    obj.tiling_mode_select()
    switch = {
        1: obj.mode_split_n_compute,
    }
    switch[obj.mode]()
    if test:
        obj.tik_output_debug()
    return 0
