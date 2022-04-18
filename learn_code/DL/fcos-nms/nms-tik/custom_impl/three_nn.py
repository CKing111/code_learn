# -*- coding:utf-8 -*-
import numpy as np
from te import tik
from uti import interface_check
from uti import check
from point_cloud import point_dis_matrix
from version.get_version import get_aicore_container
from .get_threetop import GetThreetop


def conv32_to_16(tinst, ub_fp16, ub_fp32, idx_fp16, idx_fp32, num, max_repeat):
    # fp32->fp16, idx_fp16 is starting position
    max_num_per_cmd = 64
    max_num_per_loop = max_num_per_cmd * max_repeat
    loop = num // max_num_per_loop
    with tinst.if_scope(loop > 0):
        with tinst.for_range(0, loop) as loop_i:
            offset = loop_i * max_num_per_loop
            tinst.vconv(max_num_per_cmd, "none", ub_fp16[offset + idx_fp16],
                        ub_fp32[offset + idx_fp32], max_repeat, 1, 1, 4, 8)
    with tinst.else_scope():
        pass

    cur_offset = loop * max_num_per_loop
    repeat = (num - cur_offset) // max_num_per_cmd
    if (repeat > 0):
        tinst.vconv(max_num_per_cmd, "none", ub_fp16[cur_offset + idx_fp16],
                    ub_fp32[cur_offset + idx_fp32], repeat, 1, 1, 4, 8)

    cur_offset = cur_offset + repeat * max_num_per_cmd
    remain = num - cur_offset
    if (check.is_tik_dynamic(remain, tik)):
        with tinst.if_scope(remain > 0):
            tinst.vconv(remain, "none", ub_fp16[cur_offset + idx_fp16],
                        ub_fp32[cur_offset + idx_fp32], 1, 1, 1, 0, 0)
        with tinst.else_scope():
            pass
    else:
        if (remain > 0):
            tinst.vconv(remain, "none", ub_fp16[cur_offset + idx_fp16],
                        ub_fp32[cur_offset + idx_fp32], 1, 1, 1, 0, 0)


class ThreeNN(object):

    def __init__(self, container, ori_point, samp_point,
                 output_point, output_index, kernel_name):
        self.container = container
        self.tinst = self.container.tinst
        interface_check.check_kernelname(kernel_name)
        interface_check.check_param(ori_point, [4], ["float16"], ["NCHW"])
        interface_check.check_param(samp_point, [4], ["float16"], ["NCHW"])
        interface_check.check_param(output_point, [4], ["float16"], ["NCHW"])
        interface_check.check_param(output_index, [4], ["int32"], ["NCHW"])

        self.batch, self.ori_c, self.ori_h, self.ori_w = ori_point.get('shape')
        self.ori_num = self.ori_h * self.ori_w
        self.samp_num = samp_point.get("shape")[2] * samp_point.get("shape")[3]
        self.kernel_name = kernel_name
        self.aicore_use = container.const_aicore_num
        self._check_para()
        self._check_input_para(ori_point, samp_point)
        self._check_output(ori_point, output_point, output_index)

        # define input & output
        gm_scope = self.container.tik.scope_gm
        self.ori_xyz = self.tinst.Tensor(ori_point.get("dtype"), shape=(
            self.batch, self.ori_c, 1, self.ori_num), name="ori_xyz",
            scope=gm_scope)
        self.samp_xyz = self.tinst.Tensor(samp_point.get("dtype"), shape=(
            self.batch, self.ori_c, 1, self.samp_num), name="samp_xyz",
            scope=gm_scope)
        self.out_dist = self.tinst.Tensor(output_point.get("dtype"), shape=(
            self.batch, self.ori_c, 1, self.ori_num), name="out_dist",
            scope=gm_scope)
        self.out_idx = self.tinst.Tensor(output_index.get("dtype"), shape=(
            self.batch, self.ori_c, 1, self.ori_num), name="out_idx",
            scope=gm_scope)
        dis_dtype = "float32"
        self.dis_row, self.dis_col = self._calc_dis_matrix_row_col(dis_dtype)

    def _check_para(self):
        if (self.batch > 128) or (
                self.batch * self.ori_num * self.samp_num > 7658000000):
            raise RuntimeError(
                "batch must be less than 129 and batch * ori_num * samp_num"
                "must be less than 7658000001.")
        if (self.ori_num > 81920) or (self.ori_num % 16 != 0):
            raise RuntimeError(
                "ori_num must be less than 81921 and a multiple of 16.")
        if self.samp_num > 81920:
            raise RuntimeError("samp_num must be less than 81921.")
        if self.ori_c != 3:
            raise RuntimeError("ori_c must be equal to 3.")

        if (self.ori_h != 1) and (self.ori_w != 1):
            raise RuntimeError("one of ori_h and ori_w must be 1.")

    def _check_input_para(self, ori_point, samp_point):
        if ori_point.get("shape")[0] != samp_point.get("shape")[0]:
            raise RuntimeError("the zero dimension of ori_point must be equal"
                "to the zero dimension of samp_point.")

        if samp_point.get("shape")[1] != 3:
            raise RuntimeError("the one dimension of samp_point must be"
                "equal 3.")

        if (samp_point.get("shape")[2] != 1) and (
                samp_point.get("shape")[3] != 1):
            raise RuntimeError("one of the two dimension of samp_point and the"
                "two dimension of samp_point must be 1.")

        if (ori_point.get("shape")[2] != samp_point.get("shape")[2]) and \
                (ori_point.get("shape")[3] != samp_point.get("shape")[3]):
            raise RuntimeError("the two dimension of ori_point must be equal"
                "of the two dimension of samp_point or the three dimension of"
                "ori_point must be equal of the three dimension of"
                "samp_point.")

    def _check_output(self, ori_point, output_point, output_index):
        if output_point.get("shape") != ori_point.get("shape"):
            raise RuntimeError(
                "output_point_shape must be equal to ori_point_shape")
        if output_index.get("shape") != ori_point.get("shape"):
            raise RuntimeError(
                "output_index_shape must be equal to ori_point_shape")

    def _calc_dis_matrix_row_col(self, dis_dtype="float32"):
        factor = 1 if (dis_dtype == "float32") else 2
        row = min(self.ori_num // self.aicore_use, 1024)
        col = min(4096 * factor, self.samp_num)
        if (row % 16 != 0):
            row = (row + 16 - 1) // 16 * 16

        if (col % 16 != 0):
            col = (col + 16 - 1) // 16 * 16

        return row, col

    def aicore_in_use_select(self, num):
        self.num_each_core = self.dis_row
        self.aicore_use = int(np.ceil(num * 1.0 / self.num_each_core))
        self.num_last_core = num - self.num_each_core * (self.aicore_use - 1)
        if (self.aicore_use == 1):
            self.num_last_core = self.num_each_core

    def tilling_mode_select(self):
        self.mode = 1

    def global_init(self):
        pass

    def model_compute(self):
        self.index1 = self.tinst.Scalar("int32")
        self.index2 = self.tinst.Scalar("int32")
        self.start = self.tinst.Scalar("int32", name="core_start")
        self.end = self.tinst.Scalar("int32", name="core_end")
        with self.tinst.for_range(0, self.batch) as batch:
            self.aicore_in_use_select(self.ori_num)
            with self.tinst.for_range(0, self.aicore_use,
                                      block_num=self.aicore_use) as index:
                self._common_declare()
                self.top3_m.init_index(self.int_value, self.fp32_value, 6,
                                       self.tmp_index6_fp32, self.int_ub,
                                       self.fp16_ub)
                self.top3_m.init_index(self.int_value, self.fp32_value,
                                       self.dis_m.max_col_num, self.index_fp32,
                                       self.int_ub, self.fp16_ub)
                with self.tinst.if_scope(index != self.aicore_use - 1):
                    self.start.set_as(self.num_each_core * index)
                    self.end.set_as(self.num_each_core * (index + 1))
                    self.model_compute_each_core(batch, self.start, self.end)
                with self.tinst.else_scope():
                    self.start.set_as(self.num_each_core * index)
                    self.end.set_as(self.ori_num)
                    self.model_compute_each_core(batch, self.start, self.end)
        self.tinst.BuildCCE(kernel_name=self.kernel_name,
                            inputs=[self.ori_xyz, self.samp_xyz],
                            outputs=[self.out_dist, self.out_idx],
                            enable_l2=True)

    def _common_declare(self):
        dis_dtype = "float32"
        dis_row = self.dis_row
        dis_col = self.dis_col
        self.dis_m = point_dis_matrix.DisMatrix(self.container, dis_row,
                        dis_col, dis_dtype, b_fix_row=False,
                        b_gm_fp16=True, b_debug=True)
        self.top3_m = GetThreetop(self.container)
        # top3 prepare
        ub_scope = self.container.tik.scope_ubuf
        self.int_value = self.tinst.Scalar("int32")
        self.fp32_value = self.tinst.Scalar("float32")
        self.cmp_fp32 = self.tinst.Tensor("float32", (dis_col,),
                            name="cmp_fp32", scope=ub_scope)
        self.tmp_fp32 = self.tinst.Tensor("float32", (dis_col,),
                            name="tmp_fp32", scope=ub_scope)
        self.index_fp32 = self.tinst.Tensor("float32", (dis_col,),
                            name="index_fp32", scope=ub_scope)
        self.tmp_index6_fp32 = self.tinst.Tensor("float32", (6,),
                                name="tmp_index6_fp32", scope=ub_scope)
        self.int_ub = self.tinst.Tensor("int32", (16,), name="int_ub",
                        scope=ub_scope)
        self.fp16_ub = self.tinst.Tensor("float16", (16,), name="fp16_ub",
                        scope=ub_scope)

        # + 1 for tik compile reason
        self.tmp_col_dis = self.tinst.Tensor(dis_dtype, (dis_col + 1,),
                            name="tmp_col_dis", scope=ub_scope)
        self.three_col_dis = self.tinst.Tensor(dis_dtype, (6,),
                                name="three_col_dis", scope=ub_scope)
        self.result_idx = self.tinst.Tensor("int32", (3, dis_row),
                            name="result_idx", scope=ub_scope)
        self.result_value = self.tinst.Tensor("float32", (3, dis_row),
                                name="result_value", scope=ub_scope)
        self.result_value_fp16 = self.tinst.Tensor("float16", (3, dis_row),
                                    name="result_value", scope=ub_scope)
        self.tmp_idx = self.tinst.Tensor("int32", (3, dis_row), name="tmp_idx",
                            scope=ub_scope)
        self.tmp_value = self.tinst.Tensor("float32", (3, dis_row),
                            name="tmp_value", scope=ub_scope)

    def model_compute_each_core(self, batch, start, end):
        dis_row = self.dis_row
        dis_col = self.dis_col
        # update ori_xyz as row
        self.dis_m.update_xyz(self.ori_xyz, batch, start, end - start, True)
        col_loop = self.samp_num // dis_col
        col_remain = self.samp_num - col_loop * dis_col

        # first distance calculation and get top3
        if (col_loop > 0):
            self.dis_m.update_xyz(self.samp_xyz, batch, 0, dis_col, False)
            with self.tinst.for_range(0, end - start) as idx:
                self.dis_m.calc_dis_rowcol(idx, self.tmp_col_dis, True)
                self.top3_m.get_threetop(dis_col, self.tmp_col_dis,
                    self.cmp_fp32, self.tmp_fp32, self.index_fp32,
                    self.int_value, self.fp32_value, self.fp16_ub, self.int_ub,
                    self.result_value, self.result_idx, idx)
        if (col_loop > 1):
            with self.tinst.for_range(1, col_loop) as col_loop_i:
                col_bias = col_loop_i * dis_col
                # update samp_xyz as col
                self.dis_m.update_xyz(self.samp_xyz, batch, col_bias, dis_col,
                    False)
                with self.tinst.for_range(0, end - start) as idx:
                    self.dis_m.calc_dis_rowcol(idx, self.tmp_col_dis, True)
                    self.top3_m.get_threetop(dis_col, self.tmp_col_dis,
                        self.cmp_fp32, self.tmp_fp32, self.index_fp32,
                        self.int_value, self.fp32_value, self.fp16_ub,
                        self.int_ub, self.tmp_value, self.tmp_idx, idx)
                    self._compare_three(idx, col_bias)

        # remaining data processing
        if (col_remain > 0):
            self.top3_m.init_index(self.int_value, self.fp32_value, col_remain,
                self.index_fp32, self.int_ub, self.fp16_ub)
            col_bias = self.samp_num - col_remain
            self.dis_m.update_xyz(self.samp_xyz, batch, col_bias,
                col_remain, 0)
            if (col_loop == 0):
                with self.tinst.for_range(0, end - start) as idx:
                    self.dis_m.calc_dis_rowcol(idx, self.tmp_col_dis, True)
                    self.top3_m.get_threetop(col_remain, self.tmp_col_dis,
                        self.cmp_fp32, self.tmp_fp32, self.index_fp32,
                        self.int_value, self.fp32_value, self.fp16_ub,
                        self.int_ub, self.result_value, self.result_idx, idx)
            else:
                with self.tinst.for_range(0, end - start) as idx:
                    self.dis_m.calc_dis_rowcol(idx, self.tmp_col_dis, True)
                    self.top3_m.get_threetop(col_remain, self.tmp_col_dis,
                        self.cmp_fp32, self.tmp_fp32, self.index_fp32,
                        self.int_value, self.fp32_value, self.fp16_ub,
                        self.int_ub, self.tmp_value, self.tmp_idx, idx)
                    self._compare_three(idx, col_bias)
            self.top3_m.init_index(self.int_value, self.fp32_value, dis_col,
                self.index_fp32, self.int_ub, self.fp16_ub)

        # move distance and index to gm
        self._move_out_data_to_gm(dis_row, batch, start, end)

    def _compare_three(self, idx, col_bias):
        """
            take the smallest three of the six values
            :param idx: location of the current point to compare
            :param col_bias: offset of temporarily stored points
            :return
        """
        self.three_col_dis[0].set_as(self.result_value[0, idx])
        self.three_col_dis[1].set_as(self.result_value[1, idx])
        self.three_col_dis[2].set_as(self.result_value[2, idx])
        self.three_col_dis[3].set_as(self.tmp_value[0, idx])
        self.three_col_dis[4].set_as(self.tmp_value[1, idx])
        self.three_col_dis[5].set_as(self.tmp_value[2, idx])
        ub_scope = self.container.tik.scope_ubuf
        self.three_idx = self.tinst.Tensor("int32", (3, 1), ub_scope,
                            "three_idx")
        self.three_value = self.tinst.Tensor("float32", (3, 1), ub_scope,
                            "three_value")
        self.three_tmp_idx = self.tinst.Tensor("int32", (3,), ub_scope,
                                "three_tmp_idx")

        self.top3_m.get_threetop(6, self.three_col_dis, self.cmp_fp32,
            self.tmp_fp32, self.tmp_index6_fp32, self.int_value,
            self.fp32_value, self.fp16_ub, self.int_ub, self.three_value,
            self.three_idx, 0)

        # store the original three values
        self.three_tmp_idx[0].set_as(self.result_idx[0, idx])
        self.three_tmp_idx[1].set_as(self.result_idx[1, idx])
        self.three_tmp_idx[2].set_as(self.result_idx[2, idx])

        # assign the minimum three distances and indexes to their original
        # positions
        with self.tinst.for_range(0, 3) as r_idx:
            self.index1.set_as(self.three_idx[r_idx, 0])
            self.result_value[r_idx, idx].set_as(self.three_value[r_idx, 0])
            with self.tinst.if_scope(self.index1 < 3):
                self.result_idx[r_idx, idx].set_as(
                    self.three_tmp_idx[self.index1])
            with self.tinst.else_scope():
                self.index2.set_as(self.tmp_idx[self.index1 - 3, idx])
                self.result_idx[r_idx, idx].set_as(self.index2 + col_bias)

    def _move_out_data_to_gm(self, dis_row, batch, start, end):
        conv32_to_16(self.tinst, self.result_value_fp16, self.result_value, 0,
                     0, 3 * dis_row, 255)
        value_burst = (end - start) // 16
        idx_burst = (end - start) // 8
        value_dst_stride = (self.ori_num - end + start) // 16
        idx_dst_stride = (self.ori_num - end + start) // 8
        value_src_stride = (dis_row - end + start) // 16
        idx_src_stride = (dis_row - end + start) // 8
        self.tinst.data_move(self.out_dist[batch, 0, 0, start],
                             self.result_value_fp16, 0, 3, value_burst,
                             value_src_stride, value_dst_stride)
        self.tinst.data_move(self.out_idx[batch, 0, 0, start], self.result_idx,
                             0, 3, idx_burst, idx_src_stride,
                             idx_dst_stride)

    def tik_output_debug(self):
        input1 = np.ones((self.batch, 1, self.ori_num, 3), dtype=np.float16)
        input2 = np.ones((self.batch, 1, self.samp_num, 3), dtype=np.float16)
        i = 0
        for i_0 in range(0, self.batch):
            for i_1 in range(0, self.ori_num):
                for i_2 in range(0, 3):
                    i = i % 6 + 1
                    input1[i_0, 0, i_1, i_2] = i

        j = 0
        for j_0 in range(0, self.batch):
            for j_1 in range(0, self.samp_num):
                j = j % 6 + 1
                for j_2 in range(0, 3):
                    input2[j_0, 0, j_1, j_2] = j

        input1 = input1.transpose(0, 3, 1, 2)
        input2 = input2.transpose(0, 3, 1, 2)
        feed_dict = {"ori_xyz": input1, "samp_xyz": input2}
        out_dist, out_idx = self.tinst.tikdb.start_debug(feed_dict=feed_dict,
                                                         interactive=False)
        print("self.out_dist:\n{}".format(out_dist))
        print("self.out_idx:\n{}".format(out_idx))


def three_nn(ori_point, samp_point, output_point, output_index,
             kernel_name="threenn", test=False):
    """
     Parameters
     ----------
     kernel_name : kernel name, default value is "threenn"
     function_description : Find out the minimum three values and corresponding
     index of the distance from each point in point cloud A to each point in
     point cloud B
     ori_xyz: first feature map with the shape [batch, 3, 1, ori_num]
     samp_xyz: second feature map with the shape [batch, 3, 1, samp_num]
     output: distance and index
     Returns
     -------
     None

     """
    container = get_aicore_container(("Ascend610",), c3x_support_list=())

    obj = ThreeNN(container, ori_point, samp_point, output_point, output_index,
                  kernel_name)

    obj.tilling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    obj.global_init()

    switch = {
        1: obj.model_compute,
    }

    switch[obj.mode]()
    if not test:
        return 0

    obj.tik_output_debug()
    return 0
