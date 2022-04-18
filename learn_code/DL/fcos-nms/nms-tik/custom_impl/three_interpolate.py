# coding:utf8
import numpy as np
from uti import interface_check
from version.get_version import get_aicore_container

# the multi-core design is based on dual core design
AI_COER_NUM = 2


class ThreeInterp:
    """
     Parameters
     ----------
     kernel_name : kernel name, default value is "ThreeInterp"
     function_description : According to the values of the three points and the
     corresponding weights of the three points, the inverse distance weighted
     interpolation is obtained
     input1: first feature map with the shape [input_n, input_c, 1, input1_w]
     input2: second feature map with the shape [input_n, 3, 1, input2_w]
     input3: third feature map with the shape [input_n, 3, 1, input2_w]
     output: dict shape and dtype of output
     Returns
     -------
     None

     """

    def __init__(self, container, point, idx_data, weight, output,
                 kernel_name="threeinterp"):
        self.container = container
        self.tik_inst = self.container.tinst
        interface_check.check_kernelname(kernel_name)
        interface_check.check_param(point, [4], ["float16"], ["NCHW"])
        interface_check.check_param(idx_data, [4], ["int32"], ["NCHW"])
        interface_check.check_param(weight, [4], ["float16"], ["NCHW"])
        interface_check.check_param(output, [4], ["float16"], ["NCHW"])
        self.point_shape = point.get("shape")
        self.idx_data_shape = idx_data.get("shape")
        self.weight_shape = weight.get("shape")
        self.output_shape = output.get("shape")
        self._check_input(point, idx_data, weight)
        self._check_para()
        self._check_output()

        self.input_n = self.point_shape[0]  # batch
        self.input1_c = self.point_shape[1]  # number of point
        # number of interpolated
        self.input1_w = self.point_shape[2] * self.point_shape[3]
        # number of interpolation# number of interpolation
        self.input2_w = self.idx_data_shape[2] * self.idx_data_shape[3]
        self.kernel_name = kernel_name

        self.aicore_use = AI_COER_NUM
        self._aicore_in_use_select(self.input_n * self.input2_w)

        # input/output gm buffer
        gm_scope = self.container.tik.scope_gm
        self.points_gm = self.tik_inst.Tensor(point.get("dtype"), (
            self.input_n, self.input1_c, 1, self.input1_w), name="points_gm",
            scope=gm_scope)
        self.idx_data_gm = self.tik_inst.Tensor(idx_data.get("dtype"), (
            self.input_n, 3, 1, self.input2_w), name="idx_data_gm",
            scope=gm_scope)
        self.weight_gm = self.tik_inst.Tensor(weight.get("dtype"), (
            self.input_n, 3, 1, self.input2_w), name="weight_gm",
            scope=gm_scope)
        self.output_gm = self.tik_inst.Tensor(output.get("dtype"), (
            self.input_n, self.input1_c, 1, self.input2_w), name="output_gm",
            scope=gm_scope)

    def _check_input(self, point, idx_data, weight):
        if (point.get("shape")[0] > 128):
            raise RuntimeError("input1_n must be less than 129")
        if (point.get("shape")[1] > 2048):
            raise RuntimeError("input1_c must be less than 2049")
        if (idx_data.get("shape")[2] * idx_data.get("shape")[3] < 16):
            raise RuntimeError("input2_w must greater than 15")
        if (point.get("shape")[1] == 1 and point.get("shape")[2] *
                point.get("shape")[3] == 1):
            raise RuntimeError("input1_c and input1_w can not be both 1")
        if (idx_data.get("shape")[1] != 3):
            raise RuntimeError("idx_data'C must be equal to 3")
        if (idx_data.get("shape") != weight.get("shape")):
            raise RuntimeError("idx_data_shape must be equal to weight_shape")

    def _check_para(self):
        if (self.point_shape[2] != 1) and (self.point_shape[3] != 1):
            raise RuntimeError(
                "one of point'H and point'W must be 1")
        if (self.idx_data_shape[2] != 1) and (self.idx_data_shape[3] != 1):
            raise RuntimeError(
                "one of idx_data'H and idx_data'W must be 1")
        if (self.point_shape[0] != self.idx_data_shape[0]) or (
                self.idx_data_shape[0] != self.weight_shape[0]):
            raise RuntimeError(
                "point'N, idx_data'N, weight'N must be equal")

        if (self.point_shape[2] != self.weight_shape[2]) and (
                self.point_shape[3] != self.weight_shape[3]):
            raise RuntimeError(
                "point'H must be equal weight'H or "
                "point'W must be equal to weight'W.")

    def _check_output(self):
        if self.output_shape[0] != self.point_shape[0]:
            raise RuntimeError("output'N must be equal point'N")
        if self.output_shape[1] != self.point_shape[1]:
            raise RuntimeError("output'C must be equal point'C")
        if self.output_shape[2] != self.idx_data_shape[2]:
            raise RuntimeError("output'H must be equal idx_data'H")
        if self.output_shape[3] != self.idx_data_shape[3]:
            raise RuntimeError("output'W must be equal idx_data'W")

    def _aicore_in_use_select(self, tilling_len):
        self.slen_each_core = (tilling_len + self.aicore_use - 1) //\
                              self.aicore_use
        self.slen_last_core = tilling_len - self.slen_each_core * (
                self.aicore_use - 1)
        self.aicore_use = (tilling_len + self.slen_each_core - 1) // \
                          self.slen_each_core
        if (self.aicore_use == 1):
            self.slen_last_core = self.slen_each_core

    def tilling_mode_select(self):
        mini_offset = (self.input1_c + 16 - 1) // 16 * 16
        # 2 is the number of vnchwconv tensor, 5 is the number of point tensor
        points_data = mini_offset * 16 * 2 * \
                      self.container.const_dtype_byte.get("float16") + \
                      mini_offset * 5 * \
                      self.container.const_dtype_byte.get("float16")

        # 3 is the number of threeinterp point
        idx_data = 3 * self.input2_w * \
                   self.container.const_dtype_byte.get("int32")
        weight_data = 3 * self.input2_w * \
                      self.container.const_dtype_byte.get("float16")
        if ((points_data + idx_data + weight_data) <
                self.container.const_ub_max_byte and (
                mini_offset * self.input1_w * 2) <=
                self.container.const_l1_max_byte and self.input1_w % 16 == 0
                and self.input2_w % 16 == 0):
            self.mode = 1
        else:
            self.mode = 2

    def global_init(self):
        pass

    def mode1_compute(self):
        if (self.input2_w % 32 != 0 and self.input_n % 2 != 0):
            start = 0
            end = self.input_n * self.input2_w
            self._model1_compute_each_core(start, end)
        else:
            with self.tik_inst.for_range(0, self.aicore_use,
                                         block_num=self.aicore_use) as index:
                with self.tik_inst.if_scope(index != self.aicore_use - 1):
                    start = self.slen_each_core * index
                    end = self.slen_each_core * (index + 1)
                    self._model1_compute_each_core(start, end)
                with self.tik_inst.else_scope():
                    start = self.slen_each_core * index
                    end = self.input_n * self.input2_w
                    self._model1_compute_each_core(start, end)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.points_gm, self.idx_data_gm,
                                       self.weight_gm],
                               outputs=[self.output_gm], enable_l2=True)

    def _model1_compute_each_core(self, start, end):
        mini_offset = (self.input1_c + 16 - 1) // 16 * 16
        # 3 is to get 3 points
        ub_scope = self.container.tik.scope_ubuf
        cb_scope = self.container.tik.scope_cbuf
        idx_int_ub = self.tik_inst.Tensor("int32", (3, self.input2_w),
                        name="idx_int_ub", scope=ub_scope)
        weight_ub = self.tik_inst.Tensor("float16", (3, self.input2_w),
                        name="weight_ub", scope=ub_scope)
        points1_l1 = self.tik_inst.Tensor("float16", (self.input1_w,
                        mini_offset), name="points1_l1", scope=cb_scope)
        points1_ub = self.tik_inst.Tensor("float16", (mini_offset, 16),
                        name="points1_ub", scope=ub_scope)
        points2_ub = self.tik_inst.Tensor("float16", (16, mini_offset),
                        name="points2_ub", scope=ub_scope)
        points3_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points3_ub", scope=ub_scope)
        points4_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points4_ub", scope=ub_scope)
        points5_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points5_ub", scope=ub_scope)
        points6_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points6_ub", scope=ub_scope)
        points7_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points7_ub", scope=ub_scope)

        i1_scalar = self.tik_inst.Scalar("int32")
        i2_scalar = self.tik_inst.Scalar("int32")
        i3_scalar = self.tik_inst.Scalar("int32")
        w1_scalar = self.tik_inst.Scalar("float16")
        w2_scalar = self.tik_inst.Scalar("float16")
        w3_scalar = self.tik_inst.Scalar("float16")

        with self.tik_inst.for_range(start, end) as index:
            # n is multiple of 16
            with self.tik_inst.if_scope(index == start):
                self._data_process_ceil(index, idx_int_ub, weight_ub,
                    points1_ub, points2_ub, points1_l1, mini_offset)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(index % self.input2_w == 0):
                    self._data_process_ceil(index, idx_int_ub, weight_ub,
                        points1_ub, points2_ub, points1_l1, mini_offset)
                with self.tik_inst.else_scope():
                    pass

            self._mode1_compute_one_set(index, i1_scalar, i2_scalar, i3_scalar,
                w1_scalar, w2_scalar, w3_scalar, idx_int_ub, weight_ub)

            self._mode1_compute_one_loop(index, i1_scalar, i2_scalar,
                i3_scalar, w1_scalar, w2_scalar, w3_scalar, points3_ub,
                points1_l1, points4_ub, points5_ub, points6_ub,
                points7_ub, points2_ub, mini_offset)

            # process according to each c * h * W surface
            with self.tik_inst.if_scope((index + 1) % 16 == 0):
                self._model1_move_data_to_gm(index, points2_ub, points1_ub,
                                             mini_offset)
            with self.tik_inst.else_scope():
                pass

    def _mode1_compute_one_set(self, index, i1_scalar, i2_scalar, i3_scalar,
            w1_scalar, w2_scalar, w3_scalar, idx_int_ub, weight_ub):
        i1_scalar.set_as(idx_int_ub[0, index % self.input2_w])
        i2_scalar.set_as(idx_int_ub[1, index % self.input2_w])
        i3_scalar.set_as(idx_int_ub[2, index % self.input2_w])
        w1_scalar.set_as(weight_ub[0, index % self.input2_w])
        w2_scalar.set_as(weight_ub[1, index % self.input2_w])
        w3_scalar.set_as(weight_ub[2, index % self.input2_w])

    def _mode1_compute_one_loop(self, index, i1_scalar, i2_scalar, i3_scalar,
            w1_scalar, w2_scalar, w3_scalar, points3_ub, points1_l1,
            points4_ub, points5_ub, points6_ub, points7_ub, points2_ub,
                                 mini_offset):
        # m is 16 times and non 16 times
        self.tik_inst.data_move(points3_ub, points1_l1[i1_scalar, 0], 0, 1,
                                mini_offset // 16, 0, 0)
        # calculate the first interp
        self._vmuls_compute(self.input1_c, 0, points3_ub, w1_scalar,
                            points4_ub)

        self.tik_inst.data_move(points3_ub, points1_l1[i2_scalar, 0], 0, 1,
                                mini_offset // 16, 0, 0)
        # calculate the second interp
        self._vmuls_compute(self.input1_c, 0, points3_ub, w2_scalar,
                            points5_ub)

        self.tik_inst.data_move(points3_ub, points1_l1[i3_scalar, 0], 0, 1,
                                mini_offset // 16, 0, 0)
        # calculate the third interp
        self._vmuls_compute(self.input1_c, 0, points3_ub, w3_scalar,
                            points6_ub)
        self._vadd_compute(0, self.input1_c, points4_ub, points5_ub,
                           points7_ub)
        k = index % self.input2_w
        self._vadd_compute(k % 16, self.input1_c, points7_ub, points6_ub,
                           points2_ub)

    def _model1_move_data_to_gm(self, index, points2_ub, points1_ub,
                                mini_offset):
        # transfer the output point out
        self._vnchwconv_compute(points2_ub, points1_ub, 16, mini_offset)
        self.tik_inst.data_move(
            self.output_gm[(index - 15) // self.input2_w, 0, 0, (index - 15) %
            self.input2_w], points1_ub, 0, self.input1_c,
            16 // 16, 0, (self.input2_w - 16) // 16)

    def _data_process_ceil(self, loop_index, idx_int, weight, points1, points2,
                           points_tmp, data_offset):
        self.tik_inst.data_move(idx_int, self.idx_data_gm[
            loop_index // self.input2_w, 0, 0, 0], 0, 1,
            3 * self.input2_w // 8, 0, 0)
        self.tik_inst.data_move(weight, self.weight_gm[
            loop_index // self.input2_w, 0, 0, 0], 0, 1,
            3 * self.input2_w // 16, 0, 0)

        vnchw_loop_time = self.input1_w // 16
        with self.tik_inst.for_range(0, vnchw_loop_time) as i_w:
            self.tik_inst.data_move(points1, self.points_gm[
                loop_index // self.input2_w, 0, 0, 16 * i_w], 0,
                self.input1_c, 16 // 16, (self.input1_w - 16) // 16, 0)
            # transpose input points
            self._vnchwconv_compute(points1, points2, data_offset, 16)
            self.tik_inst.data_move(points_tmp[16 * i_w, 0], points2, 0, 1,
                data_offset, 0, 0)

    def _vnchwconv_compute(self, points_src, points_dst, src_dim_one,
                           dst_dim_one):
        vnchw_repeat_time = src_dim_one * dst_dim_one // 256
        if (dst_dim_one == 16 and src_dim_one == 16):
            dst_rep_stride = 0
            src_rep_stride = 0
        elif (src_dim_one == 16):
            dst_rep_stride = 16
            src_rep_stride = 1
        else:
            dst_rep_stride = 1
            src_rep_stride = 16
        src_list = [points_src[k, 0] for k in range(16)]
        dst_list = [points_dst[k, 0] for k in range(16)]
        self.tik_inst.vnchwconv(True, True, dst_list, src_list,
                                vnchw_repeat_time, dst_rep_stride,
                                src_rep_stride)

    def _vmuls_compute(self, data_num, i_h, points_src_ub, w1_src,
                       points_dst_ub):
        vmuls_loop_time = data_num // (255 * 128)
        if (vmuls_loop_time > 0):
            with self.tik_inst.for_range(0, vmuls_loop_time) as i_vmul:
                self.tik_inst.vmuls(128, points_dst_ub[0, 255 * 128 * i_vmul],
                                    points_src_ub[i_h, 255 * 128 * i_vmul],
                                    w1_src, 255, 1, 1, 8, 8)

        vmuls_repeat_time = data_num % (255 * 128) // 128
        if (vmuls_repeat_time > 0):
            interp_offset = 255 * 128 * vmuls_loop_time
            self.tik_inst.vmuls(128, points_dst_ub[0, interp_offset],
                                points_src_ub[i_h, interp_offset], w1_src,
                                vmuls_repeat_time, 1, 1, 8, 8)

        # process the left data
        vmuls_left_data = data_num % (255 * 128) % 128
        if (vmuls_left_data > 0):
            interp_offset = 255 * 128 * vmuls_loop_time + 128 *\
                            vmuls_repeat_time
            self.tik_inst.vmuls(vmuls_left_data,
                                points_dst_ub[0, interp_offset],
                                points_src_ub[i_h, interp_offset],
                                w1_src, 1, 1, 1, 1, 1)

    def _vadd_compute(self, i_x, data_num, points_src1_ub, points_src2_ub,
                      points_dst_ub):
        vadd_loop_time = data_num // (255 * 128)
        if (vadd_loop_time > 0):
            with self.tik_inst.for_range(0, vadd_loop_time) as i_vadd:
                self.tik_inst.vadd(128, points_dst_ub[i_x, 255 * 128 * i_vadd],
                                   points_src1_ub[0, 255 * 128 * i_vadd],
                                   points_src2_ub[0, 255 * 128 * i_vadd], 255,
                                   1, 1, 1, 8, 8, 8)

        vadd_repeat_time = (data_num % (255 * 128)) // 128
        if (vadd_repeat_time > 0):
            interp_offset = 255 * 128 * vadd_loop_time
            self.tik_inst.vadd(128, points_dst_ub[i_x, interp_offset],
                               points_src1_ub[0, interp_offset],
                               points_src2_ub[0, interp_offset],
                               vadd_repeat_time, 1, 1, 1, 8, 8, 8)

        # process the left data
        vadd_left_data = data_num % (255 * 128) % 128
        if (vadd_left_data > 0):
            interp_offset = 255 * 128 * vadd_loop_time + 128 * vadd_repeat_time
            self.tik_inst.vadd(vadd_left_data,
                               points_dst_ub[i_x, interp_offset],
                               points_src1_ub[0, interp_offset],
                               points_src2_ub[0, interp_offset], 1, 1, 1, 1, 8,
                               8, 8)

    def mode2_compute(self):
        if (self.input2_w % 32 != 0 and self.input_n % 2 != 0):
            start = 0
            end = self.input_n * self.input2_w
            self._model2_compute_each_core(start, end)
        else:
            with self.tik_inst.for_range(0, self.aicore_use,
                                         block_num=self.aicore_use) as index:
                with self.tik_inst.if_scope(index != self.aicore_use - 1):
                    start = self.slen_each_core * index
                    end = self.slen_each_core * (index + 1)
                    self._model2_compute_each_core(start, end)
                with self.tik_inst.else_scope():
                    start = self.slen_each_core * index
                    end = self.input_n * self.input2_w
                    self._model2_compute_each_core(start, end)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.points_gm, self.idx_data_gm,
                                       self.weight_gm],
                               outputs=[self.output_gm], enable_l2=True)

    def _model2_compute_each_core(self, start, end):
        mini_offset = (self.input1_c // 16) * 16 + (
            0 if self.input1_c % 16 == 0 else 16)
        # 3 is to get 3 points and 16 is a block
        ub_scope = self.container.tik.scope_ubuf
        cb_scope = self.container.tik.scope_cbuf
        idx_int_ub = self.tik_inst.Tensor("int32", (3, 16), name="idx_int_ub",
                        scope=ub_scope)
        weight_ub = self.tik_inst.Tensor("float16", (3, 16), name="weight_ub",
                        scope=ub_scope)
        points_tmp_l1 = self.tik_inst.Tensor("float16", (17, mini_offset),
                        name="points_tmp_l1", scope=cb_scope)
        points1_ub = self.tik_inst.Tensor("float16", (mini_offset, 16),
                        name="points1_ub", scope=ub_scope)
        points2_ub = self.tik_inst.Tensor("float16", (16, mini_offset),
                        name="points2_ub", scope=ub_scope)
        points3_ub = self.tik_inst.Tensor("float16", (16, mini_offset),
                        name="points3_ub", scope=ub_scope)
        points4_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points4_ub", scope=ub_scope)
        points5_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points5_ub", scope=ub_scope)
        points6_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points6_ub", scope=ub_scope)
        points7_ub = self.tik_inst.Tensor("float16", (1, mini_offset),
                        name="points7_ub", scope=ub_scope)
        i_1 = self.tik_inst.Scalar("int32")
        i_2 = self.tik_inst.Scalar("int32")
        i_3 = self.tik_inst.Scalar("int32")
        w_1 = self.tik_inst.Scalar("float16")
        w_2 = self.tik_inst.Scalar("float16")
        w_3 = self.tik_inst.Scalar("float16")

        with self.tik_inst.for_range(start, end) as index:
            self._model2_data_process(index, i_1, i_2, i_3, w_1, w_2, w_3,
                                      idx_int_ub, weight_ub)

            # find out the interval of index divided by every 16 numbers
            idx1_mod = i_1 // 16 * 16
            idx1_left = i_1 % 16
            if (self.input1_w % 16 == 0):
                self.tik_inst.data_move(points1_ub, self.points_gm[
                    index // self.input2_w, 0, 0, idx1_mod], 0,
                    self.input1_c, 16 // 16, (self.input1_w - 16) // 16, 0)
            else:
                with self.tik_inst.for_range(0, self.input1_c) as i:
                    self.tik_inst.data_move(points1_ub[i, 0], self.points_gm[
                        index // self.input2_w, i, 0, idx1_mod], 0,
                        1, 16 // 16, 0, 0)

            self._model2_compute_one_loop(index, i_2, i_3, idx1_left,
                idx1_mod, w_1, w_2, w_3, mini_offset, points1_ub, points2_ub,
                points4_ub, points5_ub, points6_ub, points7_ub, points3_ub)

            self._model2_move_data_to_gm(index, points3_ub, points1_ub,
                                         points_tmp_l1, mini_offset)

    def _model2_compute_one_loop(self, index, i_2, i_3, idx1_left, idx1_mod,
        w_1, w_2, w_3, mini_offset, points1_ub, points2_ub, points4_ub,
        points5_ub, points6_ub, points7_ub, points3_ub):
        # transpose the data of the corresponding interval
        self._vnchwconv_compute(points1_ub, points2_ub, mini_offset, 16)
        self._vmuls_compute(self.input1_c, idx1_left, points2_ub, w_1,
                            points4_ub)
        idx2_mod = i_2 // 16 * 16
        idx2_left = i_2 % 16
        self._compute_ceil(index, idx2_mod, idx2_left, idx1_mod, w_2,
                           points2_ub, points5_ub, points1_ub, mini_offset)
        idx3_mod = i_3 // 16 * 16
        idx3_left = i_3 % 16
        self._compute_ceil(index, idx3_mod, idx3_left, idx2_mod, w_3,
                           points2_ub, points6_ub, points1_ub, mini_offset)
        self._vadd_compute(0, self.input1_c, points4_ub, points5_ub,
                           points7_ub)
        k = index % self.input2_w
        self._vadd_compute(k % 16, self.input1_c, points7_ub, points6_ub,
                           points3_ub)

    def _model2_data_process(self, index, i_1, i_2, i_3, w_1, w_2, w_3,
                             idx_int_ub, weight_ub):
        if (self.input2_w % 16 == 0):
            with self.tik_inst.if_scope(index % 16 == 0):
                self.tik_inst.data_move(idx_int_ub, self.idx_data_gm[
                    index // self.input2_w, 0, 0, index % self.input2_w], 0, 3,
                    16 // 8, (self.input2_w - 16) // 8, 0)
                self.tik_inst.data_move(weight_ub, self.weight_gm[
                    index // self.input2_w, 0, 0, index % self.input2_w],
                    0, 3, 16 // 16, (self.input2_w - 16) // 16, 0)
            with self.tik_inst.else_scope():
                pass

            i_1.set_as(idx_int_ub[0, index % 16])
            i_2.set_as(idx_int_ub[1, index % 16])
            i_3.set_as(idx_int_ub[2, index % 16])
            w_1.set_as(weight_ub[0, index % 16])
            w_2.set_as(weight_ub[1, index % 16])
            w_3.set_as(weight_ub[2, index % 16])
        else:
            with self.tik_inst.for_range(0, 3) as i:
                self.tik_inst.data_move(idx_int_ub[i, 0],
                    self.idx_data_gm[index // self.input2_w, i, 0, index %
                    self.input2_w], 0, 1, 16 // 8, 0, 0)
                self.tik_inst.data_move(weight_ub[i, 0],
                    self.weight_gm[index // self.input2_w, i, 0, index %
                    self.input2_w], 0, 1, 16 // 16, 0, 0)
            i_1.set_as(idx_int_ub[0, 0])
            i_2.set_as(idx_int_ub[1, 0])
            i_3.set_as(idx_int_ub[2, 0])
            w_1.set_as(weight_ub[0, 0])
            w_2.set_as(weight_ub[1, 0])
            w_3.set_as(weight_ub[2, 0])

    def _model2_move_data_to_gm(self, index, points3_ub, points1_ub,
                                points_tmp_l1, mini_offset):
        """
        move data to gm
        :param index: the current number of cycles
        :param point3_ub: the point tensor. shape:(16, mini_offset)
        :param point1_ub: the point tensor. shape:(mini_offset, 16)
        :param points_tmp_l1: the point tensor. shape:(17, mini_offset)
        :param mini_offset: integral multiple of 16
        :return
        """
        mini_left = self.input2_w % 16
        # process according to each c * h * W surface
        if (mini_left == 0):
            with self.tik_inst.if_scope((index + 1) % 16 == 0):
                self._vnchwconv_compute(points3_ub, points1_ub, 16,
                    mini_offset)
                # back 15 times to data_move out
                self.tik_inst.data_move(self.output_gm[(index - 15) //
                    self.input2_w, 0, 0, (index - 15) % self.input2_w],
                    points1_ub, 0, self.input1_c, 16 // 16, 0,
                    (self.input2_w - 16) // 16)
            with self.tik_inst.else_scope():
                pass
        else:
            # transpose every 16 cycles
            with self.tik_inst.if_scope((index % self.input2_w + 1) % 16 == 0):
                self._vnchwconv_compute(points3_ub, points1_ub, 16,
                    mini_offset)
                with self.tik_inst.for_range(0, self.input1_c) as i:
                    self.tik_inst.data_move(self.output_gm[(index - 15) //
                    self.input2_w, i, 0, (index - 15) % self.input2_w],
                    points1_ub[i, 0], 0, 1, 16 // 16, 0, 0)
            with self.tik_inst.else_scope():
                pass
            # move (16 - mini_left) pieces of data L1
            with self.tik_inst.if_scope((index % self.input2_w + 1 + mini_left)
                                        % self.input2_w == 0):
                self.tik_inst.data_move(points_tmp_l1, points3_ub[mini_left,
                    0], 0, 1, (16 - mini_left) * mini_offset // 16, 0, 0)
            with self.tik_inst.else_scope():
                pass

            with self.tik_inst.if_scope((index + 1) % self.input2_w == 0):
                # move mini_left pieces of data L1
                self.tik_inst.data_move(points_tmp_l1[16 - mini_left, 0],
                    points3_ub, 0, 1, mini_left * mini_offset // 16, 0, 0)
                # splicing data and moving to UB
                self.tik_inst.data_move(points3_ub, points_tmp_l1, 0, 1,
                    mini_offset, 0, 0)
                self._vnchwconv_compute(points3_ub, points1_ub, 16,
                    mini_offset)
                # move tail data out
                with self.tik_inst.for_range(0, self.input1_c) as i:
                    self.tik_inst.data_move(self.output_gm[(index + 1 - 16) //
                        self.input2_w, i, 0, (index + 1 - 16) % self.input2_w],
                        points1_ub[i, 0], 0, 1, 16 // 16, 0, 0)
            with self.tik_inst.else_scope():
                pass

    def _compute_ceil(self, loop_index, idx_mod, idx_left, front_mod, weight,
                      tensor2, tensor6, tensor1, data_len):
        """
        caluate the second and third interpolation
        :param loop_index: the current number of cycles
        :param idx_mod: the range of the index of the current point
        :param idx_left: the specific position of the index of the
                         current point in the interval
        :param front_mod: the range of the index of the previous point
        :param weight: weight corresponding to point
        :param tensor2: the point tensor. shape:(16, mini_offset)
        :param tensor6: the point tensor. shape:(1, mini_offset)
        :param tensor1: the point tensor. shape:(mini_offset, 16)
        :param data_len: number of points at index
        :return
        """
        with self.tik_inst.if_scope(idx_mod == front_mod):
            self._vmuls_compute(self.input1_c, idx_left, tensor2, weight,
                                tensor6)
        with self.tik_inst.else_scope():
            # if the index is not in the previous range, it needs to be moved
            # into UB again
            if (self.input1_w % 16 == 0):
                self.tik_inst.data_move(tensor1, self.points_gm[
                    loop_index // self.input2_w, 0, 0, idx_mod], 0,
                    self.input1_c, 16 // 16, (self.input1_w - 16) // 16, 0)
            else:
                with self.tik_inst.for_range(0, self.input1_c) as i:
                    self.tik_inst.data_move(tensor1[i, 0], self.points_gm[
                        loop_index // self.input2_w, i, 0, idx_mod],
                        0, 1, 16 // 16, 0, 0)
            self._vnchwconv_compute(tensor1, tensor2, data_len, 16)
            self._vmuls_compute(self.input1_c, idx_left, tensor2, weight,
                                tensor6)


def three_interpolate(point, idx_data, weight, output,
                      kernel_name="three_interpolate", test=False):
    container = get_aicore_container(("Ascend610",), c3x_support_list=())

    obj = ThreeInterp(container, point, idx_data, weight, output, kernel_name)

    obj.tilling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    obj.global_init()

    switch = {
        1: obj.mode1_compute,
        2: obj.mode2_compute
    }

    switch[obj.mode]()
    if not test:
        return 0
    else:
        return obj.tinst
