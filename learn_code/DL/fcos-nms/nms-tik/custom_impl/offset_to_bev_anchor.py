from uti import interface_check
from version import get_version


class OffsetToBevAnchor:

    def __init__(self, offsets, anchors, bev_anchor, regressed_anchor,
                 bev_extents, kernel_name):
        """
        Args:
            offsets: NC1HWC0 (n, 1, 1, 1, 16) float16
            anchors: ND (n, 6) float16
            bev_anchor: ND (n, 4) float16
            regressed_anchor: ND (n , 6) float16
            bev_extents: (min_x, max_x, min_z, max_z) Tuple[float]
            kernel_name:
        """
        interface_check.check_kernelname(kernel_name)
        self._check_attr(bev_extents)
        self._check_params(offsets, anchors, bev_anchor, regressed_anchor)
        self._check_params_shape(offsets, anchors, bev_anchor,
                                 regressed_anchor)
        self.cont = get_version.get_aicore_container(("Ascend610",))
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.kernel_name = kernel_name
        self.ub_size = self.cont.const_ub_max_byte
        self.ai_core_use = self.cont.const_aicore_num
        self.repeat_time_max = self.cont.const_vector_proc_max_rpt
        self.data_type = offsets.get("dtype")
        self.data_size, self.block_data_num, self.repeat_data_num = \
            self._get_type_const(self.data_type)

        (self.bev_x_extents_min, self.bev_x_extents_max,
         self.bev_z_extents_min, self.bev_z_extents_max) = bev_extents
        self.anchor_num = anchors.get("shape")[0]
        self.c_offsets = 16
        self.c_anchors = 6
        self.c_bev_anchor = 4
        self.c_regressed_anchor = 6
        offsets_shape = (self.anchor_num, self.c_offsets)
        anchors_shape = (self.anchor_num, self.c_anchors)
        bev_anchor_shape = (self.anchor_num, self.c_bev_anchor)
        regressed_anchor_shape = (self.anchor_num, self.c_regressed_anchor)
        self.offsets = self.tik_inst.Tensor(
            offsets.get("dtype"), offsets_shape,
            self.tik.scope_gm, "offsets")
        self.anchors = self.tik_inst.Tensor(
            anchors.get("dtype"), anchors_shape,
            self.tik.scope_gm, "anchors")
        self.bev_anchor = self.tik_inst.Tensor(
            bev_anchor.get("dtype"), bev_anchor_shape,
            self.tik.scope_gm, "bev_anchor")
        self.regressed_anchor = self.tik_inst.Tensor(
            regressed_anchor.get("dtype"), regressed_anchor_shape,
            self.tik.scope_gm, "regressed_anchor")

    def _check_attr(self, bev_extents):
        if not isinstance(bev_extents, (list, tuple)) or len(bev_extents) != 4:
            raise RuntimeError("bev_extents is not supported")
        all_float = [isinstance(extent, float) for extent in bev_extents]
        if not all(all_float):
            raise RuntimeError("bev_extents is not supported")
        min_x, max_x, min_y, max_y = bev_extents
        if min_x >= max_x or min_y >= max_y:
            raise RuntimeError("bev_extents is not supported")

    def _check_params(self, offsets, anchors, bev_anchor, regressed_anchor):
        interface_check.check_param(offsets, [5], ["float16"], ["NC1HWC0"])
        interface_check.check_param(anchors, [2], ["float16"],
                                    ["NCHW", "ND", "NHWC"])
        interface_check.check_param(bev_anchor, [2], ["float16"],
                                    ["NCHW", "ND", "NHWC"])
        interface_check.check_param(regressed_anchor, [2], ["float16"],
                                    ["NCHW", "ND", "NHWC"])

    def _check_params_shape(self, offsets, anchors, bev_anchor,
                            regressed_anchor):
        offsets_shape = offsets.get("shape")
        anchors_shape = anchors.get("shape")
        bev_anchor_shape = bev_anchor.get("shape")
        regressed_anchor_shape = regressed_anchor.get("shape")
        boxes_num = anchors_shape[0]
        if boxes_num > 1000000:
            raise RuntimeError("boxes_num is too larger")
        if offsets_shape != (boxes_num, 1, 1, 1, 16):
            raise RuntimeError("offsets_shape is not supported")
        if anchors_shape != (boxes_num, 6):
            raise RuntimeError("anchors_shape is not supported")
        if bev_anchor_shape != (boxes_num, 4):
            raise RuntimeError("bev_anchor_shape is not supported")
        if regressed_anchor_shape != (boxes_num, 6):
            raise RuntimeError("regressed_anchor_shape is not supported")

    def _get_type_const(self, data_type):
        data_size = self.cont.const_dtype_byte.get(data_type)
        block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self.cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    def _ceil_div(self, dividend, divisor):
        result = (dividend + divisor - 1) // divisor
        return result

    def _get_loop_info(self, all_data_num, each_loop_num):
        loop_times = self._ceil_div(all_data_num, each_loop_num)
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return loop_times, last_loop_num

    def _get_align_num(self, input_num, format_num, ceil=True):
        if ceil:
            result = self._ceil_div(input_num, format_num) * format_num
        else:
            result = input_num // format_num * format_num
        return result

    def mode1_compute(self):
        each_core_anchor_num = self._ceil_div(
            self.anchor_num, self.ai_core_use)
        each_core_anchor_num = self._get_align_num(
            each_core_anchor_num, self.repeat_data_num)
        self.ai_core_use, last_core_anchor_num = self._get_loop_info(
            self.anchor_num, each_core_anchor_num)
        with self.tik_inst.for_range(0, self.ai_core_use,
                                     block_num=self.ai_core_use) as core_index:
            anchor_index = each_core_anchor_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                self._mode1_compute_each_core(anchor_index,
                                              each_core_anchor_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_core(anchor_index,
                                              last_core_anchor_num)
        self.tik_inst.BuildCCE(inputs=[self.offsets, self.anchors],
                               outputs=[self.bev_anchor,
                                        self.regressed_anchor],
                               kernel_name=self.kernel_name)

    def _get_each_loop_anchor_num(self, anchor_num):
        each_loop_data_num_max = self.repeat_data_num * self.repeat_time_max
        align_num = self.repeat_data_num
        if anchor_num <= align_num:
            return align_num
        loop_time = 2
        ub_size = self.ub_size // loop_time
        each_anchor_data_size = self.data_size * (
                    self.c_regressed_anchor + self.c_offsets * 2)
        each_loop_anchor_num = ub_size // each_anchor_data_size
        each_loop_anchor_num = self._get_align_num(each_loop_anchor_num,
                                                   align_num, False)
        each_loop_anchor_num = min(each_loop_anchor_num,
                                   each_loop_data_num_max)
        return each_loop_anchor_num

    def _mode1_compute_each_core(self, anchor_index_start, anchor_num):
        each_loop_anchor_num = self._get_each_loop_anchor_num(anchor_num)
        loop_num, last_loop_anchor_num = self._get_loop_info(
            anchor_num, each_loop_anchor_num)
        with self.tik_inst.for_range(0, loop_num,
                                     thread_num=min(2, loop_num)) \
                as loop_index:
            anchor_index = (each_loop_anchor_num * loop_index +
                            anchor_index_start)
            with self.tik_inst.if_scope(loop_index != loop_num - 1):
                self._mode1_compute_each_loop(anchor_index,
                                              each_loop_anchor_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_loop(anchor_index,
                                              last_loop_anchor_num)

    def _mode1_compute_each_loop(self, anchor_index, anchor_num):
        anchor_num_format = self._get_align_num(
            anchor_num, self.repeat_data_num)
        # 0:6 anchors 6:12 offset
        regressed_anchor_ub = self.tik_inst.Tensor(
            self.data_type, (self.c_regressed_anchor, anchor_num_format),
            self.tik.scope_ubuf, "regressed_anchor_ub")
        self._offset_to_anchor(regressed_anchor_ub, anchor_index, anchor_num)
        bev_anchor_ub = self.tik_inst.Tensor(
            self.data_type, (self.c_bev_anchor, anchor_num_format),
            self.tik.scope_ubuf, "regressed_anchor_ub")
        self._project_to_bev(bev_anchor_ub, regressed_anchor_ub, anchor_num)
        self._result_move_out(bev_anchor_ub, regressed_anchor_ub, anchor_index,
                              anchor_num)

    def _data_move_in(self, tensor_out, tensor_ub, anchor_index, anchor_num,
                      channel_num):
        anchor_num_format = self._get_align_num(
            anchor_num, self.repeat_data_num)
        block_num_move_in = self._ceil_div(anchor_num * channel_num,
                                           self.block_data_num)
        with self.tik_inst.new_stmt_scope():
            data_move_in_ub = self.tik_inst.Tensor(
                self.data_type, (anchor_num_format, channel_num),
                self.tik.scope_ubuf, "data_move_in_ub")
            self.tik_inst.data_move(
                data_move_in_ub, tensor_out[anchor_index, 0],
                0, 1, block_num_move_in, 0, 0)
            self.tik_inst.v4dtrans(
                False, tensor_ub, data_move_in_ub,
                anchor_num_format, channel_num)

    def _offset_to_anchor(self, regressed_anchor_ub, anchor_index, anchor_num):
        repeat_num = self._ceil_div(anchor_num, self.repeat_data_num)
        anchor_num_format = self._get_align_num(
            anchor_num, self.repeat_data_num)
        with self.tik_inst.new_stmt_scope():
            offsets_ub = self.tik_inst.Tensor(
                self.data_type, (self.c_offsets, anchor_num_format),
                self.tik.scope_ubuf, "offsets_ub")
            self._data_move_in(self.offsets, offsets_ub, anchor_index,
                               anchor_num, self.c_offsets)
            anchors_ub = self.tik_inst.Tensor(
                self.data_type, (self.c_anchors, anchor_num_format),
                self.tik.scope_ubuf, "anchors_ub")
            self._data_move_in(self.anchors, anchors_ub, anchor_index,
                               anchor_num, self.c_anchors)

            self._compute_pred(regressed_anchor_ub, anchors_ub, offsets_ub,
                               0, 3, 0, 0, repeat_num)
            self._compute_pred(regressed_anchor_ub, anchors_ub, offsets_ub,
                               1, 4, 1, 1, repeat_num)
            self._compute_pred(regressed_anchor_ub, anchors_ub, offsets_ub,
                               2, 5, 2, 2, repeat_num)
            self._compute_d_pred(regressed_anchor_ub, anchors_ub, offsets_ub,
                                 3, 3, 3, repeat_num)
            self._compute_d_pred(regressed_anchor_ub, anchors_ub, offsets_ub,
                                 4, 4, 4, repeat_num)
            self._compute_d_pred(regressed_anchor_ub, anchors_ub, offsets_ub,
                                 5, 5, 5, repeat_num)

    def _compute_pred(self, regressed_anchor_ub, anchors_ub, offsets_ub,
                      regressed_anchor_index, anchor_index_0, anchor_index_1,
                      offset_index, repeat_num):
        mask = self.repeat_data_num
        self.tik_inst.vmul(
            mask, regressed_anchor_ub[regressed_anchor_index, 0],
            anchors_ub[anchor_index_0, 0], offsets_ub[offset_index, 0],
            repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(
            mask, regressed_anchor_ub[regressed_anchor_index, 0],
            regressed_anchor_ub[regressed_anchor_index, 0],
            anchors_ub[anchor_index_1, 0], repeat_num, 1, 1, 1, 8, 8, 8)

    def _compute_d_pred(self, regressed_anchor_ub, anchors_ub, offsets_ub,
                        regressed_anchor_index, anchor_index,
                        offset_index, repeat_num):
        mask = self.repeat_data_num
        self.tik_inst.vec_ln(
            mask, regressed_anchor_ub[regressed_anchor_index, 0],
            anchors_ub[anchor_index, 0], repeat_num, 8, 8)
        self.tik_inst.vadd(
            mask, regressed_anchor_ub[regressed_anchor_index, 0],
            regressed_anchor_ub[regressed_anchor_index, 0],
            offsets_ub[offset_index, 0], repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vec_exp(
            mask, regressed_anchor_ub[regressed_anchor_index, 0],
            regressed_anchor_ub[regressed_anchor_index, 0], repeat_num, 8, 8)

    def _project_to_bev(self, bev_anchor_ub, regressed_anchor_ub, anchor_num):
        repeat_num = self._ceil_div(anchor_num, self.repeat_data_num)
        anchor_num_format = self._get_align_num(
            anchor_num, self.repeat_data_num)
        with self.tik_inst.new_stmt_scope():
            temp_ub = self.tik_inst.Tensor(
                self.data_type, (anchor_num_format,),
                self.tik.scope_ubuf, "temp_ub")
            self._compute_x(bev_anchor_ub, regressed_anchor_ub, temp_ub,
                            repeat_num)
            self._compute_z(bev_anchor_ub, regressed_anchor_ub, temp_ub,
                            repeat_num)
            self._compute_norm(bev_anchor_ub, repeat_num)

    def _compute_x(self, bev_anchor_ub, regressed_anchor_ub, half_dim_x,
                   repeat_num):
        mask = self.repeat_data_num
        self.tik_inst.vmuls(
            mask, half_dim_x, regressed_anchor_ub[3, 0], 0.5,
            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vsub(
            mask, bev_anchor_ub[0, 0], regressed_anchor_ub[0, 0],
            half_dim_x, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(
            mask, bev_anchor_ub[2, 0], regressed_anchor_ub[0, 0],
            half_dim_x, repeat_num, 1, 1, 1, 8, 8, 8)

    def _compute_z(self, bev_anchor_ub, regressed_anchor_ub, half_dim_z,
                   repeat_num):
        mask = self.repeat_data_num
        self.tik_inst.vmuls(
            mask, half_dim_z, regressed_anchor_ub[5, 0], 0.5,
            repeat_num, 1, 1, 8, 8)
        self.tik_inst.vadd(
            mask, bev_anchor_ub[1, 0], regressed_anchor_ub[2, 0],
            half_dim_z, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(
            mask, bev_anchor_ub[3, 0], regressed_anchor_ub[2, 0],
            half_dim_z, repeat_num, 1, 1, 1, 8, 8, 8)
        # bev_z_extents_max
        self.tik_inst.vector_dup(
            mask, half_dim_z, self.bev_z_extents_max, repeat_num, 1, 8)
        # z1
        self.tik_inst.vsub(
            mask, bev_anchor_ub[1, 0], half_dim_z,
            bev_anchor_ub[1, 0], repeat_num, 1, 1, 1, 8, 8, 8)
        # z2
        self.tik_inst.vsub(
            mask, bev_anchor_ub[3, 0], half_dim_z,
            bev_anchor_ub[3, 0], repeat_num, 1, 1, 1, 8, 8, 8)

    def _compute_norm(self, bev_anchor_ub, repeat_num):
        mask = self.repeat_data_num
        bev_extents_min_tiled = \
            [self.bev_x_extents_min, self.bev_z_extents_min,
             self.bev_x_extents_min, self.bev_z_extents_min]
        bev_x_extents_range = self.bev_x_extents_max - self.bev_x_extents_min
        bev_z_extents_range = self.bev_z_extents_max - self.bev_z_extents_min
        extents_tiled = [bev_x_extents_range, bev_z_extents_range,
                         bev_x_extents_range, bev_z_extents_range]
        for channel_index in range(4):
            self.tik_inst.vadds(
                mask, bev_anchor_ub[channel_index, 0],
                bev_anchor_ub[channel_index, 0],
                -bev_extents_min_tiled[channel_index],
                repeat_num, 1, 1, 8, 8)
            self.tik_inst.vmuls(
                mask, bev_anchor_ub[channel_index, 0],
                bev_anchor_ub[channel_index, 0],
                1.0 / extents_tiled[channel_index], repeat_num, 1, 1, 8, 8)

    def _data_move_out(self, tensor_out, tensor_ub, anchor_index, anchor_num,
                       channel_num):
        anchor_num_format = self._get_align_num(
            anchor_num, self.repeat_data_num)
        block_num_move_out = self._ceil_div(anchor_num * channel_num,
                                            self.block_data_num)
        with self.tik_inst.new_stmt_scope():
            data_move_out_ub = self.tik_inst.Tensor(
                self.data_type, (anchor_num_format, channel_num),
                self.tik.scope_ubuf, "data_move_out_ub")

            self.tik_inst.v4dtrans(
                True, data_move_out_ub, tensor_ub,
                anchor_num_format, channel_num)
            self.tik_inst.data_move(
                tensor_out[anchor_index, 0], data_move_out_ub,
                0, 1, block_num_move_out, 0, 0)

    def _result_move_out(self, bev_anchor_ub, regressed_anchor_ub,
                         anchor_index, anchor_num):
        self._data_move_out(self.bev_anchor, bev_anchor_ub, anchor_index,
                            anchor_num, self.c_bev_anchor)
        self._data_move_out(self.regressed_anchor, regressed_anchor_ub,
                            anchor_index, anchor_num, self.c_regressed_anchor)

    def tik_output_debug(self):
        return self.tik_inst


def offset_to_bev_anchor(offsets, anchors, bev_anchor, regressed_anchor,
                         bev_extents, kernel_name="OffsetToAnchor",
                         test=False):
    obj = OffsetToBevAnchor(offsets, anchors, bev_anchor, regressed_anchor,
                            bev_extents, kernel_name)
    obj.mode1_compute()
    if not test:
        return 0

    obj.tik_output_debug()
    return 0
