from uti import interface_check


class PointSample:

    def __init__(self, container, batch, channel, src_npoint, dst_point,
                 kernel_n):
        self.tik_inst = container.tinst
        self.tik = container.tik
        interface_check.check_kernelname(kernel_n)
        self.kernel_name = kernel_n
        self.cont = container
        self.src_b = batch
        self.src_c = channel
        self.src_n = src_npoint
        self.src_n16 = ((self.src_n + 15) // 16) * 16
        self.src_c16 = ((self.src_c + 15) // 16) * 16

        self.point_num = dst_point
        self.aicore_use = container.const_aicore_num
        self._aicore_in_use_select(self.point_num)

        self.point_len = self.tik_inst.Scalar("int32")
        self.slr_int32 = self.tik_inst.Scalar("int32")

        self.once_pl = ((self.cont.const_ub_max_byte // 2) // (
                2 + self.src_c16 * 2)) // 16 * 16
        self._para_check()
        self.data_gm = self.tik_inst.Tensor("float16", (
            self.src_b, self.src_c, self.src_n), self.tik.scope_gm, "data_gm")
        self.seq_gm = self.tik_inst.Tensor("int32",
                                           (self.src_b, self.point_num),
                                           self.tik.scope_gm, "seq_gm")
        self.dst_gm = self.tik_inst.Tensor("float16", (
            self.src_b, self.src_c, self.point_num), self.tik.scope_gm,
                                           "dst_gm")

    def _aicore_in_use_select(self, cal_len):
        self.each_core = (cal_len + self.aicore_use - 1) // self.aicore_use
        self.xlen_each_core = ((self.each_core + 15) // 16) * 16
        self.aicore_use = (cal_len + self.xlen_each_core - 1) // \
                          self.xlen_each_core
        self.xlen_last_core = cal_len - self.xlen_each_core * (
                self.aicore_use - 1)
        if cal_len % 16 > 0 or self.aicore_use == 1:
            self.aicore_use = 1
            self.xlen_each_core, self.xlen_last_core = cal_len, cal_len

    def _para_check(self):
        if self.src_n16 * self.src_c16 > self.cont.const_l1_max_byte // 2 or\
                self.src_c16 > 3952:
            raise RuntimeError(
                " The number of points have exceeds the supported range.")
        if self.src_b * self.src_c * self.src_n < 2 or self.src_b * \
                self.point_num < 2:
            raise RuntimeError(
                " The number of sampling points should more than one.")
        if self.point_num > 65535 * 16:
            raise RuntimeError(
                " The total num of point_num has exceed supported num("
                "1048560).")

    def _transpose_general(self, src, dst, row, col):
        if row > col:
            repeat_time = row // 16
            src_rep_stride = 0 if repeat_time == 1 else col
            dst_rep_stride = 0 if repeat_time == 1 else 1
            for_time = col // 16
            src_offset = 16
            dst_offset = 16 * row
        else:
            repeat_time = col // 16
            src_rep_stride = 0 if repeat_time == 1 else 1
            dst_rep_stride = 0 if repeat_time == 1 else row
            for_time = row // 16
            src_offset = 16 * col
            dst_offset = 16

        with self.tik_inst.for_range(0, for_time) as loop_index:
            src_list = [src[i * col + loop_index * src_offset] for i in
                        range(16)]
            dst_list = [dst[i * row + loop_index * dst_offset] for i in
                        range(16)]
            self.tik_inst.vnchwconv(True, True, dst_list, src_list,
                                    repeat_time, dst_rep_stride,
                                    src_rep_stride)

    def _move_in_l1(self, length, batch, trans_ub, input_ub, l1_ub, src_gm):
        for_time = length // self.once_pl
        remain = length % self.once_pl
        if for_time > 0:
            with self.tik_inst.for_range(0, for_time) as i:
                self._move_in_ub(batch, self.src_c, self.once_pl, self.once_pl,
                                 self.src_n, input_ub, src_gm,
                                 i * self.once_pl)
                # (16,self.once_pl)->(self.once_pl,16)
                self._transpose_general(input_ub, trans_ub, self.src_c16,
                                        self.once_pl)

                self.tik_inst.data_move(l1_ub[i * self.once_pl, 0], trans_ub,
                                        0, 1,
                                        self.once_pl * self.src_c16 // 16, 0,
                                        0)
        if remain > 0:
            self._move_in_ub(batch, self.src_c, remain, self.once_pl,
                             self.src_n, input_ub, src_gm,
                             for_time * self.once_pl)
            # (16,remain)->(remain,16)
            self._transpose_general(input_ub, trans_ub, self.src_c16,
                                    self.once_pl)

            self.tik_inst.data_move(l1_ub[for_time * self.once_pl, 0],
                                    trans_ub, 0, 1,
                                    remain * self.src_c16 // 16, 0, 0)

    def _move_in_ub(self, batch, row, ub_col, ubreal_col, gm_col, in_ub, in_gm,
                    gm_offset):
        if gm_col % 16 == 0:
            self.tik_inst.data_move(in_ub, in_gm[batch, 0, gm_offset], 0, row,
                                    ub_col // 16, (gm_col - ub_col) // 16,
                                    (ubreal_col - ub_col) // 16)
        else:
            with self.tik_inst.for_range(0, row) as move_c:
                self.tik_inst.data_move(in_ub[move_c * ubreal_col],
                                        in_gm[batch, move_c, gm_offset], 0, 1,
                                        ub_col // 16, 0, 0)

    def _move_out_gm(self, batch, row, ub_col, ubreal_col, gm_col, in_ub,
                     out_gm, gm_offset):
        if gm_col % 16 == 0:
            self.tik_inst.data_move(out_gm[batch, 0, gm_offset], in_ub, 0, row,
                                    ub_col // 16, (ubreal_col - ub_col) // 16,
                                    (gm_col - ub_col) // 16)
        else:
            with self.tik_inst.for_range(0, self.src_c) as move_c:
                self.tik_inst.data_move(out_gm[batch, move_c, gm_offset],
                                        in_ub[move_c * ubreal_col], 0, 1,
                                        (ub_col + 15) // 16, 0, 0)

    def model_compute(self):
        with self.tik_inst.for_range(0, self.aicore_use,
                                     block_num=self.aicore_use) as index:
            with self.tik_inst.if_scope(index < self.aicore_use - 1):
                self.point_len.set_as(self.xlen_each_core)
            with self.tik_inst.else_scope():
                self.point_len.set_as(self.xlen_last_core)
            self._compute_eachcore(index)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.data_gm, self.seq_gm],
                               outputs=[self.dst_gm], enable_l2=True)

    def _compute_eachcore(self, index):
        input_data_l1 = self.tik_inst.Tensor("float16",
                                             (self.src_n16, self.src_c16),
                                             self.tik.scope_cbuf,
                                             "input_data_l1")
        input_seq_ub = self.tik_inst.Tensor("int32", (self.once_pl,),
                                            self.tik.scope_ubuf,
                                            "input_seq_ub")
        input_data_ub = self.tik_inst.Tensor("float16",
                                             (self.once_pl * self.src_c16,),
                                             self.tik.scope_ubuf,
                                             "input_data_ub")
        transpose_ub = self.tik_inst.Tensor("float16",
                                            (self.once_pl * self.src_c16,),
                                            self.tik.scope_ubuf,
                                            "transpose_ub")
        process_time = self.tik_inst.Scalar('int32')
        process_remain = self.tik_inst.Scalar('int32')
        process_len = self.tik_inst.Scalar('int32')
        process_time.set_as(self.point_len // self.once_pl)
        process_remain.set_as(self.point_len - process_time * self.once_pl)
        process_len.set_as(self.once_pl)
        with self.tik_inst.for_range(0, self.src_b) as batch:
            self._move_in_l1(self.src_n16, batch, transpose_ub, input_data_ub,
                             input_data_l1, self.data_gm)
            with self.tik_inst.if_scope(process_remain > 0):
                self._process_model(input_seq_ub, input_data_ub, transpose_ub,
                                    input_data_l1, batch,
                                    index * self.xlen_each_core +
                                    process_time * self.once_pl,
                                    process_remain)
            with self.tik_inst.else_scope():
                pass
            with self.tik_inst.if_scope(process_time > 0):
                with self.tik_inst.for_range(0, process_time) as fort:
                    self._process_model(input_seq_ub, input_data_ub,
                                        transpose_ub, input_data_l1, batch,
                                        index * self.xlen_each_core + fort *
                                        self.once_pl,
                                        process_len)
            with self.tik_inst.else_scope():
                pass

    def _process_model(self, input_seq_ub, input_data_ub, transpose_ub,
                       input_data_l1, batch, seq_offsset, num):
        self.tik_inst.data_move(input_seq_ub, self.seq_gm[batch, seq_offsset],
                                0, 1, (num + 7) // 8, 0, 0)
        with self.tik_inst.for_range(0, num) as i:
            self.slr_int32.set_as(input_seq_ub[i])
            self.tik_inst.data_move(input_data_ub[i * self.src_c16],
                                    input_data_l1[self.slr_int32, 0], 0, 1,
                                    self.src_c16 // 16, 0, 0)
        self._transpose_general(input_data_ub, transpose_ub, self.once_pl,
                                self.src_c16)
        self._move_out_gm(batch, self.src_c, num, self.once_pl, self.point_num,
                          transpose_ub, self.dst_gm, seq_offsset)
