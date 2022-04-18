from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ._grid_interp import Interpolation


class Bilinear(Interpolation):
    def gatherpoint_bilinear(self, process_len, nchannel_len, cur_batch,
                             seq_ub, feature_map, buf_bilinear_all,
                             snwe_weight, legal_mask=None):
        '''
        move data to ub and mul with weight
        :param process_len: the length of input tensor
        :param nchannel_len: c1*c0
        :param cur_batch: current batch
        :param seq_ub: input ub
        :param feature_map: input gm
        :param buf_bilinear_all: include all dims' location
        :param snwe_weight: weight
        :param legal_mask: whether location is legal [0,1]
        '''
        for dim in range(self.dims):
            self.loc_int_list[dim] = self.tinst.Scalar("int32")
        weight = self.tinst.Scalar("float16")

        if legal_mask is None:
            with self.tinst.for_range(0, process_len) as cur_seq:
                weight.set_as(snwe_weight[cur_seq])
                for dim in range(self.dims):
                    self.loc_int_list[dim].set_as(
                        buf_bilinear_all[self.loc_int_dict[dim]].const_tensor[
                            cur_seq])
                self.move_data(seq_ub, feature_map, cur_batch, cur_seq,
                               self.loc_int_list, nchannel_len, process_len)

                self.c0_vmuls(nchannel_len, seq_ub, cur_seq, weight,
                              process_len)

        else:
            legal = self.tinst.Scalar("int32")
            buf_mean_all = {
                "seq_ub": AVecBuf(seq_ub, process_len * nchannel_len, 0,
                                  self.cont, False, self.num_per_cmd_fp16), }
            cmd_dup_mean_tensor = [VecGCmd("vector_dup", "seq_ub", scalar=0.0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_mean_all,
                                        cmd_dup_mean_tensor, "seq_ub")
            with self.tinst.for_range(0, process_len) as cur_seq:
                for dim in range(self.dims):
                    self.loc_int_list[dim].set_as(
                        buf_bilinear_all[self.loc_int_dict[dim]].const_tensor[
                            cur_seq])
                legal.set_as(legal_mask[cur_seq])
                weight.set_as(snwe_weight[cur_seq])
                with self.tinst.if_scope(legal == 1):
                    self.move_data(seq_ub, feature_map, cur_batch, cur_seq,
                                   self.loc_int_list, nchannel_len,
                                   process_len)

                    self.c0_vmuls(nchannel_len, seq_ub, cur_seq, weight,
                                  process_len)

                with self.tinst.else_scope():
                    pass

        return seq_ub

    def bilinear(self, location_len, start_loc, bufs, nchannel_len, feature_gm,
                 output_gm, border_mode, cur_batch, cur_depth):
        '''
        根据上部计算得到的坐标值，取floor值，
        1.计算最终的不参与计算的mask，
        2.进行gatherpoint
        3.最终将gatherpoint得到的数据进行与mask取值得到最终的结果
        4.将ub中的数据move到输出中
        '''
        with self.tinst.new_stmt_scope():
            legal_mask = self.tinst.Tensor("int32", (location_len,),
                                           self.tik.scope_ubuf, "legal_mask")
            seq_ub = self.tinst.Tensor("float16",
                                       (nchannel_len // 16, location_len, 16),
                                       self.tik.scope_ubuf, "seq_ub")
            result_ub = self.tinst.Tensor("float16", (
                nchannel_len // 16, location_len, 16), self.tik.scope_ubuf,
                                          "result_ub")
            buf_bilinear_all = self._apply_buf(location_len, bufs, result_ub,
                                               seq_ub, nchannel_len)
            cmd_trans_coord_tensor = [
                VecGCmd("vconv", self.var_mem.x_loc_int,
                        self.var_mem.x_loc_fp32, round_mode="floor"),
                VecGCmd("vconv", self.var_mem.y_loc_int,
                        self.var_mem.y_loc_fp32, round_mode="floor"),
                VecGCmd("vconv", self.var_mem.xtmp_fp32,
                        self.var_mem.x_loc_int, round_mode=""),
                VecGCmd("vconv", self.var_mem.ytmp_fp32,
                        self.var_mem.y_loc_int, round_mode=""), ]
            if self.dims == 3:
                buf_bilinear_all[self.var_mem.z_loc_int] = AVecBuf.create(
                    (location_len,), "int32", self.tik.scope_ubuf,
                    self.var_mem.z_loc_int, self.cont)
                buf_bilinear_all[self.var_mem.ztmp_fp32] = AVecBuf.create(
                    (location_len,), "float32", self.tik.scope_ubuf,
                    self.var_mem.ztmp_fp32, self.cont)
                buf_bilinear_all[self.var_mem.z_loc_fp32] = bufs[
                    self.var_mem.grid_z]
                cmd_trans_coord_tensor.extend([
                    VecGCmd("vconv", self.var_mem.z_loc_int,
                            self.var_mem.z_loc_fp32, round_mode="floor"),
                    VecGCmd("vconv", self.var_mem.ztmp_fp32,
                            self.var_mem.z_loc_int, round_mode="")])
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                        cmd_trans_coord_tensor,
                                        self.var_mem.ytmp_fp32)
            if self.dims == 2:
                self.bilinear2d_cal_weight(border_mode, location_len,
                                           cur_batch, nchannel_len, seq_ub,
                                           feature_gm, buf_bilinear_all,
                                           legal_mask)
            else:
                self.bilinear3d_cal_weight(border_mode, location_len,
                                           cur_batch, nchannel_len, seq_ub,
                                           feature_gm, buf_bilinear_all,
                                           legal_mask)
            self.move_gm(output_gm, result_ub, cur_batch, cur_depth, start_loc,
                         nchannel_len, location_len)

        return output_gm

    def _apply_buf(self, location_len, bufs, result_ub, seq_ub, nchannel_len):
        buf_bilinear_all = {
            self.var_mem.x_loc_int: AVecBuf.create((location_len,), "int32",
                                                   self.tik.scope_ubuf,
                                                   self.var_mem.x_loc_int,
                                                   self.cont),
            self.var_mem.y_loc_int: AVecBuf.create((location_len,), "int32",
                                                   self.tik.scope_ubuf,
                                                   self.var_mem.y_loc_int,
                                                   self.cont),
            self.var_mem.xtmp_fp32: AVecBuf.create((location_len,), "float32",
                                                   self.tik.scope_ubuf,
                                                   self.var_mem.xtmp_fp32,
                                                   self.cont),
            self.var_mem.ytmp_fp32: AVecBuf.create((location_len,), "float32",
                                                   self.tik.scope_ubuf,
                                                   self.var_mem.ytmp_fp32,
                                                   self.cont),
            self.var_mem.snwe_fp32: AVecBuf.create((location_len,), "float32",
                                                   self.tik.scope_ubuf,
                                                   self.var_mem.snwe_fp32,
                                                   self.cont),
            self.var_mem.snwe_fp16: bufs['tmp_ub'],
            self.var_mem.x_loc_fp32: bufs[self.var_mem.grid_x],
            self.var_mem.y_loc_fp32: bufs[self.var_mem.grid_y],
            "result_ub": AVecBuf(result_ub, location_len * nchannel_len, 0,
                                 self.cont, False, self.num_per_cmd_fp16),
            "seq_ub": AVecBuf(seq_ub, location_len * nchannel_len, 0,
                              self.cont, False, self.num_per_cmd_fp16), }
        return buf_bilinear_all

    def bilinear2d_cal_weight(self, border_mode, location_len, cur_batch,
                              nchannel_len, seq_ub, feature_gm,
                              buf_bilinear_all, legal_mask):

        seq_ub = self.got_nw_value(border_mode, location_len, cur_batch,
                                   nchannel_len, seq_ub, feature_gm,
                                   buf_bilinear_all, legal_mask)
        cmd_dup_mean_tensor = [VecGCmd("vector_dup", "result_ub", scalar=0.0),
                               VecGCmd("vadd", dst_name="result_ub",
                                       src0_name="result_ub",
                                       src1_name="seq_ub"), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_dup_mean_tensor, "seq_ub")
        cal_func_list = [self.got_ne_value, self.got_se_value,
                         self.got_sw_value]
        for func in cal_func_list:
            seq_ub = func(location_len, cur_batch, nchannel_len, seq_ub,
                          feature_gm, buf_bilinear_all, legal_mask)
            cmd_dup_mean_tensor = [
                VecGCmd("vadd", dst_name="result_ub", src0_name="result_ub",
                        src1_name="seq_ub"), ]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                        cmd_dup_mean_tensor, "seq_ub")

    def bilinear3d_cal_weight(self, border_mode, location_len, cur_batch,
                              nchannel_len, seq_ub, feature_gm,
                              buf_bilinear_all, legal_mask):

        seq_ub = self.got_nwf_value(border_mode, location_len, cur_batch,
                                    nchannel_len, seq_ub, feature_gm,
                                    buf_bilinear_all, legal_mask)
        cmd_dup_mean_tensor = [VecGCmd("vector_dup", "result_ub", scalar=0.0),
                               VecGCmd("vadd", dst_name="result_ub",
                                       src0_name="result_ub",
                                       src1_name="seq_ub"), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_dup_mean_tensor, "seq_ub")
        cal_func_list = [self.got_nef_value, self.got_sef_value,
                         self.got_swf_value, self.got_swb_value,
                         self.got_seb_value, self.got_neb_value,
                         self.got_nwb_value]
        for func in cal_func_list:
            seq_ub = func(location_len, cur_batch, nchannel_len, seq_ub,
                          feature_gm, buf_bilinear_all, legal_mask)
            cmd_dup_mean_tensor = [
                VecGCmd("vadd", dst_name="result_ub", src0_name="result_ub",
                        src1_name="seq_ub"), ]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                        cmd_dup_mean_tensor, "seq_ub")

    def got_nw_value(self, border_mode, location_len, cur_batch, nchannel_len,
                     seq_ub, feature_gm, buf_bilinear_all, legal_mask):

        if border_mode == "reflection" or border_mode == "border":
            legal_mask = None
        else:
            legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                             self.loc_dict_bilinear,
                                             legal_mask, buf_bilinear_all[
                                                 self.var_mem.snwe_fp32].
                                             const_tensor)
        cmd_cal_nw_tensor = [VecGCmd("vadds", dst_name=self.var_mem.xtmp_fp32,
                                     src0_name=self.var_mem.xtmp_fp32,
                                     scalar=1.0),
                             VecGCmd("vadds", dst_name=self.var_mem.ytmp_fp32,
                                     src0_name=self.var_mem.ytmp_fp32,
                                     scalar=1.0),
                             VecGCmd("vsub", dst_name=self.var_mem.x_loc_fp32,
                                     src0_name=self.var_mem.xtmp_fp32,
                                     src1_name=self.var_mem.x_loc_fp32),
                             VecGCmd("vsub", dst_name=self.var_mem.y_loc_fp32,
                                     src0_name=self.var_mem.ytmp_fp32,
                                     src1_name=self.var_mem.y_loc_fp32),
                             VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                                     src0_name=self.var_mem.y_loc_fp32,
                                     src1_name=self.var_mem.x_loc_fp32),
                             VecGCmd("vconv", self.var_mem.snwe_fp16,
                                     self.var_mem.snwe_fp32, round_mode=""), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_nw_tensor, self.var_mem.ytmp_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)
        return seq_ub

    def _build_ne_sw_list(self):
        return [VecGCmd("vadds", dst_name=self.var_mem.x_loc_fp32,
                        src0_name=self.var_mem.x_loc_fp32, scalar=-1),
                VecGCmd("vabs", dst_name=self.var_mem.x_loc_fp32,
                        src0_name=self.var_mem.x_loc_fp32),
                VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                        src0_name=self.var_mem.y_loc_fp32,
                        src1_name=self.var_mem.x_loc_fp32),
                VecGCmd("vconv", self.var_mem.snwe_fp16,
                        self.var_mem.snwe_fp32, round_mode=""),
                VecGCmd("vconv", self.var_mem.x_loc_int,
                        self.var_mem.xtmp_fp32, round_mode="floor"), ]

    def got_ne_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                     feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_ne_tensor = [VecGCmd("vadds", dst_name=self.var_mem.ytmp_fp32,
                                     src0_name=self.var_mem.ytmp_fp32,
                                     scalar=-1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_ne_tensor, self.var_mem.ytmp_fp32)
        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_ne_tensor = self._build_ne_sw_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_ne_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_se_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                     feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_se_tensor = [VecGCmd("vadds", dst_name=self.var_mem.ytmp_fp32,
                                     src0_name=self.var_mem.ytmp_fp32,
                                     scalar=1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_se_tensor, self.var_mem.ytmp_fp32)
        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_se_tensor = [VecGCmd("vadds", dst_name=self.var_mem.y_loc_fp32,
                                     src0_name=self.var_mem.y_loc_fp32,
                                     scalar=-1),
                             VecGCmd("vabs", dst_name=self.var_mem.y_loc_fp32,
                                     src0_name=self.var_mem.y_loc_fp32),
                             VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                                     src0_name=self.var_mem.y_loc_fp32,
                                     src1_name=self.var_mem.x_loc_fp32),
                             VecGCmd("vconv", self.var_mem.snwe_fp16,
                                     self.var_mem.snwe_fp32, round_mode=""),
                             VecGCmd("vconv", self.var_mem.y_loc_int,
                                     self.var_mem.ytmp_fp32,
                                     round_mode="floor"), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_se_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_sw_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                     feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_sw_tensor = [VecGCmd("vadds", dst_name=self.var_mem.xtmp_fp32,
                                     src0_name=self.var_mem.xtmp_fp32,
                                     scalar=-1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_sw_tensor, self.var_mem.xtmp_fp32)

        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_sw_tensor = self._build_ne_sw_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_sw_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def _build_nef_swf_seb_nwb_list(self):
        return [VecGCmd("vadds", dst_name=self.var_mem.x_loc_fp32,
                        src0_name=self.var_mem.x_loc_fp32, scalar=-1),
                VecGCmd("vabs", dst_name=self.var_mem.x_loc_fp32,
                        src0_name=self.var_mem.x_loc_fp32),
                VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                        src0_name=self.var_mem.y_loc_fp32,
                        src1_name=self.var_mem.x_loc_fp32),
                VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                        src0_name=self.var_mem.snwe_fp32,
                        src1_name=self.var_mem.z_loc_fp32),
                VecGCmd("vconv", self.var_mem.snwe_fp16,
                        self.var_mem.snwe_fp32, round_mode=""),
                VecGCmd("vconv", self.var_mem.x_loc_int,
                        self.var_mem.xtmp_fp32, round_mode="floor"), ]

    def _build_sef_neb_list(self):
        return [VecGCmd("vadds", dst_name=self.var_mem.y_loc_fp32,
                        src0_name=self.var_mem.y_loc_fp32, scalar=-1),
            VecGCmd("vabs", dst_name=self.var_mem.y_loc_fp32,
                    src0_name=self.var_mem.y_loc_fp32),
            VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                    src0_name=self.var_mem.y_loc_fp32,
                    src1_name=self.var_mem.x_loc_fp32),
            VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                    src0_name=self.var_mem.snwe_fp32,
                    src1_name=self.var_mem.z_loc_fp32),
            VecGCmd("vconv", self.var_mem.snwe_fp16, self.var_mem.snwe_fp32,
                    round_mode=""),
            VecGCmd("vconv", self.var_mem.y_loc_int, self.var_mem.ytmp_fp32,
                    round_mode="floor"), ]

    def got_nwf_value(self, border_mode, location_len, cur_batch, nchannel_len,
                      seq_ub, feature_gm, buf_bilinear_all, legal_mask):

        if border_mode == "reflection" or border_mode == "border":
            # 不存在越界可能,按序输出到gm中
            legal_mask = None
        else:
            # left_top
            legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                             self.loc_dict_bilinear,
                                             legal_mask, buf_bilinear_all[
                                                 self.var_mem.snwe_fp32].
                                             const_tensor)

        cmd_cal_nwf_tensor = [VecGCmd("vadds", dst_name=self.var_mem.xtmp_fp32,
                                      src0_name=self.var_mem.xtmp_fp32,
                                      scalar=1.0),
                              VecGCmd("vadds", dst_name=self.var_mem.ytmp_fp32,
                                      src0_name=self.var_mem.ytmp_fp32,
                                      scalar=1.0),
                              VecGCmd("vadds", dst_name=self.var_mem.ztmp_fp32,
                                      src0_name=self.var_mem.ztmp_fp32,
                                      scalar=1.0),
                              VecGCmd("vsub", dst_name=self.var_mem.x_loc_fp32,
                                      src0_name=self.var_mem.xtmp_fp32,
                                      src1_name=self.var_mem.x_loc_fp32),
                              VecGCmd("vsub", dst_name=self.var_mem.y_loc_fp32,
                                      src0_name=self.var_mem.ytmp_fp32,
                                      src1_name=self.var_mem.y_loc_fp32),
                              VecGCmd("vsub", dst_name=self.var_mem.z_loc_fp32,
                                      src0_name=self.var_mem.ztmp_fp32,
                                      src1_name=self.var_mem.z_loc_fp32),
                              VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                                      src0_name=self.var_mem.y_loc_fp32,
                                      src1_name=self.var_mem.x_loc_fp32),
                              VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                                      src0_name=self.var_mem.snwe_fp32,
                                      src1_name=self.var_mem.z_loc_fp32),
                              VecGCmd("vconv", self.var_mem.snwe_fp16,
                                      self.var_mem.snwe_fp32, round_mode=""), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_nwf_tensor, self.var_mem.ytmp_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)
        return seq_ub

    def got_nef_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                      feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_nefb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.ytmp_fp32,
                    src0_name=self.var_mem.ytmp_fp32, scalar=-1),
            VecGCmd("vadds", dst_name=self.var_mem.ztmp_fp32,
                    src0_name=self.var_mem.ztmp_fp32, scalar=-1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_nefb_tensor,
                                    self.var_mem.ytmp_fp32)
        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_nef_tensor = self._build_nef_swf_seb_nwb_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_nef_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_sef_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                      feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_sefb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.ytmp_fp32,
                    src0_name=self.var_mem.ytmp_fp32, scalar=1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_sefb_tensor,
                                    self.var_mem.ytmp_fp32)
        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_sef_tensor = self._build_sef_neb_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_sef_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_swf_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                      feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_swfb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.xtmp_fp32,
                    src0_name=self.var_mem.xtmp_fp32, scalar=-1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_swfb_tensor,
                                    self.var_mem.xtmp_fp32)

        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_swf_tensor = self._build_nef_swf_seb_nwb_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_swf_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_swb_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                      feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_swbb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.ztmp_fp32,
                    src0_name=self.var_mem.ztmp_fp32, scalar=1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_swbb_tensor,
                                    self.var_mem.ztmp_fp32)

        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_swb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.z_loc_fp32,
                    src0_name=self.var_mem.z_loc_fp32, scalar=-1),
            VecGCmd("vabs", dst_name=self.var_mem.z_loc_fp32,
                    src0_name=self.var_mem.z_loc_fp32),
            VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                    src0_name=self.var_mem.y_loc_fp32,
                    src1_name=self.var_mem.x_loc_fp32),
            VecGCmd("vmul", dst_name=self.var_mem.snwe_fp32,
                    src0_name=self.var_mem.snwe_fp32,
                    src1_name=self.var_mem.z_loc_fp32),
            VecGCmd("vconv", self.var_mem.snwe_fp16, self.var_mem.snwe_fp32,
                    round_mode=""),
            VecGCmd("vconv", self.var_mem.z_loc_int, self.var_mem.ztmp_fp32,
                    round_mode="floor"), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_swb_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_seb_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                      feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_sebb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.xtmp_fp32,
                    src0_name=self.var_mem.xtmp_fp32, scalar=1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_sebb_tensor,
                                    self.var_mem.xtmp_fp32)
        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_seb_tensor = self._build_nef_swf_seb_nwb_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_seb_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_neb_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                      feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_nebb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.ytmp_fp32,
                    src0_name=self.var_mem.ytmp_fp32, scalar=-1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_nebb_tensor,
                                    self.var_mem.ytmp_fp32)
        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_neb_tensor = self._build_sef_neb_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_neb_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub

    def got_nwb_value(self, location_len, cur_batch, nchannel_len, seq_ub,
                      feature_gm, buf_bilinear_all, legal_mask):
        cmd_cal_nwbb_tensor = [
            VecGCmd("vadds", dst_name=self.var_mem.xtmp_fp32,
                    src0_name=self.var_mem.xtmp_fp32, scalar=-1)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_nwbb_tensor,
                                    self.var_mem.xtmp_fp32)
        legal_mask = self.range_judgment(location_len, buf_bilinear_all,
                                         self.loc_dict_bilinear, legal_mask,
                                         buf_bilinear_all[
                                             self.var_mem.snwe_fp32].
                                         const_tensor)
        cmd_cal_nwb_tensor = self._build_nef_swf_seb_nwb_list()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_bilinear_all,
                                    cmd_cal_nwb_tensor, self.var_mem.snwe_fp32)

        seq_ub = self.gatherpoint_bilinear(location_len, nchannel_len,
                                           cur_batch, seq_ub, feature_gm,
                                           buf_bilinear_all, buf_bilinear_all[
                                               self.var_mem.snwe_fp16].
                                           const_tensor,
                                           legal_mask)

        return seq_ub
