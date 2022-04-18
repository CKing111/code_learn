from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ._grid_interp import Interpolation


class Nearest(Interpolation):
    def gatherpoint_nearest(self, process_len, nchannel_len, cur_batch, seq_ub,
                            feature_map, buf_nearest_all, legal_mask=None):
        for dim in range(self.dims):
            self.loc_int_list[dim] = self.tinst.Scalar("int32")
        if legal_mask is None:
            with self.tinst.for_range(0, process_len) as cur_seq:
                for dim in range(self.dims):
                    self.loc_int_list[dim].set_as(
                        buf_nearest_all[self.loc_int_dict[dim]].const_tensor[
                            cur_seq])
                self.move_data(seq_ub, feature_map, cur_batch, cur_seq,
                               self.loc_int_list, nchannel_len, process_len)
        else:
            legal = self.tinst.Scalar("int32")
            buf_near_all = {
                "seq_ub": AVecBuf(seq_ub, process_len * nchannel_len, 0,
                                  self.cont, False, self.num_per_cmd_fp16), }
            cmd_dup_mean_tensor = [VecGCmd("vector_dup", "seq_ub", scalar=0.0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_near_all,
                                        cmd_dup_mean_tensor, "seq_ub")

            with self.tinst.for_range(0, process_len) as cur_seq:
                for dim in range(self.dims):
                    self.loc_int_list[dim].set_as(
                        buf_nearest_all[self.loc_int_dict[dim]].const_tensor[
                            cur_seq])
                legal.set_as(legal_mask[cur_seq])
                with self.tinst.if_scope(legal == 1):
                    self.move_data(seq_ub, feature_map, cur_batch, cur_seq,
                                   self.loc_int_list, nchannel_len,
                                   process_len)
                with self.tinst.else_scope():
                    pass
        return seq_ub

    def nearest_cal(self, location_len, bufs, border_mode, tmp_ub, legal_mask):
        buf_nearest_all = {
            self.var_mem.x_loc_int: bufs[self.var_mem.x_loc_int],
            self.var_mem.y_loc_int: bufs[self.var_mem.y_loc_int],
            self.var_mem.x_loc_fp32: bufs[self.var_mem.grid_x],
            self.var_mem.y_loc_fp32: bufs[self.var_mem.grid_y]}

        self.near_by_int(bufs, buf_nearest_all)

        if border_mode == "zeros":
            cmd_trans_coord_tensor = [VecGCmd("vconv", self.var_mem.x_loc_fp32,
                                              self.var_mem.x_loc_int,
                                              round_mode=""),
                VecGCmd("vconv", self.var_mem.y_loc_fp32,
                        self.var_mem.y_loc_int, round_mode="")]
            if self.dims == 3:
                cmd_trans_coord_tensor.append(
                    VecGCmd("vconv", self.var_mem.z_loc_fp32,
                            self.var_mem.z_loc_int, round_mode=""))
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_nearest_all,
                                        cmd_trans_coord_tensor,
                                        self.var_mem.y_loc_int)
            legal_mask = self.range_judgment(location_len, bufs,
                                             self.loc_dict_nearest, legal_mask,
                                             tmp_ub)
        return legal_mask

    def nearest_gotgm(self, location_len, start_loc, nchannel_len, feature_gm,
                      output_gm, legal_mask, cur_batch, cur_depth, bufs,
                      seq_ub, border_mode):
        buf_gotgm_all = {self.var_mem.x_loc_int: bufs[self.var_mem.x_loc_int],
                         self.var_mem.y_loc_int: bufs[
                             self.var_mem.y_loc_int], }
        if self.dims == 3:
            buf_gotgm_all[self.var_mem.z_loc_int] = bufs[
                self.var_mem.z_loc_int]
        if border_mode == "zeros":
            seq_ub = self.gatherpoint_nearest(location_len, nchannel_len,
                                              cur_batch, seq_ub, feature_gm,
                                              buf_gotgm_all, legal_mask)
        else:
            seq_ub = self.gatherpoint_nearest(location_len, nchannel_len,
                                              cur_batch, seq_ub, feature_gm,
                                              buf_gotgm_all)

        self.move_gm(output_gm, seq_ub, cur_batch, cur_depth, start_loc,
                     nchannel_len, location_len)

    def near_by_int(self, bufs, buf_nearest_all):
        if self.dims == 2:
            cmd_trans_coord_tensor = [VecGCmd("vconv", self.var_mem.x_loc_int,
                                              self.var_mem.x_loc_fp32,
                                              round_mode="round"),
                VecGCmd("vconv", self.var_mem.y_loc_int,
                        self.var_mem.y_loc_fp32, round_mode="round")]
        else:
            buf_nearest_all[self.var_mem.z_loc_int] = bufs[
                self.var_mem.z_loc_int]
            buf_nearest_all[self.var_mem.z_loc_fp32] = bufs[
                self.var_mem.grid_z]
            cmd_trans_coord_tensor = [VecGCmd("vconv", self.var_mem.x_loc_int,
                                              self.var_mem.x_loc_fp32,
                                              round_mode="away-zero"),
                VecGCmd("vconv", self.var_mem.y_loc_int,
                        self.var_mem.y_loc_fp32, round_mode="away-zero"),
                VecGCmd("vconv", self.var_mem.z_loc_int,
                        self.var_mem.z_loc_fp32, round_mode="away-zero")]

        VecGExecutor.exec_vec_g_cmd(self.cont, buf_nearest_all,
                                    cmd_trans_coord_tensor,
                                    self.var_mem.x_loc_int)
