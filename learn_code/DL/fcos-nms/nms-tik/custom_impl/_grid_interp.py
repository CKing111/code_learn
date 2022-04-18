from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor


class VariableMem(object):
    def __init__(self):
        self.grid_x = "grid_x_ub"
        self.grid_y = "grid_y_ub"
        self.grid_z = "grid_z_ub"
        self.xtmp_fp32 = "xtmp_float"
        self.ytmp_fp32 = "ytmp_float"
        self.ztmp_fp32 = "ztmp_float"
        self.snwe_fp16 = "snwe_float"
        self.snwe_fp32 = "snwe_float32"
        self.x_loc_int = "x_location_int"
        self.y_loc_int = "y_location_int"
        self.z_loc_int = "z_location_int"
        self.x_loc_fp32 = "x_location_float"
        self.y_loc_fp32 = "y_location_float"
        self.z_loc_fp32 = "z_location_float"

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise RuntimeError("Attribute: %s already exists" % key)
        self.__dict__[key] = value


class Interpolation:
    def __init__(self, container, in_arglist, out_arglist):
        self.cont = container
        self.tinst = container.tinst
        self.tik = container.tik
        self.in_arglist = in_arglist
        self.dims = len(in_arglist)
        self.var_mem = VariableMem()
        if self.dims == 2:
            self.in_width, self.in_height = in_arglist
            self.out_width, self.out_height = out_arglist
        else:
            self.in_width, self.in_height, self.in_depth = in_arglist
            self.out_width, self.out_height, self.out_depth = out_arglist
        self.loc_int_dict = {0: self.var_mem.x_loc_int,
                             1: self.var_mem.y_loc_int,
                             2: self.var_mem.z_loc_int}
        self.loc_dict_nearest = {0: self.var_mem.grid_x,
                                 1: self.var_mem.grid_y,
                                 2: self.var_mem.grid_z}
        self.loc_dict_bilinear = {0: self.var_mem.xtmp_fp32,
                                  1: self.var_mem.ytmp_fp32,
                                  2: self.var_mem.ztmp_fp32}
        self.loc_int_list = [0, 0, 0]
        self.mask = 64
        self.block_data_num = self.cont.const_block_byte // \
                              self.cont.const_dtype_byte.get(
            'float16')
        self.num_per_cmd_fp32 = self.cont.get_vec_proc_num_per_cmd("float32")
        self.num_per_cmd_fp16 = self.cont.get_vec_proc_num_per_cmd("float16")

        if self.in_width * self.in_height - 1 > \
                self.cont.const_dmove_max_stride:
            self.move_data = self.move_to_ub_with_loop
        else:
            self.move_data = self.move_to_ub_no_loop
        if self.out_width * self.out_height - 1 > \
                self.cont.const_dmove_max_stride:
            self.move_gm = self.move_to_gm_with_loop
        else:
            self.move_gm = self.move_to_gm_no_loop

    def move_to_ub_with_loop(self, seq_ub, feature_map, cur_batch, cur_seq,
                             loc_list, nchannel_len, process_len):
        with self.tinst.for_range(0, nchannel_len // self.block_data_num) as \
                c1_channel:
            self.tinst.data_move(seq_ub[c1_channel, cur_seq, 0], feature_map[
                cur_batch, loc_list[2], c1_channel, loc_list[1], loc_list[
                    0], 0], 0, 1, 1, 0, 0)

    def move_to_ub_no_loop(self, seq_ub, feature_map, cur_batch, cur_seq,
                           loc_list, nchannel_len, process_len):
        self.tinst.data_move(seq_ub[0, cur_seq, 0], feature_map[
            cur_batch, loc_list[2], 0, loc_list[1], loc_list[0], 0], 0,
                             nchannel_len // self.block_data_num, 1,
                             self.in_width * self.in_height - 1,
                             process_len - 1)

    def move_to_gm_no_loop(self, output_gm, seq_ub, cur_batch, cur_depth,
                           start_loc, nchannel_len, location_len):
        self.tinst.data_move(output_gm[cur_batch, cur_depth, 0, start_loc, 0],
                             seq_ub, 0, nchannel_len // self.block_data_num,
                             location_len, 0,
                             self.out_width * self.out_height - location_len)

    def move_to_gm_with_loop(self, output_gm, seq_ub, cur_batch, cur_depth,
                             start_loc, nchannel_len, location_len):
        with self.tinst.for_range(0, nchannel_len // self.block_data_num) as \
                c1_channel:
            self.tinst.data_move(
                output_gm[cur_batch, cur_depth, c1_channel, start_loc, 0],
                seq_ub[c1_channel, 0, 0], 0, 1, location_len, 0, 0)

    def c0_vmuls(self, nchannel_len, seq_ub, cur_seq, weight, process_len):
        for_time = nchannel_len // self.block_data_num // \
                   self.cont.const_vector_proc_max_rpt
        remain_data = nchannel_len // self.block_data_num - for_time * \
                      self.cont.const_vector_proc_max_rpt
        if process_len <= self.cont.const_vector_proc_max_rpt:
            if for_time > 0:
                with self.tinst.for_range(0, for_time) as time:
                    self.tinst.vmuls(self.block_data_num, seq_ub[
                        time * self.cont.const_vector_proc_max_rpt, cur_seq,
                        0], seq_ub[time * self.cont.const_vector_proc_max_rpt,
                                   cur_seq, 0], weight,
                                     self.cont.const_vector_proc_max_rpt, 1, 1,
                                     process_len, process_len)
            if remain_data > 0:
                self.tinst.vmuls(self.block_data_num, seq_ub[
                    for_time * self.cont.const_vector_proc_max_rpt, cur_seq,
                    0],
                                 seq_ub[
                                     for_time *
                                     self.cont.const_vector_proc_max_rpt,
                                     cur_seq, 0],
                                 weight, remain_data, 1, 1, process_len,
                                 process_len)
        else:
            with self.tinst.for_range(0, nchannel_len // self.block_data_num) \
                    as time:
                self.tinst.vmuls(self.block_data_num, seq_ub[time, cur_seq, 0],
                                 seq_ub[time, cur_seq, 0], weight, 1, 1, 1, 0,
                                 0)

    def range_judgment(self, process_len, bufs, loc_dict, result_ub, tmp_ub):
        '''
        judge the tensor whether in range(0,[width,height,depth]),
        if in than 1 else 0
        :param process_len: the length of tensor
        :param bufs: include the tensor which need to be compare
        :param loc_dict: include the boundary value of each dims
        :param result_ub: return mask, if in range[1] else [0] tensor(int32)
        :param tmp_ub: tmp_ub, tensor(float32)
        '''
        repeat_time = process_len // self.mask
        last_process = process_len - repeat_time * self.mask

        with self.tinst.new_stmt_scope():
            zero_tensor = self.tinst.Tensor("float32", (self.mask,),
                                            self.tik.scope_ubuf, "zero_tensor")
            self.tinst.vector_dup(self.mask, zero_tensor, 0.0, 1, 1, 8)
            cmp_list = []
            for dim in range(self.dims):
                cmp_list.append(self.tinst.Tensor("float32", (self.mask,),
                                                  self.tik.scope_ubuf,
                                                  "latter_tensor_{}".format(
                                                      dim)))
                self.tinst.vector_dup(self.mask, cmp_list[dim],
                                      self.in_arglist[dim], 1, 1, 8)

            if repeat_time > 0:
                with self.tinst.for_range(0, repeat_time) as rt_time:
                    self.tinst.vector_dup(self.mask, tmp_ub[rt_time * 64], 1.0,
                                          1, 1, 8)
                    for dim in range(self.dims):
                        self._whether_in_range(self.mask, rt_time, bufs[
                            loc_dict[dim]].const_tensor, tmp_ub, zero_tensor,
                                               cmp_list[dim])

            if last_process > 0:
                self.tinst.vector_dup(last_process, tmp_ub[repeat_time * 64],
                                      1.0, 1, 1, 8)
                for dim in range(self.dims):
                    self._whether_in_range(last_process, repeat_time,
                                           bufs[loc_dict[dim]].const_tensor,
                                           tmp_ub, zero_tensor, cmp_list[dim])

            buf_judge_all = {
                "result_ub": AVecBuf(result_ub, process_len, 0, self.cont,
                                     False, self.num_per_cmd_fp32),
                "tmp_ub": AVecBuf(tmp_ub, process_len, 0, self.cont, False,
                                  self.num_per_cmd_fp32), }
            cmd_dup_judge_tensor = [
                VecGCmd("vconv", "result_ub", "tmp_ub", round_mode="round")]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_judge_all,
                                        cmd_dup_judge_tensor, "result_ub")

        return result_ub

    def _whether_in_range(self, process_num, rep_offset, cur_location, tmp_ub,
                          zero_tensor, exceed_tensor):
        '''
        0 <= cur_location.x < width and 0 <= cur_location.y < height and 0
        <= cur_location.z < depth
        '''
        cmp_mask = self.tinst.vcmp_ge(process_num,
                                      cur_location[self.mask * rep_offset],
                                      zero_tensor, 1, 1)
        self.tinst.vsel(process_num, 0, tmp_ub[self.mask * rep_offset],
                        cmp_mask, tmp_ub[self.mask * rep_offset], zero_tensor,
                        1, 1, 1, 1, 1, 1)
        cmp_mask = self.tinst.vcmp_lt(process_num,
                                      cur_location[self.mask * rep_offset],
                                      exceed_tensor, 1, 1)
        self.tinst.vsel(process_num, 0, tmp_ub[self.mask * rep_offset],
                        cmp_mask, tmp_ub[self.mask * rep_offset], zero_tensor,
                        1, 1, 1, 1, 1, 1)
