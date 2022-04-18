# -*- coding:utf-8 -*-
import math

G_BLOCK_BYTE = 32
G_BLOCK_NUM_PER_CMD = 8
G_CMD_MAX_REPEAT = 255


class Utils:

    def __init__(self, container, dtype):
        self.container = container
        self.tik_inst = self.container.tinst
        self.num_per_block = G_BLOCK_BYTE //\
                             self.container.const_dtype_byte.get(dtype)
        self.max_num_per_cmd = G_BLOCK_NUM_PER_CMD * self.num_per_block
        self.max_num_per_loop = G_CMD_MAX_REPEAT * self.max_num_per_cmd

    def _parse_cmd(self, cmd, scalar, src0_buf, src1_buf, dst_buf, mask,
                   repeat, rep_stride):
        if (cmd == "vadds"):
            self.tik_inst.vadds(mask, dst_buf, src0_buf, scalar, repeat, 1, 1,
                                rep_stride, rep_stride)
        elif (cmd == "vmul"):
            self.tik_inst.vmul(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1,
                               1, rep_stride, rep_stride, rep_stride)
        elif (cmd == "vadd"):
            self.tik_inst.vadd(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1,
                               1, rep_stride, rep_stride, rep_stride)
        elif (cmd == "vmuls"):
            self.tik_inst.vmuls(mask, dst_buf, src0_buf, scalar, repeat, 1, 1,
                                rep_stride, rep_stride)
        elif (cmd == "vdiv"):
            self.tik_inst.vdiv(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1,
                               1, rep_stride, rep_stride, rep_stride)
        elif (cmd == "vsub"):
            self.tik_inst.vsub(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1,
                               1, rep_stride, rep_stride, rep_stride)
        elif (cmd == "vmax"):
            self.tik_inst.vmax(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1,
                               1, rep_stride, rep_stride, rep_stride)
        elif (cmd == "vmin"):
            self.tik_inst.vmin(mask, dst_buf, src0_buf, src1_buf, repeat, 1, 1,
                               1, rep_stride, rep_stride, rep_stride)
        elif (cmd == "vabs"):
            self.tik_inst.vabs(mask, dst_buf, src0_buf, repeat, 1, 1,
                               rep_stride, rep_stride)
        elif (cmd == "vrec"):
            self.tik_inst.vrec(mask, dst_buf, src0_buf, repeat, 1, 1,
                               rep_stride, rep_stride)
        elif (cmd == "vector_dup"):
            self.tik_inst.vector_dup(mask, dst_buf, scalar, repeat, 1,
                                     rep_stride)
        else:
            raise RuntimeError("Not supported cmd:{}".format(cmd))

    def cmd_vec(self, cmd, num, dst_buf, src0_buf, src1_buf, dst_buf_idx,
                src0_buf_idx, src1_buf_idx, scalar):
        loop = num // self.max_num_per_loop
        if loop > 0:
            with self.tik_inst.for_range(0, loop) as loop_i:
                offset = loop_i * self.max_num_per_loop
                self._parse_cmd(cmd, scalar, src0_buf[offset + src0_buf_idx],
                                src1_buf[offset + src1_buf_idx],
                                dst_buf[offset + dst_buf_idx],
                                self.max_num_per_cmd, G_CMD_MAX_REPEAT, 8)

        cur_offset = loop * self.max_num_per_loop
        repeat = (num - cur_offset) // self.max_num_per_cmd
        if repeat > 0:
            self._parse_cmd(cmd, scalar, src0_buf[cur_offset + src0_buf_idx],
                            src1_buf[cur_offset + src1_buf_idx],
                            dst_buf[cur_offset + dst_buf_idx],
                            self.max_num_per_cmd, repeat, 8)

        cur_offset = cur_offset + repeat * self.max_num_per_cmd
        remain = num - cur_offset
        if remain > 0:
            self._parse_cmd(cmd, scalar, src0_buf[cur_offset + src0_buf_idx],
                            src1_buf[cur_offset + src1_buf_idx],
                            dst_buf[cur_offset + dst_buf_idx],
                            remain, 1, 0)

    def cmd_vec_sp(self, cmd, num, dst_buf, src0_buf, src1_buf, dst_buf_idx,
                   src0_buf_idx, src1_buf_idx, scalar):
        repeat = num // self.max_num_per_cmd
        with self.tik_inst.if_scope(repeat > 0):
            self._parse_cmd(cmd, scalar, src0_buf[src0_buf_idx],
                            src1_buf[src1_buf_idx], dst_buf[dst_buf_idx],
                            self.max_num_per_cmd, repeat, 8)
        with self.tik_inst.else_scope():
            pass
        cur_offset = repeat * self.max_num_per_cmd
        remain = num - cur_offset
        with self.tik_inst.if_scope(remain > 0):
            self._parse_cmd(cmd, scalar, src0_buf[cur_offset + src0_buf_idx],
                            src1_buf[cur_offset + src1_buf_idx],
                            dst_buf[cur_offset + dst_buf_idx], remain, 1, 0)
        with self.tik_inst.else_scope():
            pass

    def round_f32toi32(self, ub_fp32, ub_fp32tmp, ub_fp16_tmp, ub_int32):
        """
        only process 1
        """
        self.tik_inst.vconv(1, "", ub_fp16_tmp, ub_fp32, 1, 1, 1, 4, 8)
        self.tik_inst.vconv(1, "round", ub_int32, ub_fp16_tmp, 1, 1, 1, 8, 4)
        self.tik_inst.vconv(1, "", ub_fp32tmp, ub_fp16_tmp, 1, 1, 1, 8, 4)
        self.tik_inst.vsub(1, ub_fp32tmp, ub_fp32, ub_fp32tmp, 1, 1, 1, 1, 8,
                           8, 8)
        self.tik_inst.vconv(1, "", ub_fp16_tmp, ub_fp32tmp, 1, 1, 1, 4, 8)
        self.tik_inst.vconv(1, "round", ub_int32[8], ub_fp16_tmp, 1, 1, 1, 8,
                            4)
        self.tik_inst.vadd(1, ub_int32, ub_int32[8], ub_int32, 1, 1, 1, 1, 8,
                           8, 8)


class GetThreetop:

    def __init__(self, container):
        self.container = container
        self.tik_inst = self.container.tinst
        self.utilfp32 = Utils(self.container, "float32")

    def init_index(self, int_value, float_value, ub_length, index_ub,
                   index_int_ub, fp16_ub):
        '''
         init_index in length
        :param int_value: every time add length
        :param float_value: add value
        :param ub_length: total length
        :param index_fp16_ub: return index_fp16
        :param index_int_ub: for initial 16number
        '''
        max_fortime = int(math.ceil(math.log(ub_length, 2)))
        max_remain = ub_length - int(math.pow(2, max_fortime - 1))
        range_num = 16
        if ub_length < 16:
            range_num = ub_length
        with self.tik_inst.for_range(0, range_num) as n_time:
            index_int_ub[n_time] = n_time + 1
        # int32_2_fp16
        self.tik_inst.vconv(range_num, "", fp16_ub, index_int_ub, 1, 1, 1, 1,
                            2, 1.0)
        self.tik_inst.vconv(range_num, "", index_ub, fp16_ub, 1, 1, 1, 1, 2)

        int_value.set_as(16)
        if ub_length > 16:
            with self.tik_inst.for_range(0, max_fortime - 5):
                float_value.set_as(index_ub[int_value - 1])
                self.utilfp32.cmd_vec_sp("vadds", int_value, index_ub,
                                         index_ub, index_ub, int_value,
                                         0, 0, float_value)
                int_value.set_as(2 * int_value)
            float_value.set_as(index_ub[int_value - 1])
            self.utilfp32.cmd_vec("vadds", max_remain, index_ub, index_ub,
                                  index_ub, int_value, 0, 0, float_value)
        self.utilfp32.cmd_vec("vadds", ub_length, index_ub, index_ub, index_ub,
                              0, 0, 0, - ub_length)
        self.utilfp32.cmd_vec("vabs", ub_length, index_ub, index_ub, index_ub,
                              0, 0, 0, 0)

    def get_threetop(self, length, input_ub, cmp_ub, tmp_ub, index_ub,
                     int_value, fp32_value, fp16tmp_ub, int_ub, result_value,
                     result_idx, point_id):
        """
        :param length: distance num
        :param input_ub: shape should be point_len*set_len
        :param cmp_ub:for compare,shape=(1,set_len)
        :param tmp_ub:for tmp,shape=(1,set_len)
        :param index_ub:for index,shape=(1,set_len)
        :param int_value: Temporary value int32
        :param fp32_value:Temporary value fp32
        :param fp16tmp_ub: fp16tmp_ub shape=(16,)
        :param int_ub: int_ub shape=(16,)
        :param result_value:return (3,point_len)
        :param result_idx:return
        :param point_id: point num
        """
        with self.tik_inst.for_range(0, 3) as time:
            self.utilfp32.cmd_vec("vmuls", length, cmp_ub, input_ub, input_ub,
                                  0, 0, 0, 1)
            self._get_best_value("vmin", length, cmp_ub, tmp_ub, int_value,
                                 fp32_value)
            result_value[time, point_id] = cmp_ub[0]

            self._get_mask(length, input_ub, tmp_ub, cmp_ub, fp32_value)
            self._get_idx(length, index_ub, cmp_ub, tmp_ub, int_value,
                          fp32_value, fp16tmp_ub, int_ub)
            result_idx[time, point_id] = int_ub[0]
            int_value.set_as(int_ub[0])

            self.utilfp32.cmd_vec("vector_dup", 1, cmp_ub, cmp_ub, cmp_ub, 0,
                                  0, 0, 65535.0)
            input_ub[int_value] = cmp_ub[0]

    def _get_idx(self, length, index_ub, mask_ub, tmp_ub, int_value,
                 float_value, fp16_ub, int_ub):
        # get max index
        self.utilfp32.cmd_vec("vmul", length, mask_ub, mask_ub, index_ub, 0,
                              0, 0, 0)
        self._get_best_value("vmax", length, mask_ub, tmp_ub, int_value,
                             float_value)
        self.utilfp32.cmd_vec("vadds", 1, mask_ub, mask_ub, mask_ub, 0, 0,
                              0, 1 - length)
        self.utilfp32.cmd_vec("vabs", 1, mask_ub, mask_ub, mask_ub, 0, 0, 0, 0)
        self.utilfp32.round_f32toi32(mask_ub, tmp_ub, fp16_ub, int_ub)

    def _get_mask(self, length, fcompare_ub, res_ub, tmp_ub, min_value):
        min_value.set_as(tmp_ub[0])
        self.utilfp32.cmd_vec("vector_dup", length, tmp_ub, tmp_ub, tmp_ub, 0,
                              0, 0, min_value)
        self.utilfp32.cmd_vec("vsub", length, res_ub, tmp_ub, fcompare_ub, 0,
                              0, 0, 0)
        self.utilfp32.cmd_vec("vadds", length, tmp_ub, res_ub, res_ub, 0, 0,
                              0, 0.0000001)
        self.utilfp32.cmd_vec("vrec", length, tmp_ub, tmp_ub, tmp_ub, 0, 0, 0,
                              0)
        self.utilfp32.cmd_vec("vmul", length, tmp_ub, res_ub, tmp_ub, 0, 0, 0,
                              0)
        self.utilfp32.cmd_vec("vadds", length, tmp_ub, tmp_ub, tmp_ub, 0, 0, 0,
                              -1.0)
        self.utilfp32.cmd_vec("vabs", length, tmp_ub, tmp_ub, tmp_ub, 0, 0, 0,
                              0)
        self.utilfp32.cmd_vec("vadds", length, tmp_ub, tmp_ub, tmp_ub, 0, 0, 0,
                              -0.5)
        self.utilfp32.cmd_vec("vector_dup", length, res_ub, res_ub, res_ub, 0,
                              0, 0, 0.0)
        self.utilfp32.cmd_vec("vmax", length, tmp_ub, tmp_ub, res_ub, 0, 0, 0,
                              0)
        self.utilfp32.cmd_vec("vmuls", length, tmp_ub, tmp_ub, tmp_ub, 0, 0, 0,
                              2.0)

    def _get_best_value(self, cmd, length, min_res_ub, tmp_ub, value_int,
                        value_fp):
        '''
        support length < 64*255
        :param length: compare length
        :param min_res_ub: the ub should be compare
        :param value_int: the para of tmp_int32
        :param value_fp: the para of tmp_fp32
        '''
        length_1 = length - length % 16
        remain = length % 16
        value_int.set_as(length_1 // 2)
        if length_1 >= 16:
            with self.tik_inst.for_range(0, int(math.log(length_1, 2)) - 3
                                            + 1):
                self.utilfp32.cmd_vec_sp(cmd, value_int, min_res_ub,
                                         min_res_ub, min_res_ub, 0, 0,
                                         value_int, 0)
                value_int.set_as((((value_int // 2) + 7) // 8) * 8)

            with self.tik_inst.for_range(0, 8) as time:
                value_fp.set_as(min_res_ub[time])
                self.utilfp32.cmd_vec("vector_dup", 8, min_res_ub, min_res_ub,
                                      min_res_ub, 8, 8, 8, value_fp)
                self.utilfp32.cmd_vec(cmd, 8, min_res_ub, min_res_ub,
                                      min_res_ub, 0, 8, 0, 0)

        if remain > 0:
            with self.tik_inst.for_range(0, remain) as time_1:
                value_fp.set_as(min_res_ub[length_1 + time_1])
                self.utilfp32.cmd_vec("vector_dup", remain, tmp_ub, tmp_ub,
                                      tmp_ub, 0, 0, 0, value_fp)
                self.utilfp32.cmd_vec(cmd, remain, min_res_ub, min_res_ub,
                                      tmp_ub, 0, 0, 0, 0)
