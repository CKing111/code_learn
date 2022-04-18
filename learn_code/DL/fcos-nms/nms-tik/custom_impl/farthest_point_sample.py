import math
from ascend import AContainer1951
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from uti import interface_check
from version.get_version import get_aicore_container


class FPS:

    def __init__(self, container: AContainer1951, input_data, output,
                 point_num, kernel_n):

        self.tik_inst = container.tinst
        self.tik = container.tik
        self.cont = container
        self.num_per_cmd_fp32 = self.cont.get_vec_proc_num_per_cmd("float32")
        self.num_per_cmd_fp16 = self.cont.get_vec_proc_num_per_cmd("float16")

        interface_check.check_kernelname(kernel_n)
        interface_check.check_param(input_data, [4], ["float16"], ["NCHW"])
        interface_check.check_param(output, [4], ["int32"], ["NCHW"])
        if not isinstance(point_num, int) or point_num <= 0:
            raise RuntimeError("The type of point_num should be int and bigger"
                               " than zero, please check.")
        self.kernel_name = kernel_n
        self.src_b, self.src_c, self.src_k, self.src_n = input_data.get(
            'shape')
        self.dst_b, self.dst_c, self.dst_k, self.dst_n = output.get('shape')
        self.src_n16 = ((self.src_n * self.src_k + 15) // 16) * 16
        self.point_num16 = ((point_num + 15) // 16) * 16
        self.point_num = point_num
        # 512 is the GPU thread
        self.gpu_thread = min(512, self.src_n16)
        self.aicore_use = container.const_aicore_num
        if self.point_num % 16 == 0:
            self._aicore_in_use_select(self.src_b)
        else:
            self.aicore_use = 1
            self.xlen_each_core, self.xlen_last_core = self.src_b, self.src_b

        if self.src_n16 * 19 + self.point_num16 * 2 <= 126880:
            self.suffice = True
        else:
            self.suffice = False
        self._para_checkup()
        self.src_gm = self.tik_inst.Tensor(input_data.get('dtype'), (
            self.src_b, self.src_c, self.src_n * self.src_k),
                                           self.tik.scope_gm, "src_gm")
        self.dst_gm = self.tik_inst.Tensor(output.get('dtype'),
                                           (self.src_b, point_num),
                                           self.tik.scope_gm, "dst_gm")

    def _aicore_in_use_select(self, tiling_len):
        if tiling_len < self.aicore_use:
            self.aicore_use = tiling_len
            self.xlen_each_core = 1
            self.xlen_last_core = 1
        else:
            self.xlen_each_core = tiling_len // self.aicore_use
            self.xlen_last_core = tiling_len - self.xlen_each_core * (
                    self.aicore_use - 1)

    def _round_f32toi32(self, ub_fp32, ub_fp32_tmp, ub_fp16_tmp, ub_int32,
                        ub_int32tmp):
        """
        only process 16
        """
        self.tik_inst.vconv(16, "", ub_fp16_tmp, ub_fp32, 1, 1, 1, 4, 8)
        self.tik_inst.vconv(16, "round", ub_int32, ub_fp16_tmp, 1, 1, 1, 8, 4)
        self.tik_inst.vconv(16, "", ub_fp32_tmp, ub_fp16_tmp, 1, 1, 1, 8, 4)
        self.tik_inst.vsub(16, ub_fp32_tmp, ub_fp32, ub_fp32_tmp, 1, 1, 1, 1,
                           8, 8, 8)
        self.tik_inst.vconv(16, "", ub_fp16_tmp, ub_fp32_tmp, 1, 1, 1, 4, 8)
        self.tik_inst.vconv(16, "round", ub_int32tmp, ub_fp16_tmp, 1, 1, 1, 8,
                            4)
        self.tik_inst.vadd(16, ub_int32, ub_int32tmp, ub_int32, 1, 1, 1, 1, 8,
                           8, 8)

    def _init_index_reverse(self, int_value, float_value, ub_length, index_ub,
                            index_int_ub, fp16_ub, need_1):
        max_fortime = int(math.ceil(math.log(ub_length, 2)))
        max_remain = ub_length - int(math.pow(2, max_fortime - 1))
        offset = int(math.pow(2, max_fortime)) - ub_length
        with self.tik_inst.for_range(0, 16) as n_dix:
            index_int_ub[n_dix] = 16 - n_dix
        # int32_2_fp16
        self.tik_inst.vconv(16, "", fp16_ub, index_int_ub, 1, 1, 1, 1, 2, 1.0)
        self.tik_inst.vconv(16, "", index_ub[ub_length - 16], fp16_ub, 1, 1, 1,
                            1, 2)
        src_offset = self.tik_inst.Scalar("int32")
        int_value.set_as(16)
        if ub_length > 16:
            with self.tik_inst.for_range(0, max_fortime - 5):
                float_value.set_as(index_ub[ub_length - int_value])
                src_offset.set_as(ub_length - int_value)
                bufs = {"dst": AVecBuf(index_ub, int_value,
                                       ub_length - int_value * 2, self.cont,
                                       False, self.num_per_cmd_fp32),
                        "src": AVecBuf(index_ub, int_value, src_offset,
                                       self.cont, False,
                                       self.num_per_cmd_fp32)}
                cmds = [VecGCmd("vadds", dst_name="dst", src0_name="src",
                                scalar=float_value)]
                VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "src")

                int_value.set_as(2 * int_value)
            float_value.set_as(index_ub[ub_length - int_value])
            bufs = {"dst": AVecBuf(index_ub, max_remain, 0, self.cont, False,
                                   self.num_per_cmd_fp32),
                    "src": AVecBuf(index_ub, max_remain,
                                   ub_length - int_value + offset, self.cont,
                                   False, self.num_per_cmd_fp32)}
            cmds = [VecGCmd("vadds", dst_name="dst", src0_name="src",
                            scalar=float_value)]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "src")
        if need_1:
            bufs = {"dst": AVecBuf(index_ub, ub_length, 0, self.cont, False,
                                   self.num_per_cmd_fp32),
                    "src": AVecBuf(index_ub, ub_length, 0, self.cont, False,
                                   self.num_per_cmd_fp32)}
            cmds = [
                VecGCmd("vadds", dst_name="dst", src0_name="src", scalar=-1.0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "src")

    def _para_checkup(self):
        if self.src_c != 3:
            raise RuntimeError(
                "Please check the channel of input data which should be 3.")
        if self.src_n * self.src_k < self.point_num:
            raise RuntimeError(
                " The number of sampling points should be less than the "
                "number of input points.")
        if self.src_n16 * 9 + self.point_num16 * 2 > 11520 * 11:
            raise RuntimeError(
                "The number of points have exceeds the supported range.")
        if self.src_n != 1 and self.src_k != 1:
            raise RuntimeError(
                "The module of fps input should be N31D or N3D1.")
        if self.dst_b != self.src_b:
            raise RuntimeError(
                "The batch of fps output should be same as input.")
        if self.dst_c != self.point_num:
            raise RuntimeError(
                "The point of fps output should be same as attr npoint.")
        if self.dst_k != 1 or self.dst_n != 1:
            raise RuntimeError("The shape of fps output should be ND11.")

    def init_offset_ub(self, offset_ub):
        self._init_index_reverse(self.slr_int32, self.slr_fp32,
                                 self.gpu_thread, offset_ub, self.tmp_intub,
                                 self.input_fp16, False)
        gpu_repeat = self.src_n16 // self.gpu_thread
        gpu_remain = self.src_n16 - gpu_repeat * self.gpu_thread
        for time in range(0, self.src_n16 // self.gpu_thread):
            bufs = {"dst": AVecBuf(offset_ub, self.gpu_thread,
                                   time * self.gpu_thread, self.cont, False,
                                   self.num_per_cmd_fp32),
                    "src": AVecBuf(offset_ub, self.gpu_thread, 0, self.cont,
                                   False, self.num_per_cmd_fp32)}
            cmds = [
                VecGCmd("vmuls", dst_name="dst", src0_name="src", scalar=1.0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "src")

        if gpu_remain > 0:
            bufs = {"dst": AVecBuf(offset_ub, gpu_remain, (
                    self.src_n16 // self.gpu_thread) * self.gpu_thread,
                                   self.cont, False, self.num_per_cmd_fp32),
                    "src": AVecBuf(offset_ub, gpu_remain, 0, self.cont, False,
                                   self.num_per_cmd_fp32)}
            cmds = [
                VecGCmd("vmuls", dst_name="dst", src0_name="src", scalar=1.0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "src")

    def compute(self):
        self.tmp_num = self.tik_inst.Scalar("int32")
        with self.tik_inst.for_range(0, self.aicore_use,
                                     block_num=self.aicore_use) as index:
            self.apply_for_cpt_ub()
            self.slr_int32 = self.tik_inst.Scalar("int32")
            self.max_id = self.tik_inst.Scalar("int32")
            self.slr_fp32 = self.tik_inst.Scalar("float32")
            self.slr_fp16 = self.tik_inst.Scalar("float16")
            # suffice means that we can transport data once
            if self.suffice:
                self.input_fp32 = self.tik_inst.Tensor("float32", (
                    self.src_n16 * self.src_c,), self.tik.scope_ubuf,
                                                       "input_fp32")
                self.index_ub = self.tik_inst.Tensor("float32",
                                                     (self.src_n16,),
                                                     self.tik.scope_ubuf,
                                                     "index_ub")
                self.offset_ub = self.tik_inst.Tensor("float32",
                                                      (self.src_n16,),
                                                      self.tik.scope_ubuf,
                                                      "offset_ub")
                self._init_index_reverse(self.slr_int32, self.slr_fp32,
                                         self.src_n16, self.index_ub,
                                         self.tmp_intub, self.input_fp16, True)
                self.init_offset_ub(self.offset_ub)

            with self.tik_inst.if_scope(index < self.aicore_use - 1):
                self._compute_eachcore(self.tmp_num, index,
                                       self.xlen_each_core)
            with self.tik_inst.else_scope():
                self._compute_eachcore(self.tmp_num, index,
                                       self.xlen_last_core)

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.src_gm], outputs=[self.dst_gm],
                               enable_l2=True)

        return self.tik_inst

    def apply_for_cpt_ub(self):
        self.input_fp16 = self.tik_inst.Tensor("float16", (self.src_n16,),
                                               self.tik.scope_ubuf,
                                               "input_fp16")
        self.tmp_ub = self.tik_inst.Tensor("float32", (self.src_n16,),
                                           self.tik.scope_ubuf, "tmp_ub")
        self.mask_ub = self.tik_inst.Tensor("float32", (self.src_n16,),
                                            self.tik.scope_ubuf, "mask_ub")
        self.res_ub = self.tik_inst.Tensor("float32", (self.src_n16,),
                                           self.tik.scope_ubuf, "res_ub")
        self.minres_ub = self.tik_inst.Tensor("float32", (self.src_n16,),
                                              self.tik.scope_ubuf, "minres_ub")
        self.output_idx = self.tik_inst.Tensor("int32", (self.point_num16,),
                                               self.tik.scope_ubuf,
                                               "output_idx")

        self.max_ub = self.tik_inst.Tensor("float32", (16,),
                                           self.tik.scope_ubuf, "max_ub")
        self.tmp_intub = self.tik_inst.Tensor("int32", (16,),
                                              self.tik.scope_ubuf, "tmp_intub")
        self.maxidx_ub = self.tik_inst.Tensor("int32", (16,),
                                              self.tik.scope_ubuf, "maxidx_ub")

    def _number1_slice_datamove(self, index, batch_once):
        if self.src_b == 1 and self.point_num == 1:
            self.tik_inst.data_move(self.dst_gm, self.output_idx, 0, 1,
                                    self.point_num16 // 8, 0, 0)
        else:
            self.tik_inst.data_move(self.dst_gm[self.point_num * (
                    index * self.xlen_each_core + batch_once)],
                                    self.output_idx, 0, 1,
                                    self.point_num16 // 8, 0, 0)

    def _compute_eachcore(self, tmp_num, index, eachcore_len):
        # transpose_gm2ub
        with self.tik_inst.for_range(0, eachcore_len) as batch_once:
            if self.suffice:
                with self.tik_inst.for_range(0, self.src_c) as c_time:
                    self.tik_inst.data_move(self.input_fp16, self.src_gm[
                        index * self.xlen_each_core + batch_once, c_time, 0],
                                            0, 1, self.src_n16 // 16, 0, 0)
                    bufs = {"dst_fp32": AVecBuf(self.input_fp32, self.src_n16,
                                                self.src_n16 * c_time,
                                                self.cont, False,
                                                self.num_per_cmd_fp32),
                            "src_fp16": AVecBuf(self.input_fp16, self.src_n16,
                                                0, self.cont, False,
                                                self.num_per_cmd_fp32)}
                    cmds = [VecGCmd("vconv", "dst_fp32", "src_fp16",
                                    round_mode="")]
                    VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds,
                                                "src_fp16")

            # init minres_ub
            bufs = {
                "dst_fp32": AVecBuf(self.minres_ub, self.src_n16, 0, self.cont,
                                    False, self.num_per_cmd_fp32)}
            cmds = [VecGCmd("vector_dup", "dst_fp32", scalar=65535.0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "dst_fp32")

            # make sure self.output_idx[0] is 0
            self.tik_inst.vector_dup(8, self.output_idx, 0, 1, 1, 1)
            with self.tik_inst.for_range(0, self.point_num16 - 1,
                                         thread_num=2) as tmp_time:
                tmp_num.set_as(tmp_time)
                self.cal_each_tmpidx(tmp_num, self.slr_int32, self.slr_fp32,
                                     self.tmp_ub, self.res_ub, self.minres_ub,
                                     self.output_idx, index, batch_once)

            self._number1_slice_datamove(index, batch_once)

    def cal_each_tmpidx(self, tmp_num, tmp_idx, tmp_idxnum, tmp_ub, res_ub,
                        minres_ub, output_idx, index, batch_once):
        bufs = {"res_ub": AVecBuf(res_ub, self.src_n16, 0, self.cont, False,
                                  self.num_per_cmd_fp32),
                "minres_ub": AVecBuf(minres_ub, self.src_n16, 0, self.cont,
                                     False, self.num_per_cmd_fp32), }
        tmp_idx.set_as(output_idx[tmp_num])
        # init res_ub->0
        cmds = [VecGCmd("vector_dup", "res_ub", scalar=0.0)]
        VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "res_ub")

        self._cal_dist_euclidean(self.src_n16, tmp_idx, tmp_idxnum, tmp_ub,
                                 res_ub, index, batch_once)
        # get the min td
        cmds = [VecGCmd("vmin", dst_name="minres_ub", src0_name="minres_ub",
                        src1_name="res_ub")]
        VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "res_ub")

        # find the max d
        self._get_max_idx_fp32(minres_ub, res_ub, tmp_ub, tmp_idx, tmp_idxnum)
        output_idx[tmp_num + 1] = tmp_idx

    def _get_mask(self, length, tmp_ub, fcompare_ub, res_ub, max_value):
        '''
        :param length: process length
        :param tmp_ub: return mask
        :param fcompare_ub:origin data,should not be changed
        :param res_ub: tmp ub
        :param max_value: max value
        :return:tmp_ub
        '''
        bufs = {"tmp_ub": AVecBuf(tmp_ub, length, 0, self.cont, False,
                                  self.num_per_cmd_fp32),
                "fcompare_ub": AVecBuf(fcompare_ub, length, 0, self.cont,
                                       False, self.num_per_cmd_fp32),
                "res_ub": AVecBuf(res_ub, length, 0, self.cont, False,
                                  self.num_per_cmd_fp32)}
        cmds = [VecGCmd("vector_dup", "tmp_ub", scalar=max_value),
                VecGCmd("vsub", dst_name="res_ub", src0_name="tmp_ub",
                        src1_name="fcompare_ub"),
                VecGCmd("vadds", dst_name="tmp_ub", src0_name="res_ub",
                        scalar=0.0000001),
                VecGCmd("vrec", dst_name="tmp_ub", src0_name="tmp_ub"),
                VecGCmd("vmul", dst_name="tmp_ub", src0_name="tmp_ub",
                        src1_name="res_ub"),
                VecGCmd("vadds", dst_name="tmp_ub", src0_name="tmp_ub",
                        scalar=-1.0),
                VecGCmd("vabs", dst_name="tmp_ub", src0_name="tmp_ub"),
                VecGCmd("vadds", dst_name="tmp_ub", src0_name="tmp_ub",
                        scalar=-0.5),
                VecGCmd("vector_dup", "res_ub", scalar=0.0),
                VecGCmd("vmax", dst_name="tmp_ub", src0_name="tmp_ub",
                        src1_name="res_ub"),
                VecGCmd("vmuls", dst_name="tmp_ub", src0_name="tmp_ub",
                        scalar=2.0)]
        VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "res_ub")

    def _get_max_idx_fp32(self, fcompare_ub, res_ub, tmp_ub, max_id,
                          max_value):
        bufs = self._apply_for_getmax_bufs(tmp_ub, fcompare_ub, res_ub)
        # get_max
        self.tik_inst.vector_dup(16, tmp_ub[self.src_n16 - 16], -1.0, 1, 1, 1)
        cmds = [
            VecGCmd("vmuls", dst_name="n_tmp_ub", src0_name="n_fcompare_ub",
                    scalar=1.0)]
        VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "n_tmp_ub")
        self._get_max_value(self.src_n16, self.max_ub, tmp_ub, max_id,
                            max_value)
        # max_mask.tmp
        max_value.set_as(self.max_ub[0])
        self.tik_inst.vector_dup(16, tmp_ub[self.src_n16 - 16], 0.0, 1, 1, 1)

        self._get_mask(self.src_n * self.src_k, tmp_ub, fcompare_ub, res_ub,
                       max_value)
        if self.suffice:
            bufs["offset_ub"] = AVecBuf(self.offset_ub, self.src_n16, 0,
                                        self.cont, False,
                                        self.num_per_cmd_fp32)
            offset_cmds = [
                VecGCmd("vmul", dst_name="tmp_ub", src0_name="tmp_ub",
                        src1_name="offset_ub")]
        else:
            self.init_offset_ub(res_ub)
            offset_cmds = [
                VecGCmd("vmul", dst_name="tmp_ub", src0_name="tmp_ub",
                        src1_name="res_ub")]

        offset_cmds.append(
            VecGCmd("vmuls", dst_name="res_ub", src0_name="tmp_ub",
                    scalar=1.0))
        VecGExecutor.exec_vec_g_cmd(self.cont, bufs, offset_cmds, "tmp_ub")

        # get max
        self._get_max_value(self.src_n16, self.max_ub, res_ub, max_id,
                            max_value)
        max_value.set_as(self.max_ub[0])

        self._get_mask(self.src_n16, self.mask_ub, tmp_ub, res_ub, max_value)
        # get max index
        if self.suffice:
            bufs["index_ub"] = AVecBuf(self.index_ub, self.src_n16, 0,
                                       self.cont, False, self.num_per_cmd_fp32)
            cmds = [VecGCmd("vmul", dst_name="tmp_ub", src0_name="mask_ub",
                            src1_name="index_ub")]
        else:
            self._init_index_reverse(self.slr_int32, self.slr_fp32,
                                     self.src_n16, res_ub, self.tmp_intub,
                                     self.input_fp16, True)
            cmds = [VecGCmd("vmul", dst_name="tmp_ub", src0_name="mask_ub",
                            src1_name="res_ub")]
        VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "tmp_ub")

        self._get_max_value(self.src_n16, self.max_ub, tmp_ub, max_id,
                            max_value)

        self._round_f32toi32(tmp_ub, res_ub, self.input_fp16, self.maxidx_ub,
                             self.tmp_intub)
        max_id.set_as(self.maxidx_ub[0])
        max_id.set_as(self.src_n16 - 1 - max_id)

    def _apply_for_getmax_bufs(self, tmp_ub, fcompare_ub, res_ub):
        bufs = {"tmp_ub": AVecBuf(tmp_ub, self.src_n16, 0, self.cont, False,
                                  self.num_per_cmd_fp32),
                "fcompare_ub": AVecBuf(fcompare_ub, self.src_n16, 0, self.cont,
                                       False, self.num_per_cmd_fp32),
                "res_ub": AVecBuf(res_ub, self.src_n16, 0, self.cont, False,
                                  self.num_per_cmd_fp32),
                "n_tmp_ub": AVecBuf(tmp_ub, self.src_n * self.src_k, 0,
                                    self.cont, False, self.num_per_cmd_fp32),
                "n_fcompare_ub": AVecBuf(fcompare_ub, self.src_n * self.src_k,
                                         0, self.cont, False,
                                         self.num_per_cmd_fp32),
                "mask_ub": AVecBuf(self.mask_ub, self.src_n16, 0, self.cont,
                                   False, self.num_per_cmd_fp32), }
        return bufs

    def _get_max_value(self, length, max_ub, min_res_ub, value_int, value_fp):
        '''
        support length < 64*255
        :param length: compare length
        :param min_res_ub: the ub should be compare, will be amend
        :param max_ub: return the max_value(16)
        :param value_int: the para of tmp_int32
        :param value_fp: the para of tmp_fp32
        '''
        value_int.set_as(length // 2)
        with self.tik_inst.for_range(0, int(math.log(length, 2)) - 4 + 1):
            bufs = {
                "former": AVecBuf(min_res_ub, value_int, 0, self.cont, False,
                                  self.num_per_cmd_fp32),
                "latter": AVecBuf(min_res_ub, value_int, value_int, self.cont,
                                  False, self.num_per_cmd_fp32), }
            cmds = [VecGCmd("vmax", dst_name="former", src0_name="former",
                            src1_name="latter")]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "latter")

            value_int.set_as((((value_int // 2) + 15) // 16) * 16)

        with self.tik_inst.for_range(0, 16) as time:
            value_int.set_as(time)
            value_fp.set_as(min_res_ub[value_int])
            self.tik_inst.vector_dup(16, max_ub, value_fp, 1, 1, 1)
            self.tik_inst.vmax(16, min_res_ub, max_ub, min_res_ub, 1, 1, 1, 1,
                               0, 0, 0)

    def _cal_dist_euclidean(self, cal_length, cur_idx, cur_idxnum, tmp_ub,
                            res_ub, index, batch_once):
        bufs = {"tmp_ub": AVecBuf(tmp_ub, cal_length, 0, self.cont, False,
                                  self.num_per_cmd_fp32),
                "res_ub": AVecBuf(res_ub, cal_length, 0, self.cont, False,
                                  self.num_per_cmd_fp32),
                "input_fp16": AVecBuf(self.input_fp16, cal_length, 0,
                                      self.cont, False,
                                      self.num_per_cmd_fp32), }
        for i in range(0, self.src_c):
            if self.suffice:
                cur_idxnum.set_as(self.input_fp32[i * cal_length + cur_idx])
                bufs["input_fp32"] = AVecBuf(self.input_fp32, cal_length,
                                             i * cal_length, self.cont, False,
                                             self.num_per_cmd_fp32)
                cmds = [
                    VecGCmd("vmuls", dst_name="tmp_ub", src0_name="input_fp32",
                            scalar=-1.0)]
                VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "tmp_ub")
            else:
                self.tik_inst.data_move(self.input_fp16, self.src_gm[
                    index * self.xlen_each_core + batch_once, i, 0], 0, 1,
                                        self.src_n16 // 16, 0, 0)

                cmds = [
                    VecGCmd("vconv", dst_name="tmp_ub", src0_name="input_fp16",
                            round_mode="")]
                VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds, "tmp_ub")
                cur_idxnum.set_as(tmp_ub[cur_idx])

                else_cmds = [
                    VecGCmd("vmuls", dst_name="tmp_ub", src0_name="tmp_ub",
                            scalar=-1.0)]
                VecGExecutor.exec_vec_g_cmd(self.cont, bufs, else_cmds,
                                            "tmp_ub")
            same_cmds = [
                VecGCmd("vadds", dst_name="tmp_ub", src0_name="tmp_ub",
                        scalar=cur_idxnum),
                VecGCmd("vmul", dst_name="tmp_ub", src0_name="tmp_ub",
                        src1_name="tmp_ub"),
                VecGCmd("vadd", dst_name="res_ub", src0_name="tmp_ub",
                        src1_name="res_ub")]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, same_cmds, "tmp_ub")

    def tik_output_debug(self):
        return self.tik_inst


def farthest_point_sample(input_data, output, point_num, kernel_name="fps",
                          test=False):
    '''
    Puts update data into the output shape based on the indice.
    Args:
        param input_shape: the dict of indice(index)
        param output:the result of fps
        param point_num: the value of point_num,should be int
        param kernel_n: the name of kernel
    '''
    container = get_aicore_container(("Ascend610",), c3x_support_list=())
    obj = FPS(container, input_data, output, point_num, kernel_name)

    obj.compute()
    if test:
        return obj.tik_output_debug()
    else:
        return 0
