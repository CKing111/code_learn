# -*- coding:utf-8 -*-
from functools import reduce

from ascend import AContainer
from ascend import AContainer1951
from ascend import CoreType
from ascend import SoftArch
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ascend import Tilerv2
from uti import interface_check
from version.get_version import get_aicore_container

from ._revert_idx import RevertIndex
from ._grid_bilinear import Bilinear
from ._grid_nearest import Nearest
from ._grid_interp import VariableMem


class GridSample(Tilerv2):
    def __init__(self, container, core_type, input_data, grid_data, mode,
                 padding_mode, align_corners, output, kernel_n, debug=False):
        super(GridSample, self).__init__(container, core_type)
        interface_check.check_kernelname(kernel_n)
        self._param_check(align_corners, int, [True, False])
        self._param_check(mode, str, ["bilinear", "nearest"])
        self._param_check(padding_mode, str, ["zeros", "border", "reflection"])
        interface_check.check_param(grid_data, [4, 5], ["float16"],
                                    ["NCHW", "ND"])
        interface_check.check_param(input_data, [5, 6], ["float16"],
                                    ["NC1HWC0", "ND"])
        interface_check.check_param(output, [5, 6], ["float16"],
                                    ["NC1HWC0", "ND"])

        self.debug = debug
        self.cont = container
        self.tik = container.tik
        self.tinst = container.tinst
        self.max_core_n = self.cont.const_aicore_num
        self.kernel_name = kernel_n
        self.mode = mode
        self.padding_mode = padding_mode
        self.num_per_cmd = self.cont.get_vec_proc_num_per_cmd("float32")
        self._shape_checkandreshape(input_data, grid_data, output)
        self._boundry_value_check(input_data.get('shape'),
                                  grid_data.get('shape'), output.get('shape'))
        self.apply_gm_and_reshape(input_data, grid_data, output)
        print("[INFO][GridSample]:", mode, padding_mode, align_corners)
        self.var_mem = VariableMem()
        self.revindex = RevertIndex(self.cont, padding_mode, align_corners)

        if mode == "nearest":
            # (2int + 1legalmask + channnel) * onceprocess
            self.once_process = (container.const_ub_max_byte // 2) // (
                (6 + 16 * self.in_c1 + (self.dims - 2) * 2))
            self.nearest = Nearest(self.cont, self.in_list, self.out_list)
        else:
            # (2int + 1legalmask +1fp16 + 2fp32 + 3fp32 + channnel * 2) *
            # onceprocess
            self.once_process = (container.const_ub_max_byte // 2) // (
                (17 + 32 * self.in_c1 + (self.dims - 2) * 6))
            self.bilinear = Bilinear(self.cont, self.in_list, self.out_list)
        if self.once_process < 1:
            raise RuntimeError(
                "check your input channel to support once_process >= 1.")

    def _param_check(self, param, ptype, support_list):
        if not isinstance(param, ptype):
            raise RuntimeError(
                "[ERROR][GridSample]: input param should be {}, but given is "
                "{}, "
                "please check!".format(ptype, type(param)))
        if param not in support_list:
            raise RuntimeError(
                "[ERROR][GridSample]: param only support {}, but given is {}, "
                "please check!".format(support_list, param))

    def _boundry_value_check(self, grid_shape, input_shape, output_shape):
        grid_num = reduce(lambda x, y: x * y, grid_shape)
        input_num = reduce(lambda x, y: x * y, input_shape)
        output_num = reduce(lambda x, y: x * y, output_shape)
        if grid_num + input_num + output_num > 1024 * 1024 * 1024 // 4:
            raise RuntimeError("[ERROR][GridSample]: process num has"
                               " exceed the workspace of device.")

        if self.ou_c0 != 16 or self.in_c0 != 16:
            raise RuntimeError(
                "[ERROR][GridSample]: c0 should be same with 16.")
        if self.in_n != self.out_n or self.out_n != self.ou_n:
            raise RuntimeError(
                "[ERROR][GridSample]: the N of each shape should be same.")
        if self.ou_c1 != self.in_c1 or self.ou_depth != self.out_depth or \
                self.ou_h != self.out_h or self.ou_w != self.out_w:
            raise RuntimeError(
                "[ERROR][GridSample]: output shape which is invalid, please "
                "check.")

    def _shape_checkandreshape(self, input_data, grid_data, output):
        grid_shape = grid_data.get('shape')
        input_shape = input_data.get('shape')
        output_shape = output.get('shape')
        if grid_shape[1] == 2 and len(grid_shape) == 4 and len(
                input_shape) == 5 and len(output_shape) == 5:
            self.out_n, self.dims, self.out_h, self.out_w = grid_shape
            self.in_n, self.in_c1, self.in_h, self.in_w, self.in_c0 = \
                input_shape
            self.ou_n, self.ou_c1, self.ou_h, self.ou_w, self.ou_c0 = \
                output_shape
            self.in_depth, self.out_depth, self.ou_depth = 1, 1, 1
            self.in_list = [self.in_w, self.in_h]
            self.out_list = [self.out_w, self.out_h]
        elif grid_shape[1] == 3 and len(grid_shape) == 5 and len(
                input_shape) == 6 and len(output_shape) == 6:
            self.out_n, self.dims, self.out_depth, self.out_h, self.out_w = \
                grid_shape
            self.in_n, self.in_depth, self.in_c1, self.in_h, self.in_w, \
            self.in_c0 = input_shape
            self.ou_n, self.ou_depth, self.ou_c1, self.ou_h, self.ou_w, \
            self.ou_c0 = output_shape
            self.in_list = [self.in_w, self.in_h, self.in_depth]
            self.out_list = [self.out_w, self.out_h, self.out_depth]
        else:
            raise RuntimeError(
                "[ERROR][GridSample]: only support 2D and 3D, your input grid"
                "is not valid which is {}".format(grid_shape[1]))

    def apply_gm_and_reshape(self, input_data, grid_data, output):
        grid_shape = grid_data.get('shape')
        input_shape = input_data.get('shape')
        output_shape = output.get('shape')
        self.input_gm = self.tinst.Tensor(input_data.get('dtype'), input_shape,
                                          self.tik.scope_gm, "input_gm")
        self.grid_gm = self.tinst.Tensor(grid_data.get('dtype'), grid_shape,
                                         self.tik.scope_gm, "grid_gm")
        self.dst_gm = self.tinst.Tensor(output.get('dtype'), output_shape,
                                        self.tik.scope_gm, "dst_gm")
        self.input_gm = self.input_gm.reshape((
            self.in_n, self.in_depth, self.in_c1, self.in_h, self.in_w,
            self.in_c0))
        self.dst_gm = self.dst_gm.reshape((
            self.in_n, self.out_depth, self.in_c1, self.out_h * self.out_w,
            self.in_c0))

        self.grid_gm = self.grid_gm.reshape(
            (self.out_n, self.dims, self.out_depth, self.out_h * self.out_w))

    def compute(self):
        self._calc_xcore_info(self.out_h * self.out_w, 0, self.max_core_n)
        self._compute_multi_core(1, self.once_process, self.max_core_n)
        self.tinst.BuildCCE(kernel_name=self.kernel_name,
                            inputs=[self.input_gm, self.grid_gm],
                            outputs=[self.dst_gm], enable_l2=True)

    def _compute_one_loop(self, start, loop_len):
        applied_looplen = (loop_len + 15) // 16 * 16
        cal_ub_list = (
            self.var_mem.grid_x, self.var_mem.grid_y, self.var_mem.grid_z)
        cal_ub_dict = {self.var_mem.grid_x: self.in_w,
                       self.var_mem.grid_y: self.in_h,
                       self.var_mem.grid_z: self.in_depth}
        if self.mode == "nearest":
            self._compute_nearest(start, loop_len, applied_looplen,
                                  cal_ub_list, cal_ub_dict)
        else:
            self._compute_bilinear(start, loop_len, applied_looplen,
                                   cal_ub_list, cal_ub_dict)

    def _compute_bilinear(self, start, loop_len, applied_looplen, cal_ub_list,
                          cal_ub_dict):
        tmp_ub = self.tinst.Tensor("float16", (applied_looplen,),
                                   self.tik.scope_ubuf, "tmp_ub")
        bufs = {"tmp_ub": AVecBuf(tmp_ub, loop_len, 0, self.cont, False,
                                  self.num_per_cmd),
                self.var_mem.grid_x: AVecBuf.create((applied_looplen,),
                                                    "float32",
                                                    self.tik.scope_ubuf,
                                                    self.var_mem.grid_x,
                                                    self.cont),
                self.var_mem.grid_y: AVecBuf.create((applied_looplen,),
                                                    "float32",
                                                    self.tik.scope_ubuf,
                                                    self.var_mem.grid_y,
                                                    self.cont), }
        if self.dims == 3:
            bufs[self.var_mem.grid_z] = AVecBuf.create((applied_looplen,),
                                                       "float32",
                                                       self.tik.scope_ubuf,
                                                       self.var_mem.grid_z,
                                                       self.cont)
        with self.tinst.for_range(0, self.in_n) as cur_batch:
            with self.tinst.for_range(0, self.out_depth) as cur_depth:
                # compute each dim's coordination
                self._calc_revindex(tmp_ub, cur_batch, cur_depth, start,
                                    loop_len, cal_ub_list, cal_ub_dict, bufs)
                # cal interp, got dst
                self.dst_gm = self.bilinear.bilinear(loop_len, start, bufs,
                                                     self.in_c1 * self.in_c0,
                                                     self.input_gm,
                                                     self.dst_gm,
                                                     self.padding_mode,
                                                     cur_batch, cur_depth)

    def _compute_nearest(self, start, loop_len, applied_looplen, cal_ub_list,
                         cal_ub_dict):
        with self.tinst.for_range(0, self.in_n) as cur_batch:
            with self.tinst.for_range(0, self.out_depth) as cur_depth:
                # compute each dim's coordination
                legal_mask = self.tinst.Tensor("int32", (loop_len,),
                                               self.tik.scope_ubuf,
                                               "legal_mask")
                x_loc_int = self.tinst.Tensor("int32", (loop_len,),
                                              self.tik.scope_ubuf,
                                              self.var_mem.x_loc_int)
                y_loc_int = self.tinst.Tensor("int32", (loop_len,),
                                              self.tik.scope_ubuf,
                                              self.var_mem.y_loc_int)
                if self.dims == 3:
                    z_loc_int = self.tinst.Tensor("int32", (loop_len,),
                                                  self.tik.scope_ubuf,
                                                  self.var_mem.z_loc_int)
                else:
                    z_loc_int = None
                with self.tinst.new_stmt_scope():
                    interme_ub = self.tinst.Tensor("float32", (loop_len,),
                                                   self.tik.scope_ubuf,
                                                   "interme_ub")
                    tmp_ub, bufs = self._near_apply_memory(applied_looplen,
                                                           loop_len, x_loc_int,
                                                           y_loc_int,
                                                           z_loc_int)
                    self._calc_revindex(tmp_ub, cur_batch, cur_depth, start,
                                        loop_len, cal_ub_list, cal_ub_dict,
                                        bufs)
                    self.nearest.nearest_cal(loop_len, bufs, self.padding_mode,
                                             interme_ub, legal_mask)
                with self.tinst.new_stmt_scope():
                    seq_ub = self.tinst.Tensor("float16",
                                               (self.in_c1, loop_len, 16),
                                               self.tik.scope_ubuf, "seq_ub")
                    self.nearest.nearest_gotgm(loop_len, start,
                                               self.in_c0 * self.in_c1,
                                               self.input_gm, self.dst_gm,
                                               legal_mask, cur_batch,
                                               cur_depth, bufs, seq_ub,
                                               self.padding_mode)

    def _calc_revindex(self, tmp_ub, cur_batch, cur_depth, start, loop_len,
                       cal_ub_list, cal_ub_dict, bufs):
        for dims in range(self.dims):
            self.tinst.data_move(tmp_ub, self.grid_gm[
                cur_batch, dims, cur_depth, start], 0, 1,
                                 (loop_len + 15) // 16, 0, 0)
            cmds = [
                VecGCmd("vconv", cal_ub_list[dims], "tmp_ub", round_mode="")]
            VecGExecutor.exec_vec_g_cmd(self.cont, bufs, cmds,
                                        cal_ub_list[dims])
            self.revindex.compute_origin_index(loop_len, bufs[
                cal_ub_list[dims]].const_tensor,
                                               cal_ub_dict[cal_ub_list[dims]])

    def _near_apply_memory(self, applied_looplen, loop_len, x_loc_int,
                           y_loc_int, z_loc_int):
        tmp_ub = self.tinst.Tensor("float16", (applied_looplen,),
                                   self.tik.scope_ubuf, "tmp_ub")
        grid_x = self.tinst.Tensor("float32", (applied_looplen,),
                                   self.tik.scope_ubuf, self.var_mem.grid_x)
        grid_y = self.tinst.Tensor("float32", (applied_looplen,),
                                   self.tik.scope_ubuf, self.var_mem.grid_y)
        bufs = {"tmp_ub": AVecBuf(tmp_ub, loop_len, 0, self.cont, False,
                                  self.num_per_cmd),
                self.var_mem.x_loc_int: AVecBuf(x_loc_int, loop_len, 0,
                                                self.cont, False,
                                                self.num_per_cmd),
                self.var_mem.y_loc_int: AVecBuf(y_loc_int, loop_len, 0,
                                                self.cont, False,
                                                self.num_per_cmd),
                self.var_mem.grid_x: AVecBuf(grid_x, applied_looplen, 0,
                                             self.cont, False,
                                             self.num_per_cmd),
                self.var_mem.grid_y: AVecBuf(grid_y, applied_looplen, 0,
                                             self.cont, False,
                                             self.num_per_cmd), }
        if self.dims == 3:
            grid_z = self.tinst.Tensor("float32", (applied_looplen,),
                                       self.tik.scope_ubuf,
                                       self.var_mem.grid_z)
            bufs[self.var_mem.grid_z] = AVecBuf(grid_z, applied_looplen, 0,
                                                self.cont, False,
                                                self.num_per_cmd)
            bufs[self.var_mem.z_loc_int] = AVecBuf(z_loc_int, loop_len, 0,
                                                   self.cont, False,
                                                   self.num_per_cmd)
        return tmp_ub, bufs

    def tik_output_debug(self):
        return self.tinst


def grid_sample(input_data, grid_data, output, mode, padding_mode,
                align_corners, kernel_n="grid_sample", test=False):
    interface_check.check_kernelname(kernel_n)
    container = get_aicore_container(("Ascend610",), c3x_support_list=())
    core_type = CoreType.AICORE

    obj = GridSample(container, core_type, input_data, grid_data, mode,
                     padding_mode, align_corners, output, kernel_n, debug=test)
    obj.compute()
    if test:
        return obj.tik_output_debug()
    else:
        return 0
