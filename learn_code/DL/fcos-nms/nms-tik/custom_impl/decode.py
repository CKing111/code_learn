# -*- coding:utf-8 -*-
from functools import reduce
import numpy as np
from uti import interface_check
from ascend import AContainer1910
from ascend import AContainer1951
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ascend import Tilerv2
from ascend import CoreType
from version.get_version import get_aicore_container

# the number of ub tensors multiplied by the number of bytes
TENSOR_NUM = 36


class Decode(Tilerv2):
    """
    Introduction
    ------------
        Intialize Decode parameter
    Parameters
    ----------
        @boxpredictor, IN: input box shape, NCHW, ND
        @anchors, IN: input anchors shape, NCHW, ND
        @out_box, IN: output box shape, NCHW, ND
        @scale_facotr1, IN: divisor attribute, float
        @scale_facotr2, IN: divisor attribute, float
        @scale_factor3, IN: divisor attribute, float
        @scale_factor4, IN: divisor attribute, float
        @kernel name, IN: Decode kernel name
    Returns
    -------
    """

    def __init__(self, container, core_type, boxpredictor, anchors, out_box,
                 scale_factor1, scale_factor2, scale_factor3, scale_factor4,
                 cmp_value, kernel_name="Decode"):
        super(Decode, self).__init__(container, core_type)
        self._aicore_check(container, boxpredictor, anchors, out_box)
        self.container = container
        self.tinst = self.container.tinst
        gm_scope = self.container.tik.scope_gm

        self.box_shape = boxpredictor.get("shape")
        self.box_dtype = boxpredictor.get("dtype")
        self.out_box_shape = out_box.get("shape")
        self.out_dtype = out_box.get("dtype")
        self.anchors_shape = anchors.get("shape")
        self.anchors_dtype = anchors.get("dtype")

        self.scale_fac1, self.scale_fac2, self.scale_fac3, self.scale_fac4 = \
            scale_factor1, scale_factor2, scale_factor3, scale_factor4
        self.cmp_val = cmp_value

        # boxpredictor shape nchw
        self.box_batch = self.box_shape[0]
        self.box_channel = self.box_shape[1]
        self.box_proposal = self.box_shape[2] * self.box_shape[3]

        self.anchors_batch = self.anchors_shape[0]
        self.anchors_channel = self.anchors_shape[1]
        self.anchors_proposal = self.anchors_shape[2] * self.anchors_shape[3]

        self._input_check(boxpredictor, anchors)
        self._attr_check(scale_factor1, scale_factor2, scale_factor3,
                         scale_factor4)
        self._output_check(out_box, anchors, cmp_value)

        self.kernel_name = kernel_name
        self.max_core_num = container.const_aicore_num

        self.box_gm = self.tinst.Tensor(self.box_dtype, (
            self.box_batch, self.box_channel, self.box_proposal),
                                        name="box_gm", scope=gm_scope)
        self.anchors_gm = self.tinst.Tensor(self.anchors_dtype, (
            self.anchors_batch, self.anchors_channel, self.anchors_proposal),
                                            name="anchors_gm", scope=gm_scope)
        self.out_box_gm = self.tinst.Tensor(self.out_dtype, (
            self.anchors_batch, self.anchors_channel, self.anchors_proposal),
                                            name="out_box_gm", scope=gm_scope)

        self.proposal = self._calc_per_proposal(self.anchors_dtype)
        self.proposal_num = int(np.ceil(self.anchors_proposal * 1.0 /
                                        self.proposal))
        self.loop_num = self.proposal_num * self.anchors_batch

    def _aicore_check(self, container, boxpredictor, anchors, out_box):
        if (isinstance(container, AContainer1910)):
            self.if_flor = False
            interface_check.check_param(boxpredictor, [4], ["float16"],
                                        ["NCHW"])
            interface_check.check_param(anchors, [4], ["float16"], ["NCHW"])
            interface_check.check_param(out_box, [4], ["float16"], ["NCHW"])
        elif (isinstance(container, AContainer1951)):
            self.if_flor = True
            interface_check.check_param(boxpredictor, [4], ["float16"],
                                        ["NCHW", "ND"])
            interface_check.check_param(anchors, [4], ["float16"],
                                        ["NCHW", "ND"])
            interface_check.check_param(out_box, [4], ["float32"],
                                        ["NCHW", "ND"])
        else:
            raise RuntimeError("unsupported type:{}".format(type(container)))

    def _input_check(self, boxpredictor, anchors):
        """
            check the input shape and boundary
            :param boxpredictor: input box shape
            :param anchors: input anchors shape
            :return
        """
        if boxpredictor.get("shape") != anchors.get("shape"):
            raise RuntimeError("boxpredictor shape must be equal"
                               "to anchors shape")
        if self.box_batch > 65535:
            raise RuntimeError(
                "The batch of boxpredictor should not exceed 65535.")
        if (boxpredictor.get("shape")[1] != 4) or \
                (anchors.get("shape")[1] != 4):
            raise RuntimeError("box_channel and anchors_channel"
                               "must be equal to 4")
        if reduce(lambda x, y: x * y, self.anchors_shape) > 2 ** 26:
            raise RuntimeError(
                "The total calculate number should not exceed 2**26.")
        if self.anchors_proposal < 16:
            raise RuntimeError("invalid decode shape, anchors_proposal"
                               "must be greater than 15")

    def _attr_check(self, scale_factor1, scale_factor2, scale_factor3,
                    scale_factor4):
        """
            check the attribute shape and dtype
            :param scale_factor1: divisor
            :param scale_factor2: divisor
            :param scale_factor3: divisor
            :param scale_factor4: divisor
            :return
        """
        if not isinstance(scale_factor1, float):
            raise TypeError(r"scale_factor1 in para type error %s"
                            % type(scale_factor1))
        if not isinstance(scale_factor2, float):
            raise TypeError(r"scale_factor2 in para type error %s"
                            % type(scale_factor2))
        if not isinstance(scale_factor3, float):
            raise TypeError(r"scale_factor3 in para type error %s"
                            % type(scale_factor3))
        if not isinstance(scale_factor4, float):
            raise TypeError(r"scale_factor4 in para type error %s"
                            % type(scale_factor4))
        if (scale_factor1 > 1000.0):
            raise RuntimeError("scale_factor1 must be less than"
                               "or equal to 1000.0")
        if (scale_factor2 > 1000.0):
            raise RuntimeError("scale_factor2 must be less than"
                               "or equal to 1000.0")
        if (scale_factor3 > 1000.0):
            raise RuntimeError("scale_factor3 must be less than"
                               "or equal to 1000.0")
        if (scale_factor4 > 1000.0):
            raise RuntimeError("scale_factor4 must be less than"
                               "or equal to 1000.0")
        if (scale_factor1 <= 0.0):
            raise RuntimeError("scale_factor1 must be greater than 0.0")
        if (scale_factor2 <= 0.0):
            raise RuntimeError("scale_factor2 must be greater than 0.0")
        if (scale_factor3 <= 0.0):
            raise RuntimeError("scale_factor3 must be greater than 0.0")
        if (scale_factor4 <= 0.0):
            raise RuntimeError("scale_factor4 must be greater than 0.0")

    def _output_check(self, out_box, anchors, cmp_value):
        """
            check the output shape
            :param out_box: out box shape
            :param anchors: input anchors shape
            :return
        """
        if (out_box.get("shape") != anchors.get("shape")):
            raise RuntimeError("out_box_shape must be equal to anchors_shape")
        if not isinstance(cmp_value, float):
            raise TypeError(r"cmp_value in para type error %s"
                            % type(cmp_value))
        if (cmp_value <= 0.0):
            raise RuntimeError("cmp_value must be greater than 0.0")
        if (cmp_value > 65504.0):
            raise RuntimeError("cmp_value must be less than"
                               "or equal to 65504.0")

    def _common_declare(self, dtype):
        ub_scope = self.container.tik.scope_ubuf
        self.xmin = self.tinst.Tensor(dtype, (self.proposal + 1,),
                                      name="xmin", scope=ub_scope)
        self.xmax = self.tinst.Tensor(dtype, (self.proposal + 1,),
                                      name="xmax", scope=ub_scope)
        self.ymin = self.tinst.Tensor(dtype, (self.proposal + 1,),
                                      name="ymin", scope=ub_scope)
        self.ymax = self.tinst.Tensor(dtype, (self.proposal + 1,),
                                      name="ymax", scope=ub_scope)

        self.tensor_y = self.tinst.Tensor(dtype, (self.proposal,),
                                          name="tensor_y", scope=ub_scope)
        self.tensor_x = self.tinst.Tensor(dtype, (self.proposal,),
                                          name="tensor_x", scope=ub_scope)
        self.tensor_h = self.tinst.Tensor(dtype, (self.proposal + 1,),
                                          name="tensor_h", scope=ub_scope)
        self.tensor_w = self.tinst.Tensor(dtype, (self.proposal + 1,),
                                          name="tensor_w", scope=ub_scope)

        self.xcenter_ub = self.tinst.Tensor("float32", (self.proposal + 1,),
                                            name="xcenter_ub", scope=ub_scope)
        self.ycenter_ub = self.tinst.Tensor("float32", (self.proposal + 1,),
                                            name="ycenter_ub", scope=ub_scope)
        self.xmin_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                           name="xmin_fp32", scope=ub_scope)
        self.ymin_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                           name="ymin_fp32", scope=ub_scope)
        self.xmax_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                           name="xmax_fp32", scope=ub_scope)
        self.ymax_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                           name="ymax_fp32", scope=ub_scope)

        self.ty_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                         name="ty_fp32", scope=ub_scope)
        self.tx_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                         name="tx_fp32", scope=ub_scope)
        self.th_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                         name="th_fp32", scope=ub_scope)
        self.tw_fp32 = self.tinst.Tensor("float32", (self.proposal + 1,),
                                         name="tw_fp32", scope=ub_scope)
        self.x_one = self.tinst.Tensor("float32", (self.proposal + 1,),
                                       name="x_one", scope=ub_scope)
        self.x_two = self.tinst.Tensor("float32", (self.proposal + 1,),
                                       name="x_two", scope=ub_scope)
        self.th_exp = self.tinst.Tensor("float32", (self.proposal + 1,),
                                        name="th_exp", scope=ub_scope)
        self.tw_exp = self.tinst.Tensor("float32", (self.proposal + 1,),
                                        name="tw_exp", scope=ub_scope)

        self.num_div = self.tinst.Scalar("float32")
        self.num_div.set_as(0.5)

    def _calc_per_proposal(self, dtype="float16"):
        factor = 4 if (dtype == "float32") else 2
        proposal = min((self.container.const_ub_max_byte - TENSOR_NUM * factor)
                       // (TENSOR_NUM * factor), self.box_proposal)

        if (proposal % 16 != 0):
            proposal = proposal // 16 * 16

        return proposal

    def model_compute(self):
        self._calc_xcore_info(self.loop_num, 0, self.max_core_num)
        self.start = self.tinst.Scalar("int32", name="core_start")
        self.end = self.tinst.Scalar("int32", name="core_end")
        with self.tinst.for_range(0, self.xcore_num,
                                  block_num=self.xcore_num) as index:
            self._common_declare(self.anchors_dtype)
            with self.tinst.if_scope(index != self.xcore_num - 1):
                self.start.set_as(self.xlen_each_core * index)
                self.end.set_as(self.xlen_each_core * (index + 1))
                self._model_compute_each_core(self.start, self.end)
            with self.tinst.else_scope():
                self.start.set_as(self.xlen_each_core * index)
                self.end.set_as(self.loop_num)
                self._model_compute_each_core(self.start, self.end)
        self.tinst.BuildCCE(kernel_name=self.kernel_name,
                            inputs=[self.box_gm, self.anchors_gm],
                            outputs=[self.out_box_gm], enable_l2=True)

    def _model_compute_each_core(self, start, end):
        with self.tinst.for_range(start, end) as index:
            box_col = self.proposal
            if (self.box_proposal % self.proposal != 0):
                # processing tail data
                with self.tinst.if_scope((index + 1) % self.proposal_num == 0):
                    box_col = (self.box_proposal % self.proposal + 16 - 1) \
                              // 16 * 16
                    self._model_compute_tail_loop(index, box_col)
                with self.tinst.else_scope():
                    box_col = self.proposal
                    self._move_data_to_ub(index, 0, box_col)
                    self._compute_ceil(box_col)
                    self._move_data_to_gm(index, 0, box_col)
            else:
                self._move_data_to_ub(index, 0, box_col)
                self._compute_ceil(box_col)
                self._move_data_to_gm(index, 0, box_col)

    def _model_compute_tail_loop(self, index, box_col):
        # the number of non-16 alignments that need to be moved forward
        if ((self.box_proposal % self.proposal % 16) != 0):
            box_left = 16 - self.box_proposal % self.proposal % 16
        else:
            box_left = 0
        self._move_data_to_ub(index, box_left, box_col)
        self._compute_ceil(box_col)
        self._move_data_to_gm(index, box_left, box_col)

    def _move_data_to_ub(self, index, value_left, box_col):
        """
            move data ub
            :param index: circular index
            :param left_value: whether the number of forward offsets is
                required
            :box_col: the total amount of data to be moved
            :return
        """
        self.tinst.data_move(self.ymin, self.anchors_gm[
            index // self.proposal_num, 0, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)
        self.tinst.data_move(self.xmin, self.anchors_gm[
            index // self.proposal_num, 1, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)
        self.tinst.data_move(self.ymax, self.anchors_gm[
            index // self.proposal_num, 2, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)
        self.tinst.data_move(self.xmax, self.anchors_gm[
            index // self.proposal_num, 3, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)
        self.tinst.data_move(self.tensor_y, self.box_gm[
            index // self.proposal_num, 0, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)
        self.tinst.data_move(self.tensor_x, self.box_gm[
            index // self.proposal_num, 1, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)
        self.tinst.data_move(self.tensor_h, self.box_gm[
            index // self.proposal_num, 2, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)
        self.tinst.data_move(self.tensor_w, self.box_gm[
            index // self.proposal_num, 3, index % self.proposal_num *
            self.proposal - value_left], 0, 1, box_col // 16, 0, 0)

    def _compute_ceil(self, box_col):
        """
            decode and calculate
            :param box_col: the total amount of data to be calculated
            :return
        """
        align_bnum = self.container.calc_blk_align_num(self.anchors_dtype,
                                                       box_col)
        num_per_cmd = self.container.get_vec_proc_num_per_cmd("float32")

        bufs = {
            "xmin": AVecBuf(self.xmin, align_bnum, 0, self.container, True,
                            num_per_cmd),
            "xmin_fp32": AVecBuf(self.xmin_fp32, align_bnum, 0,
                                 self.container, True, num_per_cmd),
            "ymin": AVecBuf(self.ymin, align_bnum, 0, self.container, True,
                            num_per_cmd),
            "ymin_fp32": AVecBuf(self.ymin_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "xmax": AVecBuf(self.xmax, align_bnum, 0, self.container, True,
                            num_per_cmd),
            "xmax_fp32": AVecBuf(self.xmax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "ymax": AVecBuf(self.ymax, align_bnum, 0, self.container, True,
                            num_per_cmd),
            "ymax_fp32": AVecBuf(self.ymax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "tensor_x": AVecBuf(self.tensor_x, align_bnum, 0, self.container,
                                True, num_per_cmd),
            "tx_fp32": AVecBuf(self.tx_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tensor_y": AVecBuf(self.tensor_y, align_bnum, 0,
                                self.container, True, num_per_cmd),
            "ty_fp32": AVecBuf(self.ty_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tensor_h": AVecBuf(self.tensor_h, align_bnum, 0,
                                self.container, True, num_per_cmd),
            "th_fp32": AVecBuf(self.th_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tensor_w": AVecBuf(self.tensor_w, align_bnum, 0,
                                self.container, True, num_per_cmd),
            "tw_fp32": AVecBuf(self.tw_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd)}

        cmds = [VecGCmd("vconv", "xmin_fp32", "xmin", round_mode="none"),
                VecGCmd("vconv", "ymin_fp32", "ymin", round_mode="none"),
                VecGCmd("vconv", "xmax_fp32", "xmax", round_mode="none"),
                VecGCmd("vconv", "ymax_fp32", "ymax", round_mode="none"),
                VecGCmd("vconv", "tx_fp32", "tensor_x", round_mode="none"),
                VecGCmd("vconv", "ty_fp32", "tensor_y", round_mode="none"),
                VecGCmd("vconv", "th_fp32", "tensor_h", round_mode="none"),
                VecGCmd("vconv", "tw_fp32", "tensor_w", round_mode="none")]

        VecGExecutor.exec_vec_g_cmd(self.container, bufs, cmds, "xmin")
        self._compute_anchor(align_bnum, num_per_cmd)
        self._compute_second(align_bnum, num_per_cmd)
        self._compute_three(align_bnum, num_per_cmd)

    def _compute_anchor(self, align_bnum, num_per_cmd):
        bufs = {
            "xmax_fp32": AVecBuf(self.xmax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "xmin_fp32": AVecBuf(self.xmin_fp32, align_bnum, 0,
                                 self.container, True, num_per_cmd),
            "ymax_fp32": AVecBuf(self.ymax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "ymin_fp32": AVecBuf(self.ymin_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "ycenter_ub": AVecBuf(self.ycenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd),
            "xcenter_ub": AVecBuf(self.xcenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd),
            "ty_fp32": AVecBuf(self.ty_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tx_fp32": AVecBuf(self.tx_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "th_fp32": AVecBuf(self.th_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tw_fp32": AVecBuf(self.tw_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd)}

        cmds = [VecGCmd("vsub", "xmax_fp32", "xmax_fp32",
                        src1_name="xmin_fp32"),
                VecGCmd("vsub", "ymax_fp32", "ymax_fp32",
                        src1_name="ymin_fp32"),
                VecGCmd("vmuls", "ycenter_ub", "ymax_fp32",
                        scalar=self.num_div),
                VecGCmd("vmuls", "xcenter_ub", "xmax_fp32",
                        scalar=self.num_div),
                VecGCmd("vadd", "ycenter_ub", "ycenter_ub",
                        src1_name="ymin_fp32"),
                VecGCmd("vadd", "xcenter_ub", "xcenter_ub",
                        src1_name="xmin_fp32"),
                VecGCmd("vmuls", "ty_fp32", "ty_fp32",
                        scalar=1.0 / self.scale_fac1),
                VecGCmd("vmuls", "tx_fp32", "tx_fp32",
                        scalar=1.0 / self.scale_fac2),
                VecGCmd("vmuls", "th_fp32", "th_fp32",
                        scalar=1.0 / self.scale_fac3),
                VecGCmd("vmuls", "tw_fp32", "tw_fp32",
                        scalar=1.0 / self.scale_fac4)]

        VecGExecutor.exec_vec_g_cmd(self.container, bufs, cmds, "xmax_fp32")

    def _compute_second(self, align_bnum, num_per_cmd):
        # calculate th_fp32 and tw_fp32
        bufs = {
            "th_fp32": AVecBuf(self.th_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "th_exp": AVecBuf(self.th_exp, align_bnum, 0,
                              self.container, True, num_per_cmd),
            "xmax_fp32": AVecBuf(self.xmax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "tw_fp32": AVecBuf(self.tw_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tw_exp": AVecBuf(self.tw_exp, align_bnum, 0,
                              self.container, True, num_per_cmd),
            "ymax_fp32": AVecBuf(self.ymax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "ycenter_ub": AVecBuf(self.ycenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd),
            "xcenter_ub": AVecBuf(self.xcenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd),
            "ty_fp32": AVecBuf(self.ty_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tx_fp32": AVecBuf(self.tx_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd)}

        if not self.if_flor:
            self._vexp_taylor_series(align_bnum, num_per_cmd, "th_exp",
                                     self.th_exp, "th_fp32", self.th_fp32)
            self._vexp_taylor_series(align_bnum, num_per_cmd, "tw_exp",
                                     self.tw_exp, "tw_fp32", self.tw_fp32)
        else:
            flor_cmds = [VecGCmd("vexp", "th_exp", "th_fp32"),
                         VecGCmd("vexp", "tw_exp", "tw_fp32"),
                         VecGCmd("vmins", "th_exp", "th_exp",
                                 scalar=self.cmp_val),
                         VecGCmd("vmins", "tw_exp", "tw_exp",
                                 scalar=self.cmp_val)]
            VecGExecutor.exec_vec_g_cmd(self.container, bufs, flor_cmds,
                                        "th_fp32")

        cmds = [
            VecGCmd("vmul", "th_fp32", "th_exp", src1_name="ymax_fp32"),
            VecGCmd("vmul", "tw_fp32", "tw_exp", src1_name="xmax_fp32"),
            VecGCmd("vmul", "ty_fp32", "ty_fp32", src1_name="ymax_fp32"),
            VecGCmd("vmul", "tx_fp32", "tx_fp32", src1_name="xmax_fp32"),
            VecGCmd("vadd", "ycenter_ub", "ty_fp32",
                    src1_name="ycenter_ub"),
            VecGCmd("vadd", "xcenter_ub", "tx_fp32",
                    src1_name="xcenter_ub")]

        VecGExecutor.exec_vec_g_cmd(self.container, bufs, cmds, "th_fp32")

    def _compute_three(self, align_bnum, num_per_cmd):
        bufs = {
            "th_fp32": AVecBuf(self.th_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "tw_fp32": AVecBuf(self.tw_fp32, align_bnum, 0,
                               self.container, True, num_per_cmd),
            "xmin_fp32": AVecBuf(self.xmin_fp32, align_bnum, 0,
                                 self.container, True, num_per_cmd),
            "xcenter_ub": AVecBuf(self.xcenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd),
            "ycenter_ub": AVecBuf(self.ycenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd),
            "ymin_fp32": AVecBuf(self.ymin_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "xmax_fp32": AVecBuf(self.xmax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "ymax_fp32": AVecBuf(self.ymax_fp32, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            "xmin": AVecBuf(self.xmin, align_bnum, 0, self.container, True,
                            num_per_cmd),
            "ymin": AVecBuf(self.ymin, align_bnum, 0, self.container, True,
                            num_per_cmd),
            "xmax": AVecBuf(self.xmax, align_bnum, 0, self.container, True,
                            num_per_cmd),
            "ymax": AVecBuf(self.ymax, align_bnum, 0, self.container, True,
                            num_per_cmd)}

        cmds = [VecGCmd("vmuls", "th_fp32", "th_fp32", scalar=self.num_div),
                VecGCmd("vmuls", "tw_fp32", "tw_fp32", scalar=self.num_div),
                VecGCmd("vsub", "ymin_fp32", "ycenter_ub",
                        src1_name="th_fp32"),
                VecGCmd("vsub", "xmin_fp32", "xcenter_ub",
                        src1_name="tw_fp32"),
                VecGCmd("vadd", "ymax_fp32", "ycenter_ub",
                        src1_name="th_fp32"),
                VecGCmd("vadd", "xmax_fp32", "xcenter_ub",
                        src1_name="tw_fp32"),
                ]

        VecGExecutor.exec_vec_g_cmd(self.container, bufs, cmds, "th_fp32")

        if not self.if_flor:
            flo_cmds = [
                VecGCmd("vconv", "xmin", "xmin_fp32", round_mode="none"),
                VecGCmd("vconv", "ymin", "ymin_fp32", round_mode="none"),
                VecGCmd("vconv", "xmax", "xmax_fp32", round_mode="none"),
                VecGCmd("vconv", "ymax", "ymax_fp32", round_mode="none")]
            VecGExecutor.exec_vec_g_cmd(self.container, bufs,
                                        flo_cmds, "th_fp32")

    def _vexp_taylor_series(self, align_bnum, num_per_cmd, dst_n, dst_buf,
                            src_n, src_buf):
        """
            the exponential operation is expanded by the sixth order taylor
                series
            :param num: the total amount of data to be calculated
            :param dst_buf: dst operands
            :param src_buf: src operands
            :return
        """
        bufs = {
            "xmax_fp32": AVecBuf(self.x_one, align_bnum, 0, self.container,
                                 True, num_per_cmd),
            src_n: AVecBuf(src_buf, align_bnum, 0,
                           self.container, True, num_per_cmd),
            "x_one": AVecBuf(self.x_one, align_bnum, 0, self.container,
                             True, num_per_cmd),
            "x_two": AVecBuf(self.x_two, align_bnum, 0, self.container,
                             True, num_per_cmd),
            dst_n: AVecBuf(dst_buf, align_bnum, 0, self.container,
                           True, num_per_cmd),
            "ycenter_ub": AVecBuf(self.ycenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd),
            "xcenter_ub": AVecBuf(self.xcenter_ub, align_bnum, 0,
                                  self.container, True, num_per_cmd)}

        cmds = [VecGCmd("vmul", "x_one", src_n, src1_name=src_n),
                VecGCmd("vmuls", "x_two", "x_one", scalar=1.0 / 2.0),
                VecGCmd("vadds", dst_n, src_n, scalar=1.0),
                VecGCmd("vadd", dst_n, "x_two", src1_name=dst_n),

                VecGCmd("vmul", "x_one", "x_one", src1_name=src_n),
                VecGCmd("vmuls", "x_two", "x_one", scalar=1.0 / 6.0),
                VecGCmd("vadd", dst_n, "x_two", src1_name=dst_n),

                VecGCmd("vmul", "x_one", "x_one", src1_name=src_n),
                VecGCmd("vmuls", "x_two", "x_one", scalar=1.0 / 24.0),
                VecGCmd("vadd", dst_n, "x_two", src1_name=dst_n),

                VecGCmd("vmul", "x_one", "x_one", src1_name=src_n),
                VecGCmd("vmuls", "x_two", "x_one", scalar=1.0 / 120.0),
                VecGCmd("vadd", dst_n, "x_two", src1_name=dst_n),

                VecGCmd("vmul", "x_one", "x_one", src1_name=src_n),
                VecGCmd("vmuls", "x_two", "x_one", scalar=1.0 / 720.0),
                VecGCmd("vadd", dst_n, "x_two", src1_name=dst_n)]

        VecGExecutor.exec_vec_g_cmd(self.container, bufs, cmds, "x_one")

    def _move_data_to_gm(self, index, left_value, box_col):
        """
            move data outside
            :param index: circular index
            :param left_value: whether the number of forward offsets is
                required
            :param box_col: the total amount of data to be moved
            :return
        """
        if self.if_flor:
            ymin, xmin, ymax, xmax = self.ymin_fp32, self.xmin_fp32, \
                                     self.ymax_fp32, self.xmax_fp32
            blk_num = 8
        else:
            ymin, xmin, ymax, xmax = self.ymin, self.xmin, \
                                     self.ymax, self.xmax
            blk_num = 16
        self.tinst.data_move(self.out_box_gm[
                                 index // self.proposal_num, 0, index %
                                 self.proposal_num * self.proposal -
                                 left_value], ymin, 0, 1, box_col // blk_num,
                             0, 0)
        self.tinst.data_move(self.out_box_gm[
                                 index // self.proposal_num, 1, index %
                                 self.proposal_num * self.proposal -
                                 left_value], xmin, 0, 1, box_col // blk_num,
                             0, 0)
        self.tinst.data_move(self.out_box_gm[
                                 index // self.proposal_num, 2, index %
                                 self.proposal_num * self.proposal -
                                 left_value], ymax, 0, 1, box_col // blk_num,
                             0, 0)
        self.tinst.data_move(self.out_box_gm[
                                 index // self.proposal_num, 3, index %
                                 self.proposal_num * self.proposal -
                                 left_value], xmax, 0, 1, box_col // blk_num,
                             0, 0)


def decode(boxpredictor, anchors, out_box, scale_factor1, scale_factor2,
           scale_factor3, scale_factor4, cmp_value=65504.0,
           kernel_name="decode", test=False):
    interface_check.check_kernelname(kernel_name)
    container = get_aicore_container(("Ascend610",),
                                     c3x_support_list=("Ascend310",))
    core_type = CoreType.AICORE

    obj = Decode(container, core_type, boxpredictor, anchors, out_box,
                 scale_factor1, scale_factor2, scale_factor3, scale_factor4,
                 cmp_value, kernel_name)

    obj.model_compute()

    if not test:
        return 0
    else:
        return obj.tinst
