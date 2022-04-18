# -*- coding: utf-8 -*-
from functools import reduce
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ascend import Tiler
from ascend import AContainer1910
from ascend import AContainer1951
from ascend import CoreType
from uti import interface_check
from uti import check
from version.get_version import get_aicore_container

G_THREAD = 2


class ClipToWindow(Tiler):
    """
    Introduction
    ------------
        Intialize ClipToWindow parameter
    Parameters
    ----------
        @container, IN: the container of 1910 or 1951
        @core_type, IN: the core type of 1910 or 1951
        @max_core_num, IN: select the maximum number of core
        @box, IN: input box shape, NCHW, ND
        @out_box, IN: output box shape, NCHW, ND
        @clip_y_min, IN: minimum value of Y direction, float
        @clip_x_min, IN: minimun value of X direction, float
        @clip_y_max, IN: maximum value of Y direction, float
        @clip_x_max, IN: maximum value of X direction, float
        @kernel_name, IN: ClipToWindow kernel name
    Returns
    -------
    """

    def __init__(self, container, core_type, max_core_num, box, out_box,
                 clip_y_min, clip_x_min, clip_y_max, clip_x_max, kernel_name):
        check.turn_off_tik_debug()
        super(ClipToWindow, self).__init__(container, core_type)
        self._aicore_check(container, box, out_box)
        self.container = container
        tik = self.container.tik
        self.tinst = self.container.tinst
        check.check_param_type(max_core_num, int, "err max_core_num")
        check.check_param_low(max_core_num, 1, "invalid max_core_num")

        self.box_shape = box.get("shape")
        self.box_dtype = box.get("dtype")
        self.out_box_shape = out_box.get("shape")
        self.out_box_dtype = out_box.get("dtype")
        self.clip_y_min = clip_y_min
        self.clip_x_min = clip_x_min
        self.clip_y_max = clip_y_max
        self.clip_x_max = clip_x_max
        self.box_batch = self.box_shape[0]
        self.box_channel = self.box_shape[1]
        self.box_proposal = self.box_shape[2] * self.box_shape[3]

        self._check_input()
        self._check_output()
        self._attr_check(clip_y_min, clip_x_min, clip_y_max, clip_x_max)

        self.kernel_name = kernel_name
        self.max_core_num = max_core_num

        # define input & output
        self.box_gm = self.tinst.Tensor(self.box_dtype, (self.box_batch,
            self.box_channel, self.box_proposal),
            name="box_gm", scope=tik.scope_gm)
        self.out_box_gm = self.tinst.Tensor(self.out_box_dtype,
            (self.box_batch, self.box_channel, self.box_proposal),
            name="out_box_gm", scope=tik.scope_gm)
        self.max_box_num = self._calc_max_box_num(self.box_dtype)

    def _aicore_check(self, container, box, out_box):
        if (isinstance(container, AContainer1910)):
            interface_check.check_param(box, [4], ["float16"], ["NCHW"])
            interface_check.check_param(out_box, [4], ["float16"], ["NCHW"])
        elif (isinstance(container, AContainer1951)):
            interface_check.check_param(box, [4], ["float16"], ["NCHW", "ND"])
            interface_check.check_param(out_box, [4], ["float16"],
                                        ["NCHW", "ND"])
        else:
            raise RuntimeError("unsupported type:{}".format(type(container)))

    def _check_input(self):
        if self.box_batch > 65535:
            raise RuntimeError("box_batch must be less than or equal to 65535")
        if self.box_channel != 4:
            raise RuntimeError("box_channel must be equal to 4")
        if reduce(lambda x, y: x * y, self.box_shape) > 2 ** 26:
            raise RuntimeError(
                "The total calculate number should not exceed 2**26.")

    def _attr_check(self, clip_y_min, clip_x_min, clip_y_max, clip_x_max):
        if not isinstance(clip_y_min, float):
            raise TypeError(r"clip_y_min in para type error %s"
                            % type(clip_y_min))
        if not isinstance(clip_x_min, float):
            raise TypeError(r"clip_x_min in para type error %s"
                            % type(clip_x_min))
        if not isinstance(clip_y_max, float):
            raise TypeError(r"clip_y_max in para type error %s"
                            % type(clip_y_max))
        if not isinstance(clip_x_max, float):
            raise TypeError(r"clip_x_max in para type error %s"
                            % type(clip_x_max))

        if clip_y_min >= clip_y_max:
            raise RuntimeError("clip_y_min must be less than clip_y_max")
        if clip_x_min >= clip_x_max:
            raise RuntimeError("clip_x_min must be less than clip_x_max")

        if clip_y_min < 0.0:
            raise RuntimeError("clip_y_min must be greater than or"
                               "equal to 0.0")
        if clip_x_min < 0.0:
            raise RuntimeError("clip_x_min must be greater than or"
                               "equal to 0.0")
        if clip_y_max > 10000.0:
            raise RuntimeError("clip_y_max must be less than or"
                               "equal to 10000.0")
        if clip_x_max > 10000.0:
            raise RuntimeError("clip_x_max must be less than or"
                               "equal to 10000.0")

    def _check_output(self):
        if self.box_shape != self.out_box_shape:
            raise RuntimeError("box_shape must be equal to out_box_shape")

    def _calc_max_box_num(self, dtype):
        elm_byte = self.container.const_dtype_byte.get(dtype)
        max_ub_byte = self.container.const_ub_max_byte
        align_box_proposal = self.container.calc_blk_align_num(dtype,
            self.box_proposal)
        if (elm_byte is not None):
            box_num = min((max_ub_byte) // (8 * elm_byte * G_THREAD),
                          align_box_proposal)
            box_num = box_num // 16 * 16
            return box_num
        else:
            raise ValueError("invalid dtype {}".format(dtype))

    def compute(self):
        # align to multiple of 16
        align_num = self.container.calc_blk_align_num(self.box_dtype, 1)

        # compute open core
        self._calc_xcore_info(self.box_proposal, align_num, self.max_core_num)

        # turn on multi-core and multi thread data partitioning
        with self.tinst.for_range(0, self.box_batch) as batch_i:
            self._batch_idx = batch_i
            self._compute_multi_core(G_THREAD, self.max_box_num,
                                     self.max_core_num)
        self.tinst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.box_gm],
                            outputs=[self.out_box_gm], enable_l2=True)

    def _compute_one_loop(self, loop_begin, loop_end):
        batch_i = self._batch_idx
        tik = self.container.tik
        left = self.tinst.Scalar(dtype="int32", name="left")
        box_bnum = loop_end - loop_begin

        # handle the case that the last core data is not 16 times
        left.set_as(0)
        with self.tinst.if_scope(loop_begin != 0):
            with self.tinst.if_scope(box_bnum % 16 != 0):
                left.set_as(16 - box_bnum % 16)
            with self.tinst.else_scope():
                pass
        with self.tinst.else_scope():
            pass

        y_min = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                  name="y_min", scope=tik.scope_ubuf)
        x_min = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                  name="x_min", scope=tik.scope_ubuf)
        y_max = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                  name="y_max", scope=tik.scope_ubuf)
        x_max = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                  name="x_max", scope=tik.scope_ubuf)

        win_y_min = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                      name="win_y_min", scope=tik.scope_ubuf)
        win_x_min = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                      name="win_x_min", scope=tik.scope_ubuf)
        win_y_max = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                      name="win_y_max", scope=tik.scope_ubuf)
        win_x_max = self.tinst.Tensor(self.box_dtype, (self.max_box_num, ),
                                      name="win_x_max", scope=tik.scope_ubuf)

        align_bnum = self.container.calc_blk_align_num(self.box_dtype,
                                                       box_bnum)
        burst = self.container.calc_block_num(self.box_dtype, align_bnum)

        boxes = [y_min, x_min, y_max, x_max]
        for i in range(0, 4):
            self.tinst.data_move(boxes[i], self.box_gm[batch_i, i,
                loop_begin - left], 0, 1, burst, 0, 0)

        self._compute_ceil(align_bnum, y_min, x_min, y_max, x_max, win_y_min,
                           win_x_min, win_y_max, win_x_max)

        for i in range(0, 4):
            self.tinst.data_move(self.out_box_gm[batch_i, i, loop_begin -
                left], boxes[i], 0, 1, burst, 0, 0)

    def _compute_ceil(self, align_bnum, y_min, x_min, y_max, x_max, win_y_min,
                      win_x_min, win_y_max, win_x_max):
        # the amount of data that can be processed at one time is dtype
        num_per_cmd = self.container.get_vec_proc_num_per_cmd(self.box_dtype)

        # bufs to be used
        bufs = {"y_min": AVecBuf(y_min, align_bnum, 0, self.container,
                                 True, num_per_cmd),
                "x_min": AVecBuf(x_min, align_bnum, 0, self.container,
                                 True, num_per_cmd),
                "y_max": AVecBuf(y_max, align_bnum, 0, self.container,
                                 True, num_per_cmd),
                "x_max": AVecBuf(x_max, align_bnum, 0, self.container,
                                 True, num_per_cmd),
                "win_y_min": AVecBuf(win_y_min, align_bnum, 0, self.container,
                                     True, num_per_cmd),
                "win_x_min": AVecBuf(win_x_min, align_bnum, 0, self.container,
                                     True, num_per_cmd),
                "win_y_max": AVecBuf(win_y_max, align_bnum, 0, self.container,
                                     True, num_per_cmd),
                "win_x_max": AVecBuf(win_x_max, align_bnum, 0, self.container,
                                     True, num_per_cmd)}

        # instructions to be used
        cmds = [VecGCmd("vector_dup", "win_y_min", scalar=self.clip_y_min),
                VecGCmd("vector_dup", "win_x_min", scalar=self.clip_x_min),
                VecGCmd("vector_dup", "win_y_max", scalar=self.clip_y_max),
                VecGCmd("vector_dup", "win_x_max", scalar=self.clip_x_max),
                VecGCmd("vmin", "y_min", "y_min", src1_name="win_y_max"),
                VecGCmd("vmax", "y_min", "y_min", src1_name="win_y_min"),
                VecGCmd("vmin", "y_max", "y_max", src1_name="win_y_max"),
                VecGCmd("vmax", "y_max", "y_max", src1_name="win_y_min"),
                VecGCmd("vmin", "x_min", "x_min", src1_name="win_x_max"),
                VecGCmd("vmax", "x_min", "x_min", src1_name="win_x_min"),
                VecGCmd("vmin", "x_max", "x_max", src1_name="win_x_max"),
                VecGCmd("vmax", "x_max", "x_max", src1_name="win_x_min")]

        # execute instruction in cmds
        VecGExecutor.exec_vec_g_cmd(self.container, bufs, cmds, "y_min")


def clip_to_window(box, out_box, clip_y_min, clip_x_min, clip_y_max,
                   clip_x_max, kernel_name="clip_to_window", test=False):
    interface_check.check_kernelname(kernel_name)
    container = get_aicore_container(("Ascend610",),
                                     c3x_support_list=("Ascend310",))
    core_type = CoreType.AICORE
    max_core_num = 1024

    cpp_obj = ClipToWindow(container, core_type, max_core_num, box, out_box,
                           clip_y_min, clip_x_min, clip_y_max, clip_x_max,
                           kernel_name)

    cpp_obj.compute()

    if not test:
        return 0
    else:
        return cpp_obj.tinst
