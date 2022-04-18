# -*- coding:utf-8 -*-
import importlib
import numpy as np

from ascend import AContainer
from ascend import AContainer1951
from ascend import CoreType
from ascend import SoftArch
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ascend import Tiler
from uti import check
from uti import interface_check
from point_cloud.point_dis_matrix import DisMatrix
from version.get_version import get_aicore_container
from ._mask_idx import MaskIdx
from ._mask_idx import dup_ub_buf


class UBStuffs(object):
    def __init__(self, qb):
        ub_scope = qb.const_container.tik.scope_ubuf
        self.dis_m = DisMatrix(qb.const_container, qb.dis_row, qb.dis_col,
                               qb.dis_dtype, b_fix_row=False, b_gm_fp16=True,
                               b_debug=qb.debug)
        self.mask_idx = MaskIdx(qb.const_container, idx_type=qb.idx_type,
                                debug=qb.debug)
        self.mask_res_ub = qb.tinst.Tensor("uint8", (qb.dis_col//8, ),
                                           name="mask_res_ub", scope=ub_scope)
        # + 1 for tik compile reason
        self.tmp_col_dis = qb.tinst.Tensor(qb.dis_dtype, (qb.dis_col+1, ),
                                           name="tmp_col_dis", scope=ub_scope)
        assert(qb.dis_dtype == "float32" and qb.dis_cmp_dtype == "float32")
        self.dis_to_compare = self.tmp_col_dis
        # + 8 to avoid idx out of bounds
        self.tmp_out_idx = qb.tinst.Tensor(qb.idx_type,
                                           (qb.dis_row, qb.nsample + 8),
                                           name="tmp_out_idx", scope=ub_scope)
        self.tmp_out_idx_cnt = qb.tinst.Tensor("int32", (qb.dis_row, ),
                                               name="tmp_out_idx_cnt",
                                               scope=ub_scope)
        self.radius_ub = qb.tinst.Tensor(qb.dis_cmp_dtype, (qb.dis_pnum, ),
                                         name="radius_ub", scope=ub_scope)
        qb.tinst.vector_dup(qb.dis_pnum, self.radius_ub, qb.radius, 1, 1, 0)


class QueryBallPoint(Tiler):
    def __init__(self, container, core_type, batch, ori_num, samp_num,
                 radius, nsample, kernel_name, debug=False):
        self._check_param(batch, ori_num, samp_num, radius, nsample)
        self._init_vars(batch, ori_num, samp_num, radius, nsample,
                        kernel_name, debug)
        super(QueryBallPoint, self).__init__(container, core_type)
        self.tinst = self.const_container.tinst
        self.max_core_n = 1024
        self.debug = debug
        check.turn_off_tik_debug()

        self.gm_in_dtype = "float16"
        self.gm_align_n = container.calc_blk_align_num(self.gm_in_dtype, 1)
        self.dis_dtype = "float32"
        self.dis_cmp_dtype = "float32"
        self.dis_pnum = container.get_vec_proc_num_per_cmd(self.dis_cmp_dtype)
        self.dis_row, self.dis_col = self._calc_dis_mat_row_col()
        self.idx_type = "int32"

        gm_scope = self.const_container.tik.scope_gm
        self.ori_xyz = self.tinst.Tensor(self.gm_in_dtype,
                                         shape=(batch, 3, 1, ori_num),
                                         name="ori_xyz", scope=gm_scope)
        self.samp_xyz = self.tinst.Tensor(self.gm_in_dtype,
                                          shape=(batch, 3, 1, samp_num),
                                          name="samp_xyz", scope=gm_scope)
        self.out_idx = self.tinst.Tensor("int32",
                                         shape=(batch, 1, samp_num, nsample),
                                         name="out_idx", scope=gm_scope)

    def _check_param(self, batch, ori_num, samp_num, radius, nsample):
        check.check_param_type(batch, int,
                               "batch type {} is not int".format(type(batch)))
        err_str = "batch {} out of supported range [1~8]".format(batch)
        check.check_param_range(batch, 1, 8, err_str)

        check.check_param_type(
            ori_num, int, "ori_num type {} not int".format(type(ori_num)))
        err_str = "ori_num {} out of supported range [1~64000)".format(ori_num)
        check.check_param_range(ori_num, 1, 64000 - 1, err_str)

        check.check_param_type(
            samp_num, int, "samp_num {} not int".format(type(samp_num)))
        err_str = "samp_num {} out of support range [1~64000)".format(samp_num)
        check.check_param_range(samp_num, 1, 64000 - 1, err_str)

        check.check_param_type(
            radius, float, "radius type {} not float".format(type(radius)))
        err_str = "radius {} out of range [0.01, 45.0)".format(radius)
        check.check_param_range(radius, 0.01, 45.0, err_str)
        check.check_param_not_equal(radius, 45.0, err_str)

        check.check_param_type(
            nsample, int, "nsample type {} not int".format(type(nsample)))
        err_str = "nsample {} out of supported range [8, 1024]".format(nsample)
        check.check_param_range(nsample, 8, 1024, err_str)
        check.check_param_mod(
            nsample, 8, "nsample {} must be % 8 == 0".format(nsample))

        err_str = "batch*samp_num {} * {} = {} out of supported range \
                    [1, 64000)".format(batch, samp_num, batch*samp_num)
        check.check_param_range(batch*samp_num, 1, 64000 - 1, err_str)

        err_str = "batch*ori_num {} * {} = {} out of supported range \
                    [1, 64000)".format(batch, ori_num, batch*ori_num)
        check.check_param_range(batch*ori_num, 1, 64000 - 1, err_str)

        if (ori_num < samp_num):
            raise RuntimeError("samp_num {} should be \
                               <= ori_num {}".format(samp_num, ori_num))

    def _init_vars(self, batch, ori_num, samp_num, radius, nsample,
                   kernel_name, debug):
        self.batch = batch
        self.ori_num = ori_num
        self.samp_num = samp_num
        self._in_ori_radius = radius
        # no need to sqrt(dis). self.radius is fp16, so radius >= 0.01 and < 45
        self.radius = radius * radius
        self.nsample = nsample
        self.kernel_name = kernel_name
        self.debug = debug

    def _calc_dis_mat_row_col(self):
        factor = 1 if (self.dis_dtype == "float32") else 2
        max_row = 32*1024 // self.nsample
        max_col = 1024
        core_num = self.const_container.get_core_num(self.const_core_type,
                                                     self.max_core_n)
        col = min(max_col * factor, self.ori_num)
        row = min(self.samp_num // core_num, min(512, max_row))
        # col to cmp dis in dis_cmp_dtype
        if (col % self.dis_pnum != 0):
            col = self.dis_pnum * (col // self.dis_pnum) + self.dis_pnum
        # row to move gm_data. need % 16 == 0
        if (row % 16 != 0 or row == 0):
            row = max(16 * (row // 16), 16)
        return row, col

    def compute(self):
        self._calc_xcore_info(self.samp_num, self.gm_align_n, self.max_core_n)
        self._compute_multi_core(1, self.dis_row, self.max_core_n)
        self.tinst.BuildCCE(kernel_name=self.kernel_name,
                            inputs=[self.ori_xyz, self.samp_xyz],
                            outputs=[self.out_idx], enable_l2=True)

    def _compute_each_core(self, core_begin, core_end, core_thread,
                           core_max_proc_once):
        self.ubs = UBStuffs(self)
        super(QueryBallPoint, self)._compute_each_core(core_begin, core_end,
                                                       core_thread,
                                                       core_max_proc_once)

    def _compute_one_loop(self, start, end):
        ubs = self.ubs
        dis_col = self.dis_col
        with self.tinst.for_range(0, self.batch) as batch:
            dup_ub_buf(self.const_container, ubs.tmp_out_idx_cnt, 0, 
                       self.dis_row, 0)
            # update samp_xyz as row
            ubs.dis_m.update_xyz(self.samp_xyz, batch, start,
                                 end - start, True)
            # loop iter ori_xyz as col
            col_loop = self.ori_num // dis_col
            col_remain = self.ori_num - col_loop * dis_col
            if (col_loop > 0):
                self._proc_col_loop(ubs, batch, start, end, col_loop, dis_col)

            if (col_remain > 0):
                self._proc_col_remain(ubs, batch, start, end, col_remain)

            # move res to gm
            self._move_out_idx_to_gm(ubs, batch, start, end)

    def _proc_col_loop(self, ubs, batch, start, end, col_loop, dis_col):
        with self.tinst.for_range(0, col_loop) as col_loop_i:
            col_bias = col_loop_i * dis_col
            ubs.dis_m.update_xyz(self.ori_xyz, batch, col_bias, dis_col, 0)
            with self.tinst.for_range(0, end - start) as r_idx:
                with self.tinst.if_scope(ubs.tmp_out_idx_cnt[r_idx] <
                                         self.nsample):
                    self._gather_idx(ubs, r_idx, dis_col, 0, col_bias)
                with self.tinst.else_scope():
                    pass

    def _proc_col_remain(self, ubs, batch, start, end, col_remain):
        col_bias = self.ori_num - col_remain
        ubs.dis_m.update_xyz(self.ori_xyz, batch, col_bias, col_remain, 0)
        align_num = self.dis_pnum
        if (col_remain % align_num != 0):
            align_remain = align_num * (col_remain // align_num) + align_num
            assert(align_remain <= ubs.dis_m.max_col_num)
            dis_init = self.radius + 3
        else:
            align_remain = col_remain
            dis_init = 0

        with self.tinst.for_range(0, end - start) as r_idx:
            with self.tinst.if_scope(ubs.tmp_out_idx_cnt[r_idx] <
                                     self.nsample):
                self._gather_idx(ubs, r_idx, align_remain, dis_init, col_bias)
            with self.tinst.else_scope():
                    pass

    def _move_out_idx_to_gm(self, ubs, batch, start, end):
        t_num = 2 if self.idx_type == "float32" else 1
        with self.tinst.for_range(0, end-start, thread_num=t_num) as r_idx:
            out_idx_offset = r_idx * ubs.tmp_out_idx.shape[1]
            dst_int32 = ubs.tmp_out_idx[out_idx_offset]
            self.tinst.data_move(self.out_idx[batch, 0, start + r_idx, 0],
                                 dst_int32, 0, 1, self.nsample // 8, 0, 0)

    def _gather_idx(self, ubs, r_idx, dis_num, dis_init, idx_bias):
        container = self.const_container
        assert(dis_num <= ubs.dis_m.max_col_num and
              (dis_num % self.dis_pnum == 0))
        with self.tinst.if_scope(dis_init > self.radius):
            # make sure all left mask_res_ub fill with 0 when cmp with radius
            dup_ub_buf(container, ubs.tmp_col_dis, 0, dis_num, dis_init)

        ubs.dis_m.calc_dis_rowcol(r_idx, ubs.tmp_col_dis, True)

        # compare dis with radius and select small ones's idx
        self.tinst.vcmpv_lt(ubs.mask_res_ub, ubs.dis_to_compare, ubs.radius_ub,
                            dis_num // self.dis_pnum, 1, 1, 8, 0)

        # use vreduce in 610. in 310 use a slow way
        cnt_sca = self.tinst.Scalar("int32", name="cnt_sca")
        cnt_sca.set_as(ubs.tmp_out_idx_cnt[r_idx])
        tmp_offset = ubs.tmp_out_idx.shape[1] * r_idx
        ubs.mask_idx.get_idx_with_first_fill(ubs.mask_res_ub, dis_num // 8,
                                             ubs.tmp_out_idx, tmp_offset,
                                             cnt_sca, self.nsample, idx_bias)
        ubs.tmp_out_idx_cnt[r_idx].set_as(cnt_sca)


def _check_same_value(value1, value2, tar_value):
    if (value1 != tar_value and value2 != tar_value):
        raise RuntimeError("all value not equal {}".format(tar_value))


def _cross_check_shapes(ori_shape, samp_shape, res_shape, nsample):
    if (ori_shape[0] != samp_shape[0] or ori_shape[0] != res_shape[0]):
        raise RuntimeError("batch is not valid")
    check.check_param_equal(ori_shape[1], 3, "ori c is not 3")
    check.check_param_equal(samp_shape[1], 3, "samp c is not 3")
    check.check_param_equal(res_shape[1], 1, "res c is not 1")
    _check_same_value(ori_shape[2], ori_shape[3], 1)
    _check_same_value(samp_shape[2], samp_shape[3], 1)
    samp_num = samp_shape[2] * samp_shape[3]
    check.check_param_equal(res_shape[2], samp_num, "err res samp_num")
    check.check_param_equal(res_shape[3], nsample, "err res nsample")


def query_ball_point(ori_points, samp_points, res_idx, radius, nsample,
                     kernel_name="query_ball_point", test=False):
    interface_check.check_kernelname(kernel_name)
    interface_check.check_param(ori_points, [4], ["float16"], ["NCHW"])
    interface_check.check_param(samp_points, [4], ["float16"], ["NCHW"])
    interface_check.check_param(res_idx, [4], ["int32"], ["NCHW"])
    check.check_param_type(nsample, int, "invalid param nsample")
    check.check_param_type(test, bool, "invalid param test")
    ori_shape = ori_points.get("shape")
    samp_shape = samp_points.get("shape")
    res_shape = res_idx.get("shape")
    _cross_check_shapes(ori_shape, samp_shape, res_shape, nsample)

    container = get_aicore_container(("Ascend610",), c3x_support_list=())

    core_type = CoreType.AICORE
    batch = ori_shape[0]
    ori_num = ori_shape[2] * ori_shape[3]
    samp_num = samp_shape[2] * samp_shape[3]
    obj = QueryBallPoint(container, core_type, batch, ori_num, samp_num, 
                         radius, nsample, kernel_name, debug=test)
    obj.compute();
    if test:
        return obj.tinst
    else:
        return 0
