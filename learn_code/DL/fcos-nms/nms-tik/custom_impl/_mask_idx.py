# -*- coding: utf-8 -*-
import importlib

from uti import check
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ascend import AContainer1910


def dup_ub_buf(container, out_ub, out_ub_offset, dup_num, dup_scalar):
    bufs = {"out": AVecBuf(out_ub, dup_num, out_ub_offset, container, False)}
    cmds = [VecGCmd("vector_dup", "out", scalar=dup_scalar)]
    VecGExecutor.exec_vec_g_cmd(container, bufs, cmds, "out")


class MaskIdx(object):
    def __init__(self, container, idx_type="int32", debug=False):
        assert(idx_type in ["int32"])

        self.tik = container.tik
        self.tinst = container.tinst
        self.container = container
        self.debug = debug
        self.idx_type = idx_type
        ub_scope = self.tik.scope_ubuf
        self.mask_idx_map = self.tinst.Tensor(idx_type, (256, 8),
                                              name="mask_idx_map",
                                              scope=ub_scope)
        self.mask_map_num = self.tinst.Tensor("int8", (256,),
                                              name="mask_map_num",
                                              scope=ub_scope)
        # gen mask dict/num
        with self.tinst.new_stmt_scope():
            idx_08 = self.tinst.Tensor(idx_type, (8, ),
                                       name="idx_08", scope=ub_scope)
            self._init_idx_08(idx_08)

            tmp_key = self.tinst.Scalar("int32", name="tmp_key")
            count = self.tinst.Scalar("int8", name="count")
            with self.tinst.for_range(0, 256) as key:
                count.set_as(0)
                tmp_key.set_as(key)
                with self.tinst.for_range(0, 8) as value:
                    with self.tinst.if_scope(tmp_key % 2 == 1):
                        self.mask_idx_map[key, count].set_as(idx_08[value])
                        count.set_as(count + 1)
                    with self.tinst.else_scope():
                        pass
                    tmp_key.set_as(tmp_key // 2)
                self.mask_map_num[key].set_as(count)

    @staticmethod
    def use_ub_byte():
        return 256 * 8 * 4 + 256

    def get_idx_with_first_fill(self, mask_uint8_ub, mask_num, out_idx_ub,
                                out_idx_offset, cnt_sca, need_idx_num,
                                idx_bias):
        """
        input
            mask_uint8_ub:  uint8 ub stores mask res. one elm for 8 res(1 or 0)
                            usually from vcmpv_xx.
            mask_num:       len(mask_uint8_ub)
            need_idx_num:   total idx num needed to get.None or Scalar/Tik Expr
            out_idx_ub:     res idx in gm
            out_idx_offset: out_idx_ub's offset
            cnt_sca:        cur total idx cnt
            idx_bias:       idx bias add to out_idx_ub
            e.g.  if    mask_uint8_ub = [5,0,0,2], mask_num = 4, need_idx_num=4
                then    out_idx_ub = [2,3,1+24]. and return 3
                  if    need_idx_num = 2, then out_idx_ub = [2,3], and return 2
        return cur mask num
        """
        self._check_get_idx_params(mask_uint8_ub, mask_num, out_idx_ub,
                                   out_idx_offset, cnt_sca, need_idx_num,
                                   idx_bias)
        # start calc idx
        tmp_th = need_idx_num if (need_idx_num is not None) else mask_num
        tmp_res = self.tinst.Tensor(self.idx_type, (8,),
                                    name="tmp_res", scope=self.tik.scope_ubuf)
        tmp_key = self.tinst.Scalar("uint8", name="tmp_key")
        tmp_num = self.tinst.Scalar("int8", name="tmp_num")
        tmp_idx = self.tinst.Scalar(self.idx_type, name="tmp_i", init_value=0)
        with self.tinst.for_range(0, mask_num) as offset:
            with self.tinst.if_scope(mask_uint8_ub[offset] > 0):
                with self.tinst.if_scope(cnt_sca < tmp_th):
                    tmp_key.set_as(mask_uint8_ub[offset])
                    self._calc_idx(tmp_res, tmp_idx, tmp_key, offset, idx_bias)
                    self._fill_idx(out_idx_ub, cnt_sca, tmp_res, tmp_idx,
                                   tmp_key, tmp_num,
                                   out_idx_offset, need_idx_num)
                with self.tinst.else_scope():
                    pass
            with self.tinst.else_scope():
                    pass

    def _check_get_idx_params(self, mask_uint8_ub, mask_num, out_idx_ub,
                              out_idx_offset, cnt_sca, need_idx_num, idx_bias):
        tik = self.tik
        check.check_tik_tensor(mask_uint8_ub, ["local.UB"], ["uint8"], tik)
        check.check_tik_tensor(out_idx_ub, ["local.UB"], [self.idx_type], tik)
        assert("int32" == cnt_sca.dtype)
        if (self.debug):
            err_str = '"MaskIdx get_idx invalid mask_num:"+str(mask_num)'
            check.check_tik_param_low(mask_num, tik, self.tinst, 0, err_str)
            check.check_tik_param_not_equal(mask_num, tik, self.tinst, 0,
                                            err_str)

            err_str = '"MaskIdx get_idx out_idx_offset:"+str(out_idx_offset)'
            check.check_tik_param_low(out_idx_offset, tik, self.tinst, 0,
                                      err_str)

            err_str = '"MaskIdx get_idx need_idx_num:"+str(need_idx_num)'
            check.check_tik_param_low(need_idx_num, tik, self.tinst, 0,
                                      err_str)
            check.check_tik_param_not_equal(need_idx_num, tik, self.tinst, 0,
                                            err_str)
            check.check_tik_param_high(need_idx_num, tik, self.tinst, 65500,
                                       err_str)

            err_str = '"MaskIdx get_idx idx_bias:"+str(idx_bias)'
            check.check_tik_param_low(idx_bias, tik, self.tinst, 0, err_str)
            check.check_tik_param_mod(idx_bias, tik, self.tinst, 8, err_str)

            err_str = '"MaskIdx get_idx cnt_sca:"+str(cnt_sca)'
            check.check_tik_param_low(cnt_sca, tik, self.tinst, 0, err_str)
            check.check_tik_param_high(cnt_sca, tik, self.tinst, need_idx_num,
                                       err_str)
            check.check_tik_param_not_equal(cnt_sca, tik, self.tinst,
                                            need_idx_num, err_str)

    def _init_idx_08(self, idx_08):
        tmp_s = self.tinst.Scalar(self.idx_type, name="tmp_s", init_value=0)
        idx_08[0].set_as(tmp_s)
        tmp_s.set_as(1)
        idx_08[1].set_as(tmp_s)
        tmp_s.set_as(2)
        idx_08[2].set_as(tmp_s)
        tmp_s.set_as(3)
        idx_08[3].set_as(tmp_s)
        tmp_s.set_as(4)
        idx_08[4].set_as(tmp_s)
        tmp_s.set_as(5)
        idx_08[5].set_as(tmp_s)
        tmp_s.set_as(6)
        idx_08[6].set_as(tmp_s)
        tmp_s.set_as(7)
        idx_08[7].set_as(tmp_s)

    def _calc_idx(self, tmp_res, tmp_idx, tmp_key, offset, idx_bias):
        self.tinst.vadds(8, tmp_res, self.mask_idx_map[tmp_key, 0],
                         8 * offset + idx_bias, 1, 0, 0, 0, 0)

    def _fill_idx(self, out_idx_ub, cnt_sca, tmp_res, tmp_idx, tmp_key,
                  tmp_num, out_idx_offset, need_idx_num):
        with self.tinst.if_scope(cnt_sca == 0):
            tmp_idx.set_as(tmp_res[0])
            dup_ub_buf(self.container, out_idx_ub, out_idx_offset,
                       need_idx_num, tmp_idx)
        with self.tinst.else_scope():
            pass

        tmp_num.set_as(self.mask_map_num[tmp_key])
        with self.tinst.for_range(0, tmp_num) as tmp_i:
            out_idx_ub[out_idx_offset + cnt_sca + tmp_i].set_as(
                                        tmp_res[tmp_i])

        cnt_sca.set_as(cnt_sca + tmp_num)
