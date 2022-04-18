"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

s_relu
"""
import re

import numpy as np
from te import tik


UB_BUFF_MAX = 240 * 1024


class Srelu():
    def __init__(self, shape, channel_shared, kern_name):
        if (channel_shared != False):
            raise RuntimeError("not support channel_shared")
        self.dim_n = shape[0]
        self.dim_c = shape[1] * 16
        self.dim_h = shape[2]
        self.dim_w = shape[3]
        self.dim_c1 = shape[1]
        self.hw_ = shape[2] * shape[3]
        # Consider that hw was too bigger to make ub space overflow
        max_hw = (UB_BUFF_MAX - 16 * self.dim_c1 * 2 * 4) // (3 * 16 * 2)
        if (max_hw <= 0):
            raise RuntimeError("dim c can not bigger than", 30704)
        self.channel_shared = channel_shared
        self.kernel_name = kern_name
        self.aicore_use = 10
        self.aicore_in_use_select(self.hw_)
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.gm_input = self.tik_instance.Tensor("float16", 
                                                (self.dim_n, self.dim_c1, 
                                                self.dim_h, self.dim_w, 16),
                                                name="gm_input", 
                                                scope=tik.scope_gm)
        self.gm_output = self.tik_instance.Tensor("float16", 
                                                (self.dim_n, self.dim_c1, 
                                                self.dim_h, self.dim_w, 16), 
                                                name="gm_output", 
                                                scope=tik.scope_gm)
        self.gm_tr = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="gm_tr", scope=tik.scope_gm)
        self.gm_ar = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="gm_ar", scope=tik.scope_gm)
        self.gm_tl = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="gm_tl", scope=tik.scope_gm)
        self.gm_al = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="gm_al", scope=tik.scope_gm)
        self.max_hw = max_hw
        if (max_hw >= self.xlen_each_core):
            self.max_hw = self.xlen_each_core
            self.cycle_hw = 1
        else:
            self.cycle_hw = (self.xlen_each_core + max_hw - 1) // max_hw
        self.ask_for_ub()
    
    def ask_for_ub(self):
        self.ub_tr = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="ub_tr", 
                                            scope=tik.scope_ubuf)
        self.ub_ar = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="ub_ar", 
                                            scope=tik.scope_ubuf)
        self.ub_tl = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="ub_tl", 
                                            scope=tik.scope_ubuf)
        self.ub_al = self.tik_instance.Tensor("float16", 
                                            (1, self.dim_c1, 1, 1, 16), 
                                            name="ub_al", 
                                            scope=tik.scope_ubuf)
        self.ub_step = self.tik_instance.Tensor("float16", 
                                            (1, 1, self.max_hw, 16), 
                                            name="ub_step",
                                            scope=tik.scope_ubuf)
        self.ub_out = self.tik_instance.Tensor("float16", 
                                            (1, 1, self.max_hw, 16), 
                                            name="ub_out", 
                                            scope=tik.scope_ubuf)
        self.ub_tmpout = self.tik_instance.Tensor("float16", 
                                                (1, 1, self.max_hw, 16), 
                                                name="ub_tmpout",
                                                scope=tik.scope_ubuf)
    
    def aicore_in_use_select(self, length):
        self.xlen_each_core = (length + self.aicore_use - 1) // self.aicore_use
        self.xlen_last_core = length - self.xlen_each_core * \
                            (self.aicore_use - 1)
        self.aicore_use = \
            (length + self.xlen_each_core - 1) // self.xlen_each_core
        if (self.aicore_use == 1):
            self.xlen_last_core = self.xlen_each_core

    def check_params(self, dtype):
        if dtype != "float16":
            raise RuntimeError("data type only support float16")
        if self.kernel_name is None:
            raise RuntimeError("kernel_name can not be None, but got %s" %
                            type(self.kernel_name))
        if not isinstance(self.kernel_name, str):
            raise RuntimeError("kernel_name must be string, but got %s" %
                            type(self.kernel_name))
        if len(self.kernel_name) > 200:
            err_str = "kernel_name len must be less than %d, but got %d" % \
                            (200, len(self.kernel_name))
            raise RuntimeError(err_str)
        pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        if not pattern.match(self.kernel_name):
            err_str = "kernel_name can only contain letters, \
                numbers and underscores, \
                and begin with underscores or letters"
            raise RuntimeError(err_str)  

    def mode1_compute(self):
        with self.tik_instance.for_range(0, 
                                        self.aicore_use, 
                                        block_num=self.aicore_use) as index:
            with self.tik_instance.if_scope(index != self.aicore_use - 1):
                self.mode1_compute_each_core(self.xlen_each_core, 
                                            (index * self.xlen_each_core))
            with self.tik_instance.else_scope():
                self.mode1_compute_each_core(self.xlen_last_core, 
                                            (index * self.xlen_each_core))
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.gm_input, 
                                        self.gm_tr, 
                                        self.gm_ar, 
                                        self.gm_tl, 
                                        self.gm_al],
                                   outputs=[self.gm_output])
        return self.tik_instance
        
    def mode1_compute_each_core(self, length, offset):
        self.cur_hw = self.tik_instance.Scalar("uint32")
        self.max_times = self.tik_instance.Scalar("uint16")
        if (self.channel_shared == False):
            self.tik_instance.data_move(self.ub_tr, 
                                        self.gm_tr, 0, 1, 
                                        self.dim_c1, 0, 0, 0)
            self.tik_instance.data_move(self.ub_ar, 
                                        self.gm_ar, 0, 1, 
                                        self.dim_c1, 0, 0, 0)
            self.tik_instance.data_move(self.ub_tl, 
                                        self.gm_tl, 0, 1, 
                                        self.dim_c1, 0, 0, 0)
            self.tik_instance.data_move(self.ub_al, 
                                        self.gm_al, 0, 1, 
                                        self.dim_c1, 0, 0, 0)
        self.mode1_compute_each_core_process_one(length, offset)
    
    def mode1_compute_each_core_process_one(self, length, offset):
        with self.tik_instance.for_range(0, self.dim_n) as i_n:
            with self.tik_instance.for_range(0, self.dim_c1) as i_c1:
                with self.tik_instance.for_range(0, 
                                                self.cycle_hw) as i_cycle_hw:
                    self.mode1_compute_each_core_process_two(length, offset,
                                                            i_n, i_c1, 
                                                            i_cycle_hw)

    def mode1_compute_each_core_process_two(self, length, offset,
                                            i_n, i_c1, i_cycle_hw):
        with self.tik_instance.if_scope(
                                        i_cycle_hw != (self.cycle_hw - 1)):
            self.cur_hw.set_as(self.max_hw)
        with self.tik_instance.else_scope():
            self.cur_hw.set_as(length - (self.max_hw * i_cycle_hw))
        hoffset = (offset + i_cycle_hw * self.max_hw) // self.dim_w
        woffset = (offset + i_cycle_hw * self.max_hw) - \
                hoffset * self.dim_w
        self.tik_instance.data_move(self.ub_step, 
                                    self.gm_input[i_n, i_c1, 
                                        hoffset, woffset, 0], 
                                    0, 1, self.cur_hw, 0, 0, 0)
        
        cycle1 = self.cur_hw // (8 * 255)
        remain1 = (self.cur_hw - cycle1 * 8 * 255) // 8
        remain2 = (self.cur_hw - cycle1 * 8 * 255) - remain1 * 8
        with self.tik_instance.if_scope(cycle1 != 0):
            self.max_times.set_as(255)
            with self.tik_instance.for_range(0, 
                                            cycle1) as i_cycle1:
                offset = 128 * 255 * i_cycle1
                self.compute(128, offset, i_c1, 
                            self.max_times, 8, 8, 8)
        with self.tik_instance.else_scope():
            self.max_times.set_as(0)
            pass

        offset = 128 * 255 * cycle1
        with self.tik_instance.if_scope(remain1 != 0):
            self.compute(128, offset, i_c1, remain1, 8, 8, 8)
        with self.tik_instance.else_scope():
            pass

        offset = offset + 128 * remain1
        with self.tik_instance.if_scope(remain2 != 0):
            self.compute(16, offset, i_c1, remain2, 1, 1, 1)
        with self.tik_instance.else_scope():
            pass

        self.tik_instance.data_move(self.gm_output[i_n, 
                                                i_c1, 
                                                hoffset, 
                                                woffset, 0], 
                                    self.ub_out, 0, 1,
                                    self.cur_hw, 0, 0, 0)

    def compute(self, mask, offset, i_c1, repeat_times, dst_r_stride, 
                src0_r_stride, src1_r_stride):
        # use 4 steps to get srelu result.
        # 1. pick out src > tr
        # 2. pick out src < tl
        # 3. pick out tl < src < tr
        # 4. add above 3 steps result
        # set dst equals to ar * (src - tr) when src greater than tr,
        # set dst = 0 when src less than or equal to tr
        self.tik_instance.vmax(mask, self.ub_out[offset], 
                            self.ub_step[offset], 
                            self.ub_tr[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        self.tik_instance.vsub(mask, self.ub_out[offset], 
                            self.ub_out[offset], 
                            self.ub_tr[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        self.tik_instance.vmul(mask, self.ub_out[offset], 
                            self.ub_out[offset], 
                            self.ub_ar[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        # set dst equals to al * (src - tl) when src less than tl,
        # set dst = 0 when src greater than or equal to tl
        self.tik_instance.vmin(mask, self.ub_tmpout[offset], 
                            self.ub_step[offset], 
                            self.ub_tl[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        self.tik_instance.vsub(mask, self.ub_tmpout[offset], 
                            self.ub_tmpout[offset], 
                            self.ub_tl[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        self.tik_instance.vmul(mask, self.ub_tmpout[offset], 
                            self.ub_tmpout[offset], 
                            self.ub_al[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        self.tik_instance.vadd(mask, self.ub_out[offset], 
                            self.ub_out[offset], self.ub_tmpout[offset], 
                            repeat_times, 1, 1, 1, 
                            dst_r_stride, src0_r_stride, src1_r_stride)
        # set dst equals to tr when src greater than tr
        # set dst equals to tl when src less than tl
        # set dst equals to src when src greater than tl and less than tr
        self.tik_instance.vmin(mask, self.ub_tmpout[offset], 
                            self.ub_step[offset], 
                            self.ub_tr[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        self.tik_instance.vmax(mask, self.ub_tmpout[offset], 
                            self.ub_tmpout[offset], 
                            self.ub_tl[0, i_c1, 0, 0, 0],
                            repeat_times, 1, 1, 0, 
                            dst_r_stride, src0_r_stride, 0)
        self.tik_instance.vadd(mask, self.ub_out[offset], 
                            self.ub_out[offset], self.ub_tmpout[offset],
                            repeat_times, 1, 1, 1, 
                            dst_r_stride, src0_r_stride, src1_r_stride)


def check_input_shape(input_x, tr_, ar_, tl_, al_):
    x_shape = input_x.get('shape')
    tr_shape = tr_.get('shape')
    ar_shape = ar_.get('shape')
    tl_shape = tl_.get('shape')
    al_shape = al_.get('shape')
    if x_shape[1] != tr_shape[1]:
        raise RuntimeError("input data's dim c is not equal to tr dim c")
    if x_shape[1] != ar_shape[1]:
        raise RuntimeError("input data's dim c is not equal to ar dim c")
    if x_shape[1] != tl_shape[1]:
        raise RuntimeError("input data's dim c is not equal to tl dim c")
    if x_shape[1] != al_shape[1]:
        raise RuntimeError("input data's dim c is not equal to al dim c") 


def s_re_lu(input_x, 
            tr_, 
            ar_, 
            tl_, 
            al_, 
            output_y, 
            channel_shared, 
            kern_name="s_re_lu"):
    """
    piecewise linear activation function
    :param input_x: input
    :param tr: tr
    :param ar: ar
    :param tl: tl
    :param al: al
    :param output_y: output
    :param channel_shared:
    :param kern_name:kern_name
    :constrictions:
        1) channel_shared must be false
        2) Dim(c) of input_x must be less than 30704
    """
    check_input_shape(input_x, tr_, ar_, tl_, al_)
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    obj = Srelu(shape, channel_shared, kern_name)
    obj.check_params(dtype)
    obj.mode1_compute()
