#!/usr//bin//env python
# -*- coding:utf-8 -*-
"""
copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org//licenses//LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License == distributed on an "AS IS" BASIS,p
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

correlation
"""

from te import tik
from uti import interface_check
from ascend import AContainer1951
from version.get_version import get_aicore_container


class TikCorrelation():
    """
    Associate each pixel in the first featuremap with a range of pixels in the
    second featuremap
    :param input_img0：the previous frame of the featuremap with the
     shape[1, 96, 48, 64]
    :param input_img1: the next frame of the featuremap with the shape the same
     as the img0
    """

    def __init__(self, container, kernel_name='Correlation'):
        interface_check.check_kernelname(kernel_name)
        self.kernel_name = kernel_name
        self.tik_instance = container.tinst
        self.tik = container.tik
        self.input_img0 = self.tik_instance.Tensor("float16", (1, 96, 48, 64),
                                                   name="img0",
                                                   scope=self.tik.scope_gm)
        self.input_img1 = self.tik_instance.Tensor("float16", (1, 96, 48, 64),
                                                   name="img1",
                                                   scope=self.tik.scope_gm)
        self.featuremap = self.tik_instance.Tensor("float16", (1, 441, 48, 64),
                                                   name="featuremap",
                                                   scope=self.tik.scope_gm)

        self.img1_big = self.tik_instance.Tensor("float16",
                                                 (1, 6, 51, 104, 16),
                                                 name="img1_big",
                                                 scope=self.tik.scope_cbuf)

        self.h_base = self.tik_instance.Scalar("int64")

    def global_init(self):
        """
        Basic initialization;
        """
        print("global init success ! kernel_name:", self.kernel_name)

    def _transpose(self, img_src, img_dst):
        with self.tik_instance.for_range(0, 4) as k:
            self.tik_instance.vnchwconv(True, True,
                                        [img_dst[0, 0, 16 * k + 0, 0],
                                         img_dst[0, 0, 16 * k + 1, 0],
                                         img_dst[0, 0, 16 * k + 2, 0],
                                         img_dst[0, 0, 16 * k + 3, 0],
                                         img_dst[0, 0, 16 * k + 4, 0],
                                         img_dst[0, 0, 16 * k + 5, 0],
                                         img_dst[0, 0, 16 * k + 6, 0],
                                         img_dst[0, 0, 16 * k + 7, 0],
                                         img_dst[0, 0, 16 * k + 8, 0],
                                         img_dst[0, 0, 16 * k + 9, 0],
                                         img_dst[0, 0, 16 * k + 10, 0],
                                         img_dst[0, 0, 16 * k + 11, 0],
                                         img_dst[0, 0, 16 * k + 12, 0],
                                         img_dst[0, 0, 16 * k + 13, 0],
                                         img_dst[0, 0, 16 * k + 14, 0],
                                         img_dst[0, 0, 16 * k + 15, 0]],
                                        [img_src[0, 0, 0, 16 * k],
                                         img_src[0, 1, 0, 16 * k],
                                         img_src[0, 2, 0, 16 * k],
                                         img_src[0, 3, 0, 16 * k],
                                         img_src[0, 4, 0, 16 * k],
                                         img_src[0, 5, 0, 16 * k],
                                         img_src[0, 6, 0, 16 * k],
                                         img_src[0, 7, 0, 16 * k],
                                         img_src[0, 8, 0, 16 * k],
                                         img_src[0, 9, 0, 16 * k],
                                         img_src[0, 10, 0, 16 * k],
                                         img_src[0, 11, 0, 16 * k],
                                         img_src[0, 12, 0, 16 * k],
                                         img_src[0, 13, 0, 16 * k],
                                         img_src[0, 14, 0, 16 * k],
                                         img_src[0, 15, 0, 16 * k]], 6, 1, 64)

    def _process_imag_h_0(self):
        with self.tik_instance.for_range(0, 20) as img_j:
            top_pad = self.tik_instance.Tensor("float16", (1, 6, 1, 64, 16),
                                               name="top_pad",
                                               scope=self.tik.scope_ubuf)
            self.tik_instance.vector_dup(128, top_pad, 0, 48, 1, 8, 0)
            self.tik_instance.data_move(self.img1_big[0, 0, img_j, 20, 0],
                                        top_pad, 0, 6, 64, 0, 5240)

        with self.tik_instance.for_range(0, 50, thread_num=2) as img_i:
            left_pad = self.tik_instance.Tensor("float16", (1, 6, 1, 20, 16),
                                                name="left_pad",
                                                scope=self.tik.scope_ubuf)
            self.tik_instance.vector_dup(128, left_pad, 0, 15, 1, 8, 0)
            self.tik_instance.data_move(self.img1_big[0, 0, img_i, 0, 0],
                                        left_pad, 0, 6, 20, 0, 5284)
            self.tik_instance.data_move(self.img1_big[0, 0, img_i, 84, 0],
                                        left_pad, 0, 6, 20, 0, 5284)

        with self.tik_instance.for_range(0, 30, thread_num=2) as img_f:
            img_dst_front = self.tik_instance.Tensor("float16", (1, 1, 64, 96),
                                                     name="img_dst_front",
                                                     scope=self.tik.scope_ubuf)
            img_src_front = self.tik_instance.Tensor("float16", (1, 96, 1, 64),
                                                     name="img_src_front",
                                                     scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(img_src_front,
                                        self.input_img1[0, 0, img_f, 0], 0, 96,
                                        4, 188, 0)

            self._transpose(img_src_front, img_dst_front)

            with self.tik_instance.for_range(0, 6) as imag_b:
                self.tik_instance.data_move(
                    self.img1_big[0, imag_b, 20 + img_f, 20, 0],
                    img_dst_front[0, 0, 0, 16 * imag_b], 0,
                    64, 1, 5, 0)

    def _process_imag_h_10(self):
        with self.tik_instance.for_range(0, 40, thread_num=2) as img_f:
            scope_ubuf = self.tik.scope_ubuf
            img_dst_middle = self.tik_instance.Tensor("float16",
                                                      (1, 1, 64, 96),
                                                      name="img_dst_middle",
                                                      scope=scope_ubuf)
            img_src_middle = self.tik_instance.Tensor("float16",
                                                      (1, 96, 1, 64),
                                                      name="img_src_middle",
                                                      scope=scope_ubuf)
            self.tik_instance.data_move(img_src_middle,
                                        self.input_img1[0, 0, img_f, 0], 0, 96,
                                        4, 188, 0)

            self._transpose(img_src_middle, img_dst_middle)

            with self.tik_instance.for_range(0, 6) as imag_b:
                self.tik_instance.data_move(
                    self.img1_big[0, imag_b, 10 + img_f, 20, 0],
                    img_dst_middle[0, 0, 0, 16 * imag_b],
                    0, 64, 1, 5, 0)

    def _process_imag_h_20(self):
        with self.tik_instance.for_range(0, 48, thread_num=2) as img_f:
            img_dst_last = self.tik_instance.Tensor("float16", (1, 1, 64, 96),
                                                    name="img_dst_last",
                                                    scope=self.tik.scope_ubuf)
            img_src_last = self.tik_instance.Tensor("float16", (1, 96, 1, 64),
                                                    name="img_src_last",
                                                    scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(img_src_last,
                                        self.input_img1[0, 0, img_f, 0], 0, 96,
                                        4, 188, 0)

            self._transpose(img_src_last, img_dst_last)

            with self.tik_instance.for_range(0, 6) as imag_b:
                self.tik_instance.data_move(
                    self.img1_big[0, imag_b, img_f, 20, 0],
                    img_dst_last[0, 0, 0, 16 * imag_b],
                    0, 64, 1, 5, 0)

    def _process_imag_h_28(self):
        with self.tik_instance.for_range(0, 10) as j:
            top_pad = self.tik_instance.Tensor("float16", (1, 6, 1, 64, 16),
                                               name="top_pad",
                                               scope=self.tik.scope_ubuf)
            self.tik_instance.vector_dup(128, top_pad, 0, 48, 1, 8, 0)
            self.tik_instance.data_move(self.img1_big[0, 0, 40 + j, 20, 0],
                                        top_pad, 0, 6, 64, 0,
                                        5240)

        with self.tik_instance.for_range(0, 40, thread_num=2) as img_f:
            img_dst_back = self.tik_instance.Tensor("float16", (1, 1, 64, 96),
                                                    name="img_dst_back",
                                                    scope=self.tik.scope_ubuf)
            img_src_back = self.tik_instance.Tensor("float16", (1, 96, 1, 64),
                                                    name="img_src_back",
                                                    scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(img_src_back,
                                        self.input_img1[0, 0, 8 + img_f, 0], 0,
                                        96, 4, 188,
                                        0)

            self._transpose(img_src_back, img_dst_back)

            with self.tik_instance.for_range(0, 6) as imag_b:
                self.tik_instance.data_move(
                    self.img1_big[0, imag_b, img_f, 20, 0],
                    img_dst_back[0, 0, 0, 16 * imag_b], 0, 64, 1, 5, 0)

    def _process_imag_h_38(self):
        with self.tik_instance.for_range(0, 10, thread_num=2) as j:
            top_pad = self.tik_instance.Tensor("float16", (1, 6, 1, 64, 16),
                                               name="top_pad",
                                               scope=self.tik.scope_ubuf)
            self.tik_instance.vector_dup(128, top_pad, 0, 48, 1, 8, 0)
            self.tik_instance.data_move(self.img1_big[0, 0, 30 + j, 20, 0],
                                        top_pad, 0, 6, 64,
                                        0, 5240)

        with self.tik_instance.for_range(0, 30, thread_num=2) as img_f:
            img_dst_final = self.tik_instance.Tensor("float16", (1, 1, 64, 96),
                                                     name="img_dst_final",
                                                     scope=self.tik.scope_ubuf)
            img_src_final = self.tik_instance.Tensor("float16", (1, 96, 1, 64),
                                                     name="img_src_final",
                                                     scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(img_src_final,
                                        self.input_img1[0, 0, 18 + img_f, 0],
                                        0, 96, 4, 188, 0)

            self._transpose(img_src_final, img_dst_final)

            with self.tik_instance.for_range(0, 6) as imag_b:
                self.tik_instance.data_move(
                    self.img1_big[0, imag_b, img_f, 20, 0],
                    img_dst_final[0, 0, 0, 16 * imag_b], 0, 64, 1, 5, 0)

    def _imag_tensor(self):
        img0_mod = self.tik_instance.Tensor("float16", (16, 16),
                                            name="img0_mod",
                                            scope=self.tik.scope_cbuf)
        img1_l0a = self.tik_instance.Tensor("float16", (448, 16),
                                            name="img1_l0a",
                                            scope=self.tik.scope_ca)
        img0_l0b = self.tik_instance.Tensor("float16", (16, 16),
                                            name="img0_l0b",
                                            scope=self.tik.scope_cb)

        return img0_mod, img1_l0a, img0_l0b

    def _l0c_tensor(self):
        c_l0c = self.tik_instance.Tensor("float16", (448, 16), name="c_l0c",
                                         scope=self.tik.scope_cc)
        c_ub = self.tik_instance.Tensor("float16", (448, 16), name="c_ub",
                                        scope=self.tik.scope_ubuf)
        c_ub1 = self.tik_instance.Tensor("float16", (28, 16, 16), name="c_ub1",
                                         scope=self.tik.scope_ubuf)

        return c_l0c, c_ub, c_ub1

    def _vnchw_img(self, img_dst1, img_src1):
        with self.tik_instance.for_range(0, 4) as img_z:
            self.tik_instance.vnchwconv(True, True,
                                        [img_dst1[0, 0, 16 * img_z, 0],
                                         img_dst1[0, 0, 16 * img_z + 1, 0],
                                         img_dst1[0, 0, 16 * img_z + 2, 0],
                                         img_dst1[0, 0, 16 * img_z + 3, 0],
                                         img_dst1[0, 0, 16 * img_z + 4, 0],
                                         img_dst1[0, 0, 16 * img_z + 5, 0],
                                         img_dst1[0, 0, 16 * img_z + 6, 0],
                                         img_dst1[0, 0, 16 * img_z + 7, 0],
                                         img_dst1[0, 0, 16 * img_z + 8, 0],
                                         img_dst1[0, 0, 16 * img_z + 9, 0],
                                         img_dst1[0, 0, 16 * img_z + 10, 0],
                                         img_dst1[0, 0, 16 * img_z + 11, 0],
                                         img_dst1[0, 0, 16 * img_z + 12, 0],
                                         img_dst1[0, 0, 16 * img_z + 13, 0],
                                         img_dst1[0, 0, 16 * img_z + 14, 0],
                                         img_dst1[0, 0, 16 * img_z + 15, 0]],
                                        [img_src1[0, 0, 0, 16 * img_z],
                                         img_src1[0, 1, 0, 16 * img_z],
                                         img_src1[0, 2, 0, 16 * img_z],
                                         img_src1[0, 3, 0, 16 * img_z],
                                         img_src1[0, 4, 0, 16 * img_z],
                                         img_src1[0, 5, 0, 16 * img_z],
                                         img_src1[0, 6, 0, 16 * img_z],
                                         img_src1[0, 7, 0, 16 * img_z],
                                         img_src1[0, 8, 0, 16 * img_z],
                                         img_src1[0, 9, 0, 16 * img_z],
                                         img_src1[0, 10, 0, 16 * img_z],
                                         img_src1[0, 11, 0, 16 * img_z],
                                         img_src1[0, 12, 0, 16 * img_z],
                                         img_src1[0, 13, 0, 16 * img_z],
                                         img_src1[0, 14, 0, 16 * img_z],
                                         img_src1[0, 15, 0, 16 * img_z]], 6, 1,
                                        64)
        return img_dst1

    def _vnchw_cub(self, c_ub2, temp_j):
        c_ub3 = self.tik_instance.Tensor("float16", (1, 1, 32, 64),
                                         name="c_ub3",
                                         scope=self.tik.scope_ubuf)
        with self.tik_instance.for_range(0, 2) as temp_k:
            width = 32 * temp_j + 16 * temp_k
            self.tik_instance.vnchwconv(True, True,
                                        [c_ub3[0, 0, 16 * temp_k, 0],
                                         c_ub3[0, 0, 16 * temp_k + 1, 0],
                                         c_ub3[0, 0, 16 * temp_k + 2, 0],
                                         c_ub3[0, 0, 16 * temp_k + 3, 0],
                                         c_ub3[0, 0, 16 * temp_k + 4, 0],
                                         c_ub3[0, 0, 16 * temp_k + 5, 0],
                                         c_ub3[0, 0, 16 * temp_k + 6, 0],
                                         c_ub3[0, 0, 16 * temp_k + 7, 0],
                                         c_ub3[0, 0, 16 * temp_k + 8, 0],
                                         c_ub3[0, 0, 16 * temp_k + 9, 0],
                                         c_ub3[0, 0, 16 * temp_k + 10, 0],
                                         c_ub3[0, 0, 16 * temp_k + 11, 0],
                                         c_ub3[0, 0, 16 * temp_k + 12, 0],
                                         c_ub3[0, 0, 16 * temp_k + 13, 0],
                                         c_ub3[0, 0, 16 * temp_k + 14, 0],
                                         c_ub3[0, 0, 16 * temp_k + 15, 0]],
                                        [c_ub2[0, 0, 0, width],
                                         c_ub2[0, 1, 0, width],
                                         c_ub2[0, 2, 0, width],
                                         c_ub2[0, 3, 0, width],
                                         c_ub2[0, 4, 0, width],
                                         c_ub2[0, 5, 0, width],
                                         c_ub2[0, 6, 0, width],
                                         c_ub2[0, 7, 0, width],
                                         c_ub2[0, 8, 0, width],
                                         c_ub2[0, 9, 0, width],
                                         c_ub2[0, 10, 0, width],
                                         c_ub2[0, 11, 0, width],
                                         c_ub2[0, 12, 0, width],
                                         c_ub2[0, 13, 0, width],
                                         c_ub2[0, 14, 0, width],
                                         c_ub2[0, 15, 0, width]],
                                        4, 1, 448)
        return c_ub3

    def _process_h_0(self, imag_h):
        with self.tik_instance.if_scope(imag_h == 0):
            self.h_base.set_as(imag_h)
            self._process_imag_h_0()
        with self.tik_instance.else_scope():
            pass

    def _process_h_10(self, imag_h):
        with self.tik_instance.if_scope(imag_h == 10):
            self.h_base.set_as(imag_h)
            self._process_imag_h_10()
        with self.tik_instance.else_scope():
            pass

    def _process_h_20(self, imag_h):
        with self.tik_instance.if_scope(imag_h == 20):
            self.h_base.set_as(imag_h)
            self._process_imag_h_20()
        with self.tik_instance.else_scope():
            pass

    def _process_h_28(self, imag_h):
        with self.tik_instance.if_scope(imag_h == 28):
            self.h_base.set_as(imag_h)
            self._process_imag_h_28()
        with self.tik_instance.else_scope():
            pass

    def _process_h_38(self, imag_h):
        with self.tik_instance.if_scope(imag_h == 38):
            self.h_base.set_as(imag_h)
            self._process_imag_h_38()
        with self.tik_instance.else_scope():
            pass

    def _calculate(self, imag_h, imag_w, img_dst1, c_l0c):
        img0_mod, img1_l0a, img0_l0b = self._imag_tensor()
        with self.tik_instance.for_range(0, 6, thread_num=2) as imag_b:
            self.tik_instance.load3dv1(img1_l0a, self.img1_big[
                0, imag_b, imag_h - self.h_base, imag_w, 0],
                                       [0, 0, 0, 0], 41, 104, 0, 0, 0, 0, 0, 2,
                                       2, 64, 1, 1, 1, 1, 1, 28)

            self.tik_instance.data_move(img0_mod,
                                        img_dst1[0, 0, imag_w, 16 * imag_b], 0,
                                        1, 1, 0, 0)
            self.tik_instance.load2dv2(img0_l0b, img0_mod, 0, 1, 0, 0, 0)

            with self.tik_instance.if_scope(imag_b == 0):
                self.tik_instance.mmad(c_l0c, img1_l0a, img0_l0b, 448, 16, 16,
                                       0)

            with self.tik_instance.else_scope():
                self.tik_instance.mmad(c_l0c, img1_l0a, img0_l0b, 448, 16, 16,
                                       1)

    def _results_mov(self, c_ub2, imag_h):
        with self.tik_instance.for_range(0, 14, thread_num=2) as temp_j:
            c_ub3 = self._vnchw_cub(c_ub2, temp_j)
            with self.tik_instance.if_scope(temp_j == 13):
                self.tik_instance.data_move(
                    self.featuremap[0, 32 * temp_j, imag_h, 0], c_ub3, 0, 25,
                    4, 0, 188)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    self.featuremap[0, 32 * temp_j, imag_h, 0], c_ub3, 0, 32,
                    4, 0, 188)

    def compute(self):
        """
        The function compute correlation;
        """
        with self.tik_instance.for_range(0, 48) as imag_h:
            self._process_h_0(imag_h)
            self._process_h_10(imag_h)
            self._process_h_20(imag_h)
            self._process_h_28(imag_h)
            self._process_h_38(imag_h)

            img_dst1 = self.tik_instance.Tensor("float16", (1, 1, 64, 96),
                                                name="img_dst1",
                                                scope=tik.scope_ubuf)
            img_src1 = self.tik_instance.Tensor("float16", (1, 96, 1, 64),
                                                name="img_src1",
                                                scope=tik.scope_ubuf)
            self.tik_instance.data_move(img_src1,
                                        self.input_img0[0, 0, imag_h, 0], 0,
                                        96, 4, 188, 0)

            img_dst1 = self._vnchw_img(img_dst1, img_src1)

            c_ub2 = self.tik_instance.Tensor("float16", (1, 64, 1, 448),
                                             name="c_ub2",
                                             scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, 64, thread_num=2) as imag_w:
                c_l0c, c_ub, c_ub1 = self._l0c_tensor()
                self._calculate(imag_h, imag_w, img_dst1, c_l0c)

                self.tik_instance.data_move(c_ub, c_l0c, 0, 1, 28, 0, 0)

                with self.tik_instance.for_range(0, 28, thread_num=2) \
                        as temp_h:
                    self.tik_instance.vtranspose(c_ub1[temp_h, 0, 0],
                                                 c_ub[16 * temp_h, 0])
                self.tik_instance.vmuls(16, c_ub2[0, imag_w, 0, 0], c_ub1,
                                        1.0 / 96, 28, 1, 1, 1, 16)

            self._results_mov(c_ub2, imag_h)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_img0, self.input_img1],
                                   outputs=[self.featuremap])
        return self.tik_instance


def correlation(input0, input1, output0, pad=20, kernel_size=1,
                max_displacement=20, stride_1=1, stride_2=2,
                kernel_name='Correlation'):
    """
    :param input0: the previous frame of the featuremap with the
     shape[1, 96, 48, 64]
    :param input1: the next frame of the featuremap with the shape the same as
     the input0
    :param output0: the output of correlation
    :param pad: the pad of correlation
    :param kernel_size: the kernel_size of correlation
    :param max_displacement: the max_displacement of correlation
    :param stride_1: the stride_1 of correlation
    :param stride_2: the stride_2 of correlation
    :param kernel_name: the kernel_name of correlation
    :return:
    """
    if input0["shape"] != (1, 96, 48, 64):
        raise RuntimeError("input0_shape only support [1, 96, 48, 64]!")
    if input1["shape"] != (1, 96, 48, 64):
        raise RuntimeError("input1_shape only support [1, 96, 48, 64]!")
    if output0["shape"] != (1, 441, 48, 64):
        raise RuntimeError("output_shape should be [1, 441, 48, 64]!")
    if pad != 20:
        raise RuntimeError("pad should be 20!")
    if kernel_size != 1:
        raise RuntimeError("kernel_size should be 1!")
    if max_displacement != 20:
        raise RuntimeError("max_displacement should be 20!")
    if stride_1 != 1:
        raise RuntimeError("stride_1 should be 1!")
    if stride_2 != 2:
        raise RuntimeError("stride_2 should be 2!")
    container = get_aicore_container(("Ascend610",), c3x_support_list=())
    obj = TikCorrelation(container, kernel_name)
    obj.global_init()
    obj.compute()
