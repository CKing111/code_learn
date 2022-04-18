# coding: utf-8

import math
from uti import interface_check
from ascend import AContainer1951
from version.get_version import get_aicore_container


class DeconvolutionV1:

    def __init__(self, container, input_data, weight, bias, output_data,
                 strides, pads, dilations, groups, data_format, offset_x,
                 kernel_name):
        """
        Deconvolution
        @param input_data: input data
        @param weight: deconvolution kernel
        @param bias: deconvolution bias
        @param output_data: output data
        @param num_output: num of output data
        @param bias_term: if bias exist
        @param pad: zero padding height and weight
        @param kernel_size: kernel'index_1 height and weight
        @param group: num of deconvolution group
        @param stride: kernel move stride
        @param kernel_name: kernel name
        :return:NA
        """
        self.tik_inst = container.tinst
        self.tik = container.tik
        self.input_data_dict = input_data
        self.weight_dict = weight
        self.output_data_dict = output_data
        self.num_output = weight.get("shape")[1] * groups
        self.use_bias = bias is not None
        self.pad = pads[0]
        self.k_size = weight.get("shape")[2]
        self.offset_x = offset_x
        self.group = groups
        self.stride = strides[0]
        self.kernel_name = kernel_name

        self.k_stride = self.k_size // self.stride
        self.k_stride_square = self.k_stride * self.k_stride
        self.p_size = self.k_size - 1 - self.pad
        self.width_k = self.num_output // self.group
        input_data_shape = input_data.get("shape")
        weight_data_shape = weight.get("shape")
        self.n_batch, self.channel_1, self.height, self.width, self.channel_0\
            = input_data_shape
        self.channel = weight_data_shape[0]

        interface_check.check_kernelname(kernel_name)
        interface_check.check_param(input_data, [5], ["float16"], ["NC1HWC0"])
        interface_check.check_param(output_data, [5], ["float16"], ["NC1HWC0"])
        interface_check.check_shape(input_data.get("ori_shape"), [4], ['NCHW'])
        interface_check.check_shape(output_data.get("ori_shape"), [4],
                                    ['NCHW'])
        self._input_checkout()
        self._attr_checkout()
        self._output_checkout()
        output_data_shape = output_data.get("shape")
        self.output_n, self.output_c1, self.output_h, self.output_w,\
        self.output_c0 = output_data_shape
        self.aicore_use = 2

        self.input_data = self.tik_inst.Tensor(
            "float16", input_data_shape, name="input_data",
            scope=self.tik.scope_gm)
        self.weight = self.tik_inst.Tensor(
            "float16", weight_data_shape, name="weight",
            scope=self.tik.scope_gm)
        if self.use_bias:
            self.bias_dict = bias
            self._bias_checkout()
            self.bias = self.tik_inst.Tensor(
                "float16", bias.get("shape"), name="bias",
                scope=self.tik.scope_gm)
        self.output_data = self.tik_inst.Tensor(
            "float16", output_data_shape, name="output_data",
            scope=self.tik.scope_gm)

    def _input_checkout(self):
        weight_data_shape = self.weight_dict.get("shape")
        if weight_data_shape[1:] != (self.width_k, self.k_size, self.k_size):
            raise RuntimeError("weight shape does not match")
        if self.offset_x != 0:
            raise RuntimeError("offset_x is not supported!")

    def _attr_checkout(self):
        l0a_size = 64 * 1024
        l0b_size = 64 * 1024
        l0c_size = 256 * 1024
        ub_size = 240 * 1024
        l1_size = 1024 * 1024
        unit_size = 2
        if self.k_size - self.stride - 2 * self.pad != 0:
            raise RuntimeError("unsuitable kernel_size, stride and pad!")

        if ((self.k_size - 1 - self.pad) // self.stride < 0 or
                (self.k_size - 1 - self.pad) // self.stride > 255):
            raise RuntimeError("the pad should be in the range of [0, 255] !")
        if math.ceil(
                self.width / 16.0) * 16 * 16 * self.k_stride_square *\
                unit_size > l0a_size:
            raise RuntimeError(
                "width and (kernel_size // stride)^2 are too big to cause l0a"
                " overflow!")
        if 256 * self.k_stride_square * unit_size > l0b_size:
            raise RuntimeError(
                "(kernel_size // stride)^2 is too big to cause l0b overflow!")
        if self.stride * math.ceil(
                self.width / 16.0) * 16 * unit_size > l0c_size:
            raise RuntimeError(
                "stride * ceil(w, 16) are too big to cause l0c overflow!")
        if (self.k_size * self.k_size * 256 * unit_size +
                2 * self.k_stride * self.width * 16 * unit_size > l1_size):
            raise RuntimeError(
                "kernel_size, width and kernel_size // stride are too big to"
                " cause L1 overflow!")
        self._attr_checkout_inner(unit_size, ub_size)

    def _attr_checkout_inner(self, unit_size, ub_size):
        if self.group > 1:
            if (self.k_size * self.k_size * 16 * 3 * unit_size +
                    self.stride * math.ceil(
                        self.width / 16.0) * 256 * unit_size + 0.53 * 1024 >
                    ub_size):
                raise RuntimeError(
                    "kernel_size and width are too big to cause UB overflow!")
            if (self.k_size * self.k_size * 16 * 3 * unit_size +
                    16 * self.k_size * self.k_size * 16 * unit_size +
                    0.53 * 1024 > ub_size):
                raise RuntimeError(
                    "kernel_size is too big to cause UB overflow!")
        else:
            if (self.k_size * self.k_size * 16 * 3 * unit_size +
                    self.stride * math.ceil(
                        self.width / 16.0) * 256 * unit_size > ub_size):
                raise RuntimeError(
                    "kernel_size and width are too big to cause UB overflow!")

    def _output_checkout(self):
        output_data_shape = self.output_data_dict.get("shape")
        output_shape = (self.n_batch,
                        int(math.ceil(self.num_output / 16.0)),
                        self.stride * (self.height - 1) + self.k_size - 2 *
                        self.pad, self.stride * (self.width - 1) + self.k_size
                        - 2 * self.pad, 16)
        if output_data_shape != output_shape:
            raise RuntimeError("output data shape does not match")
        if self.output_data_dict.get("dtype") != "float16":
            raise RuntimeError("output type does not match")

    def _bias_checkout(self):
        bias_shape = self.bias_dict.get("shape")
        if bias_shape != (self.num_output,):
            raise RuntimeError("bias shape does not match")
        if self.bias_dict.get("dtype") != "float16":
            raise RuntimeError("bias type does not match")

    def aicore_in_use_select(self, length):
        self.num_each_core = (length + self.aicore_use - 1) // self.aicore_use
        self.num_last_core = length - self.num_each_core * (
                    self.aicore_use - 1)
        self.aicore_use = (length + self.num_each_core - 1) //\
                          self.num_each_core
        if (self.aicore_use == 1):
            self.num_last_core = self.num_each_core

    def tilling_mode_select(self):
        if self.group == 1:
            self.mode = 1
        else:
            self.mode = 2

    def model_compute_gen(self):
        with self.tik_inst.for_range(0, self.n_batch) as batch:
            self.aicore_in_use_select(self.output_c1)
            with self.tik_inst.for_range(
                    0, self.aicore_use, block_num=self.aicore_use
            ) as index:
                with self.tik_inst.if_scope(index != self.aicore_use - 1):
                    start = self.num_each_core * index
                    end = self.num_each_core * (index + 1)
                    self.compute_each_core_gen(batch, start, end)

                with self.tik_inst.else_scope():
                    start = self.num_each_core * index
                    end = self.output_c1
                    self.compute_each_core_gen(batch, start, end)

        if self.use_bias:
            inputs = [self.input_data, self.weight, self.bias]
        else:
            inputs = [self.input_data, self.weight]
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=inputs,
                               outputs=[self.output_data], enable_l2=True)

    def model_compute_gro(self):
        with self.tik_inst.for_range(0, self.n_batch) as batch:
            self.aicore_in_use_select(self.channel_1)

            with self.tik_inst.for_range(0, self.aicore_use,
                                         block_num=self.aicore_use) as index:
                with self.tik_inst.if_scope(index != self.aicore_use - 1):
                    start = self.num_each_core * index
                    end = self.num_each_core * (index + 1)
                    self.compute_each_core_gro(batch, start, end)
                with self.tik_inst.else_scope():
                    start = self.num_each_core * index
                    end = self.channel_1
                    self.compute_each_core_gro(batch, start, end)

        if self.use_bias:
            inputs = [self.input_data, self.weight, self.bias]
        else:
            inputs = [self.input_data, self.weight]
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=inputs,
                               outputs=[self.output_data], enable_l2=True)

    def weight_transpose(self, kernel_nchw, kernel_nhwc):
        # nchw->nhwc self.k_size^2 must be the times of 16
        dst_list = [kernel_nhwc[i // self.k_size, i % self.k_size, 0] for i in
                    range(16)]

        src_list = [kernel_nchw[i, 0, 0] for i in range(16)]
        if self.k_size * self.k_size == 16:
            self.tik_inst.vnchwconv(
                True, True, dst_list, src_list,
                self.k_size * self.k_size // 16, 0, 0
            )
        else:
            self.tik_inst.vnchwconv(
                True, True, dst_list, src_list,
                self.k_size * self.k_size // 16, 16, 1
            )

    def weight_slice(self, kernel_nhwc, kernel_slice):
        # split weight by step
        with self.tik_inst.for_range(0, self.stride) as c_index:
            with self.tik_inst.for_range(0, self.stride) as w_index:
                # filter rotation
                self.weight_slice_inner(kernel_nhwc, kernel_slice, c_index,
                                        w_index)

    def weight_slice_inner(self, kernel_nhwc, kernel_slice, c_index, w_index):
        with self.tik_inst.for_range(0, self.k_stride) as row:
            with self.tik_inst.for_range(0, self.k_stride) as col:
                self.tik_inst.data_move(
                    kernel_slice[c_index, w_index, row, col, 0],
                    kernel_nhwc[self.pad - (self.pad + c_index) // self.stride
                                * self.stride + c_index +
                                (self.k_stride - 1 - row) * self.stride,
                                self.pad - (self.pad + w_index) // self.stride
                                * self.stride + w_index +
                                (self.k_stride - 1 - col) * self.stride, 0],
                    0, 1, 1, 0, 0)

    def filter_diag(self, kernel_slice, kernel_c):
        # generate diagonal matrix
        self.diagonal = self.tik_inst.Tensor("float16", (16, 16),
                                             name="diagonal",
                                             scope=self.tik.scope_ubuf)
        one = self.tik_inst.Tensor("float16", (16,), name="one",
                                   scope=self.tik.scope_ubuf)
        self.tik_inst.vector_dup(16, one, 1, 1, 0, 0)
        self.tik_inst.vector_dup(128, self.diagonal, 0, 2, 1, 8)

        with self.tik_inst.for_range(0, 16) as i:
            self.diagonal[i, i] = one[0]

        with self.tik_inst.new_stmt_scope():
            kernel_angle = self.tik_inst.Tensor(
                "float16", (
                self.stride, self.stride, self.k_stride, self.k_stride, 16,
                16),
                name="kernel_slice", scope=self.tik.scope_ubuf
            )
            with self.tik_inst.for_range(0, self.stride) as i_index:
                self.filter_diag_inner(kernel_angle, kernel_slice, i_index)
            self.tik_inst.data_move(
                kernel_c, kernel_angle, 0, 1,
                self.stride * self.stride * self.k_stride * self.k_stride * 16,
                0, 0
            )

    def filter_diag_inner(self, kernel_angle, kernel_slice, i_index):
        with self.tik_inst.for_range(0, self.stride) as j_index:
            with self.tik_inst.for_range(0, self.k_stride) as k_index:
                with self.tik_inst.for_range(0, self.k_stride) as g_index:
                    self.tik_inst.vmul(
                        128,
                        kernel_angle[i_index, j_index, k_index, g_index, 0, 0],
                        kernel_slice[i_index, j_index, k_index, g_index, 0],
                        self.diagonal, 2, 1, 0, 1, 8, 0, 8
                    )

    def featuremap_pad(self, featuremap, actual_row):
        padd = self.tik_inst.Tensor(
            "float16", (1, int(math.ceil(self.width / 8.0)) * 8, 16),
            name="padd", scope=self.tik.scope_ubuf
        )
        self.tik_inst.vector_dup(128, padd, 0,
                                 int(math.ceil(self.width / 8.0)), 1, 8)
        with self.tik_inst.for_range(0,
                                     2 * self.k_stride - actual_row) as i:
            self.tik_inst.data_move(featuremap[0, 0, actual_row + i, 0, 0],
                                    padd, 0, 1, self.width, 0, 0)

    def featuremap_prepare(self, featuremap, batch, channel_1, n_h):
        handle_row = self.tik_inst.Scalar(
            dtype="int64", name="handle_row", init_value=self.k_stride
        )
        remain_row = self.tik_inst.Scalar(dtype="int64", name="remain_row")
        actual_row = self.tik_inst.Scalar(dtype="int64", name="actual_row")

        with self.tik_inst.if_scope(n_h == 0):
            remain_row.set_as(self.height)
            handle_row.set_as(2 * self.k_stride - self.p_size // self.stride)
            self.tik_inst.scalar_min(actual_row, handle_row, remain_row)
        with self.tik_inst.else_scope():
            remain_row.set_as(
                self.height + self.p_size // self.stride - self.k_stride * n_h)
            handle_row.set_as(2 * self.k_stride)
            self.tik_inst.scalar_min(actual_row, handle_row, remain_row)

        with self.tik_inst.if_scope(n_h == 0):
            self.tik_inst.data_move(
                featuremap, self.input_data[batch, channel_1, 0, 0, 0], 0, 1,
                actual_row * self.width, 0, 0
            )
            self.featuremap_pad(featuremap, actual_row)

        with self.tik_inst.else_scope():
            self.tik_inst.data_move(
                featuremap, self.input_data[
                    batch, channel_1, self.k_stride * n_h -
                    self.p_size // self.stride, 0, 0], 0, 1,
                actual_row * self.width, 0, 0)
            self.featuremap_pad(featuremap, actual_row)

    def load_l0a(self, l0a, featuremap, n_h, high, channel_z, width_z):
        scalar_1 = self.tik_inst.Scalar(dtype="int64", name="scalar_1")
        scalar_2 = self.tik_inst.Scalar(dtype="int64", name="scalar_2")
        scalar_1.set_as(self.p_size - width_z)
        scalar_2.set_as(self.p_size - channel_z)
        self.tik_inst.scalar_max(scalar_1, scalar_1, 0)
        self.tik_inst.scalar_max(scalar_2, scalar_2, 0)
        with self.tik_inst.if_scope(n_h == 0):
            with self.tik_inst.for_range(0, int(
                    math.ceil(self.width / 16.0))) as i:
                self.tik_inst.load3dv1(
                    l0a[16 * i, 0], featuremap, [self.p_size // self.stride,
                                                 self.p_size // self.stride,
                                                 self.p_size // self.stride,
                                                 self.p_size // self.stride],
                    2 * self.k_stride, self.width, 0, 0, 0,
                    scalar_1 // self.stride * (-1) + 16 * i,
                    scalar_2 // self.stride * (-1) + high,
                    1, 1, self.k_stride, self.k_stride, 1, 1, 1, 0,
                    self.k_stride_square
                )
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, int(
                    math.ceil(self.width / 16.0))) as i:
                self.tik_inst.load3dv1(
                    l0a[16 * i, 0], featuremap, [self.p_size // self.stride,
                                                 self.p_size // self.stride,
                                                 self.p_size // self.stride,
                                                 self.p_size // self.stride],
                    2 * self.k_stride, self.width, 0, 0, 0,
                    scalar_1 // self.stride * (-1) + 16 * i,
                    high + self.p_size // self.stride -
                    scalar_2 // self.stride, 1, 1, self.k_stride,
                    self.k_stride, 1, 1, 1, 0, self.k_stride_square
                )

    def load_l0b(self, l0b, kernel_c, channel_z, width_z):
        self.tik_inst.load2dv2(
            l0b, kernel_c[channel_z, width_z, 0, 0, 0, 0], 0,
            self.k_stride_square, 0, 1, 0, False
        )

    def calculation(self, l0c, l0a, l0b, channel_1, width_z):
        if (self.group == 1):
            self.calculation_if(l0c, width_z, l0a, l0b, channel_1)
        else:
            self.calculation_else(l0c, width_z, l0a, l0b)

    def calculation_if(self, l0c, width_z, l0a, l0b, channel_1):
        if (self.use_bias):
            self.tik_inst.mmad(
                l0c[width_z, 0, 0], l0a, l0b,
                int(math.ceil(self.width / 16.0)) * 16,
                16 * self.k_stride_square, 16, 1
            )
        else:
            with self.tik_inst.if_scope(channel_1 == 0):
                self.tik_inst.mmad(
                    l0c[width_z, 0, 0], l0a, l0b,
                    int(math.ceil(self.width / 16.0)) * 16,
                    16 * self.k_stride_square, 16, 0
                )
            with self.tik_inst.else_scope():
                self.tik_inst.mmad(
                    l0c[width_z, 0, 0], l0a, l0b,
                    int(math.ceil(self.width / 16.0)) * 16,
                    16 * self.k_stride_square, 16, 1
                )

    def calculation_else(self, l0c, width_z, l0a, l0b):
        if (self.use_bias):
            self.tik_inst.mmad(
                l0c[width_z, 0, 0], l0a, l0b,
                int(math.ceil(self.width / 16.0)) * 16,
                16 * self.k_stride_square, 16, 1
            )
        else:
            self.tik_inst.mmad(
                l0c[width_z, 0, 0], l0a, l0b,
                int(math.ceil(self.width / 16.0)) * 16,
                16 * self.k_stride_square, 16, 0
            )

    def result_store(self, l0c, res_src, res_dst, batch, out_c1, channel_1,
                     n_h, high, channel_z):
        self.tik_inst.data_move(
            res_src, l0c, 0, 1,
            int(math.ceil(self.width / 16.0)) * self.stride, 0, 0
        )
        with self.tik_inst.if_scope(channel_1 == self.channel_1 - 1):
            with self.tik_inst.for_range(0, self.stride) as j:
                self.tik_inst.data_move(
                    res_dst[0, j, 0], res_src[j, 0, 0], 0, self.width, 1, 0,
                    self.stride - 1
                )
            self.tik_inst.data_move(
                self.output_data[batch, out_c1, self.stride * (
                            n_h * self.k_stride + high) + channel_z, 0, 0],
                res_dst, 0, 1, self.output_w, 0, 0
            )

        with self.tik_inst.else_scope():
            self.tik_inst.data_move(
                self.output_data[batch, out_c1, self.stride * (
                            n_h * self.k_stride + high) + channel_z, 0, 0],
                res_src, 0, self.stride, self.width,
                int(math.ceil(self.width / 16.0)) * 16 - self.width, 0
            )

    def result_carry(self, l0c, res_src, res_dst, batch, channel_1, n_h, high,
                     channel_z):
        self.tik_inst.data_move(
            res_src, l0c, 0, 1,
            int(math.ceil(self.width / 16.0)) * self.stride, 0, 0
        )
        with self.tik_inst.for_range(0, self.stride) as j:
            self.tik_inst.data_move(
                res_dst[0, j, 0], res_src[j, 0, 0], 0, self.width, 1, 0,
                self.stride - 1
            )
        self.tik_inst.data_move(
            self.output_data[batch, channel_1, self.stride * (
                        n_h * self.k_stride + high) + channel_z, 0, 0],
            res_dst, 0, 1, self.output_w, 0, 0
        )

    def compute_each_core_gen(self, batch, start, end):
        with self.tik_inst.for_range(start, end) as out_c1:
            with self.tik_inst.for_range(0, self.channel_1) as channel_1:
                l0c = self.tik_inst.Tensor(
                    "float16",
                    (self.stride, int(math.ceil(self.width / 16.0)) * 16, 16),
                    name="l0c", scope=self.tik.scope_cc
                )
                kernel_c = self.tik_inst.Tensor(
                    "float16", (
                    self.stride, self.stride, self.k_stride, self.k_stride, 16,
                    16),
                    name="kernel_c", scope=self.tik.scope_cbuf
                )
                # calculate unprocessed in_channel and out_channel
                remain_out_c = self.tik_inst.Scalar(dtype="int64",
                                                    name="remain_out_c")
                remain_in_c = self.tik_inst.Scalar(dtype="int64",
                                                   name="remain_in_c")
                remain_out_c.set_as(self.width_k - 16 * out_c1)
                remain_in_c.set_as(self.channel - 16 * channel_1)
                self.tik_inst.scalar_min(remain_out_c, remain_out_c, 16)
                self.tik_inst.scalar_min(remain_in_c, remain_in_c, 16)
                self.compute_each_core_gen_inner(remain_out_c, channel_1,
                                                 out_c1, remain_in_c, kernel_c,
                                                 batch, l0c)

    def compute_each_core_gen_inner(self, remain_out_c, channel_1, out_c1,
                                    remain_in_c, kernel_c, batch, l0c):
        with self.tik_inst.for_range(0, remain_out_c) as out_c:
            # load a kernel
            kernel_nhwc = self.tik_inst.Tensor(
                "float16", (self.k_size, self.k_size, 16), name="kernel_nhwc",
                scope=self.tik.scope_ubuf
            )
            kernel_nchw = self.tik_inst.Tensor(
                "float16", (16, self.k_size, self.k_size), name="kernel_nchw",
                scope=self.tik.scope_ubuf
            )
            # carry 16 in_channels every time or all of unprocessed
            # in_channels the last time
            self.tik_inst.data_move(
                kernel_nchw,
                self.weight[16 * channel_1, 16 * out_c1 + out_c, 0, 0], 0,
                remain_in_c,
                self.k_size * self.k_size // 16,
                (self.num_output - 1) * self.k_size * self.k_size // 16, 0
            )
            # transpose weight
            self.weight_transpose(kernel_nchw, kernel_nhwc)

            # filter segmentation
            kernel_slice = self.tik_inst.Tensor(
                "float16",
                (self.stride, self.stride, self.k_stride, self.k_stride, 16),
                name="kernel_slice", scope=self.tik.scope_ubuf
            )

            # channel separation, the filter will be devided into
            # self.stride^2 parts
            self.weight_slice(kernel_nhwc, kernel_slice)

            self.tik_inst.data_move(
                kernel_c[0, 0, 0, 0, out_c, 0], kernel_slice, 0,
                self.stride * self.stride * self.k_stride * self.k_stride, 1,
                0, 15
            )
        self.compute_each_core_gen_inner_inner(batch, channel_1, out_c1, l0c,
                                               kernel_c)

    def compute_each_core_gen_inner_inner(self, batch, channel_1, out_c1, l0c,
                                          kernel_c):
        with self.tik_inst.for_range(0, int(
                math.ceil((self.height / (self.k_stride * 1.0))))) as n_h:
            featuremap = self.tik_inst.Tensor(
                "float16", (1, 1, 2 * self.k_stride, self.width, 16),
                name="featuremap", scope=self.tik.scope_cbuf
            )

            # load input to L1
            self.featuremap_prepare(featuremap, batch, channel_1, n_h)

            cal_row = self.tik_inst.Scalar(dtype="int64", name="cal_row")
            com_row = self.tik_inst.Scalar(dtype="int64", name="com_row")
            mov_row = self.tik_inst.Scalar(dtype="int64", name="mov_row")
            com_row.set_as(self.k_stride)
            mov_row.set_as(self.height - n_h * self.k_stride)
            self.tik_inst.scalar_min(cal_row, com_row, mov_row)
            with self.tik_inst.for_range(0, cal_row) as high:
                self.compute_each_core_gen_inner_inner_inner(channel_1, out_c1,
                                                             l0c, batch, n_h,
                                                             high, featuremap,
                                                             kernel_c)

    def compute_each_core_gen_inner_inner_inner(self, channel_1, out_c1, l0c,
                                                batch, n_h, high, featuremap,
                                                kernel_c):
        with self.tik_inst.for_range(0, self.stride) as channel_z:
            if (self.use_bias):
                # load bias
                self.compute_each_core_gen_inner_inner_if(channel_1, out_c1,
                                                          l0c, batch, n_h,
                                                          high, channel_z)
            else:
                self.compute_each_core_gen_inner_inner_else(channel_1, batch,
                                                            out_c1, n_h, high,
                                                            channel_z, l0c)

            with self.tik_inst.for_range(0, self.stride) as width_z:
                l0a = self.tik_inst.Tensor(
                    "float16", (int(math.ceil(self.width / 16.0)) * 16,
                                16 * self.k_stride_square),
                    name="l0a", scope=self.tik.scope_ca
                )
                # load input to l0a
                self.load_l0a(l0a, featuremap, n_h, high, channel_z, width_z)
                l0b = self.tik_inst.Tensor(
                    "float16", (16 * self.k_stride_square, 16),
                    name="l0b", scope=self.tik.scope_cb
                )
                # load weight to l0b
                self.load_l0b(l0b, kernel_c, channel_z, width_z)

                # calculation
                self.calculation(l0c, l0a, l0b, channel_1, width_z)

            res_src = self.tik_inst.Tensor(
                "float16",
                (self.stride, int(math.ceil(self.width / 16.0)) * 16, 16),
                name="res_src", scope=self.tik.scope_ubuf
            )
            res_dst = self.tik_inst.Tensor(
                "float16", (int(math.ceil(self.width / 16.0)) * 16,
                            self.stride, 16),
                name="res_dst", scope=self.tik.scope_ubuf)
            # store results
            self.result_store(
                l0c, res_src, res_dst, batch, out_c1, channel_1, n_h, high,
                channel_z
            )

    def compute_each_core_gen_inner_inner_if(self, channel_1, out_c1, l0c,
                                             batch, n_h, high, channel_z):
        with self.tik_inst.if_scope(channel_1 == 0):
            bias_ub = self.tik_inst.Tensor("float16", (16,), name="bias_ub",
                                           scope=self.tik.scope_ubuf)
            self.tik_inst.data_move(
                bias_ub, self.bias[16 * out_c1],
                0, 1, 1, 0, 0
            )
            with self.tik_inst.for_range(0, self.stride) as index_1:
                with self.tik_inst.for_range(
                        0, int(math.ceil(self.width / 16.0))
                ) as fracal:
                    self.tik_inst.broadcast_ub_to_l0c(
                        l0c[index_1, 16 * fracal, 0], bias_ub,
                        1, 1, 0, 0
                    )
        with self.tik_inst.else_scope():
            # load the previous calculation results to l0c
            temp = self.tik_inst.Tensor(
                "float16",
                (self.stride, int(math.ceil(self.width / 16.0)) * 16, 16),
                name="temp", scope=self.tik.scope_ubuf
            )
            self.tik_inst.data_move(
                temp, self.output_data[
                    batch, out_c1,
                    (n_h * self.k_stride + high) * self.stride + channel_z,
                    0, 0],
                0, self.stride, self.width, 0,
                int(math.ceil(self.width / 16.0)) * 16 - self.width
            )

            self.tik_inst.data_move(
                l0c, temp, 0, 1,
                int(math.ceil(self.width / 16.0)) * self.stride,
                0, 0
            )

    def compute_each_core_gen_inner_inner_else(self, channel_1, batch, out_c1,
                                               n_h, high, channel_z, l0c):
        with self.tik_inst.if_scope(channel_1 > 0):
            # load the previous calculation results to l0c
            temp = self.tik_inst.Tensor(
                "float16",
                (self.stride, int(math.ceil(self.width / 16.0)) * 16, 16),
                name="temp", scope=self.tik.scope_ubuf
            )
            self.tik_inst.data_move(
                temp, self.output_data[batch, out_c1, (
                            n_h * self.k_stride + high) * self.stride +
                                       channel_z, 0, 0],
                0, self.stride, self.width, 0,
                int(math.ceil(self.width / 16.0)) * 16 - self.width
            )
            self.tik_inst.data_move(
                l0c, temp, 0, 1,
                int(math.ceil(self.width / 16.0)) * self.stride, 0, 0
            )
        with self.tik_inst.else_scope():
            pass

    def compute_each_core_gro(self, batch, start, end):
        with self.tik_inst.for_range(start, end) as channel_1:
            # NCHW --> NHWC
            kernel_nhwc = self.tik_inst.Tensor(
                "float16", (self.k_size, self.k_size, 16), name="kernel_nhwc",
                scope=self.tik.scope_ubuf
            )
            kernel_nchw = self.tik_inst.Tensor(
                "float16", (16, self.k_size, self.k_size), name="kernel_nchw",
                scope=self.tik.scope_ubuf
            )

            # calculate unprocessed in_channel and out_channel
            remain_in_c = self.tik_inst.Scalar(dtype="int64",
                                               name="remain_in_c")
            remain_in_c.set_as(self.channel - 16 * channel_1)
            # carry 16 in_channels every time or all of unprocessed
            # in_channels the last time
            self.tik_inst.scalar_min(remain_in_c, remain_in_c, 16)
            self.tik_inst.data_move(
                kernel_nchw, self.weight[16 * channel_1, 0, 0, 0], 0, 1,
                self.k_size * self.k_size * remain_in_c // 16, 0, 0
            )
            # transpose weight
            self.weight_transpose(kernel_nchw, kernel_nhwc)

            # filter segmentation
            kernel_slice = self.tik_inst.Tensor(
                "float16",
                (self.stride, self.stride, self.k_stride, self.k_stride, 16),
                name="kernel_slice", scope=self.tik.scope_ubuf
            )
            # channel separation, the filter will be devided into
            # self.stride^2 parts
            self.weight_slice(kernel_nhwc, kernel_slice)

            # channel expansion
            kernel_c = self.tik_inst.Tensor(
                "float16", (
                self.stride, self.stride, self.k_stride, self.k_stride, 16,
                16),
                name="kernel_c", scope=self.tik.scope_cbuf
            )
            self.filter_diag(kernel_slice, kernel_c)
            self.compute_each_core_gro_inner(batch, channel_1, kernel_c)

    def compute_each_core_gro_inner(self, batch, channel_1, kernel_c):
        with self.tik_inst.for_range(0, int(
                math.ceil((self.height / (self.k_stride * 1.0))))) as n_h:
            featuremap_gro = self.tik_inst.Tensor(
                "float16", (1, 1, 2 * self.k_stride, self.width, 16),
                name="featuremap", scope=self.tik.scope_cbuf
            )
            # load input to L1
            self.featuremap_prepare(featuremap_gro, batch, channel_1, n_h)

            cal_row_gro = self.tik_inst.Scalar(dtype="int64", name="cal_row")
            com_row_gro = self.tik_inst.Scalar(dtype="int64", name="com_row")
            mov_row_gro = self.tik_inst.Scalar(dtype="int64", name="mov_row")
            com_row_gro.set_as(self.k_stride)
            mov_row_gro.set_as(self.height - n_h * self.k_stride)
            self.tik_inst.scalar_min(cal_row_gro, com_row_gro, mov_row_gro)
            with self.tik_inst.for_range(0, cal_row_gro) as high:
                l0c = self.tik_inst.Tensor(
                    "float16",
                    (self.stride, int(math.ceil(self.width / 16.0)) * 16, 16),
                    name="l0c", scope=self.tik.scope_cc
                )
                self.compute_each_core_gro_inner_inner(channel_1, l0c,
                                                       featuremap_gro, n_h,
                                                       high, kernel_c, batch)

    def compute_each_core_gro_inner_inner(self, channel_1, l0c, featuremap,
                                          n_h, high, kernel_c, batch):
        with self.tik_inst.for_range(0, self.stride) as channel_z:
            if (self.use_bias):
                self.compute_each_core_gro_inner_inner_if(channel_1, l0c)
            with self.tik_inst.for_range(0, self.stride) as width_z:
                l0a = self.tik_inst.Tensor(
                    "float16", (int(math.ceil(self.width / 16.0)) * 16,
                                16 * self.k_stride_square),
                    name="l0a", scope=self.tik.scope_ca
                )
                # load input to l0a
                self.load_l0a(l0a, featuremap, n_h, high, channel_z, width_z)

                l0b = self.tik_inst.Tensor(
                    "float16", (16 * self.k_stride_square, 16), name="l0b",
                    scope=self.tik.scope_cb
                )
                # load weight to l0b
                self.load_l0b(l0b, kernel_c, channel_z, width_z)
                # calculation
                self.calculation(l0c, l0a, l0b, channel_1, width_z)

            res_src = self.tik_inst.Tensor(
                "float16",
                (self.stride, int(math.ceil(self.width / 16.0)) * 16, 16),
                name="res_src", scope=self.tik.scope_ubuf
            )
            res_dst = self.tik_inst.Tensor(
                "float16",
                (int(math.ceil(self.width / 16.0)) * 16, self.stride, 16),
                name="res_dst", scope=self.tik.scope_ubuf)
            # store results
            self.result_carry(l0c, res_src, res_dst, batch, channel_1, n_h,
                              high, channel_z)

    def compute_each_core_gro_inner_inner_if(self, channel_1, l0c):
        bias_ub = self.tik_inst.Tensor("float16", (16,), name="bias_ub",
                                       scope=self.tik.scope_ubuf)
        self.tik_inst.data_move(bias_ub, self.bias[16 * channel_1], 0, 1, 1, 0,
                                0)
        with self.tik_inst.for_range(0, self.stride) as index_1:
            with self.tik_inst.for_range(0, int(
                    math.ceil(self.width / 16.0))) as fracal:
                self.tik_inst.broadcast_ub_to_l0c(l0c[index_1, 16 * fracal, 0],
                                                  bias_ub, 1, 1, 0, 0)


def deconvolution_v1(input_data, weight, bias, output_data, strides, pads,
                     dilations, groups, data_format, offset_x,
                     kernel_name="deconvolution_v1"):
    container = get_aicore_container(("Ascend610",), c3x_support_list=())
    obj = DeconvolutionV1(container, input_data, weight, bias, output_data,
                          strides, pads, dilations, groups, data_format,
                          offset_x, kernel_name)

    obj.tilling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    switch = {
        1: obj.model_compute_gen,
        2: obj.model_compute_gro
    }

    switch[obj.mode]()
