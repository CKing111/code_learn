from uti import interface_check
from version import get_version


class BroadcastSub:

    def __init__(self, minuend, subtrahend, sub_result, kernel_name):
        """
        Args:
            minuend:
                Dict; Rank(ori_shape) == 4; format: 5D; dtype: "flaot16"
            subtrahend:
                Dict; Rank(ori_shape) == 4; format: 5D; dtype: "flaot16"
            sub_result:
                Dict; Rank(ori_shape) == 4; format: 5D; dtype: "flaot16"
            kernel_name:
        """
        interface_check.check_kernelname(kernel_name)
        self._check_params(minuend, subtrahend, sub_result)
        self._check_params_shape(minuend, subtrahend, sub_result)
        self.cont = get_version.get_aicore_container(("Ascend610",))
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.kernel_name = kernel_name
        self.ub_size = self.cont.const_ub_max_byte
        self.ai_core_use = self.cont.const_aicore_num
        self.repeat_time_max = self.cont.const_vector_proc_max_rpt
        self.repeat_block_num = (self.cont.const_vector_proc_byte //
                                 self.cont.const_block_byte)

        self.data_type = minuend.get("dtype")
        self.data_size, self.block_data_num, self.repeat_data_num = \
            self._get_type_const(self.data_type)

        minuend_shape = minuend.get("shape")
        batch_num, channel_1, height, width, channel_0 = minuend_shape
        self.batch_num, self.channel_1 = batch_num, channel_1
        self.data_num = height * width
        self.channel_0 = channel_0
        minuend_shape = \
            (self.batch_num, self.channel_1, self.data_num, self.channel_0)
        subtrahend_shape = (1, self.channel_1, 1, self.channel_0)
        result_shape = minuend_shape
        self.minuend = self.tik_inst.Tensor(
            minuend.get("dtype"), minuend_shape, self.tik.scope_gm, "minuend")
        self.subtrahend = self.tik_inst.Tensor(
            subtrahend.get("dtype"), subtrahend_shape,
            self.tik.scope_gm, "subtrahend")
        self.sub_result = self.tik_inst.Tensor(
            sub_result.get("dtype"), result_shape,
            self.tik.scope_gm, "sub_result")

    def _get_type_const(self, data_type):
        data_size = self.cont.const_dtype_byte.get(data_type)
        block_data_num = self.cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = self.cont.get_vec_proc_num_per_cmd(data_type)
        return data_size, block_data_num, repeat_data_num

    def _check_params(self, minuend, subtrahend, sub_result):
        interface_check.check_param(minuend, [5], ["float16"], ["NC1HWC0"])
        interface_check.check_param(subtrahend, [5], ["float16"], ["NC1HWC0"])
        interface_check.check_param(sub_result, [5], ["float16"], ["NC1HWC0"])

    def _check_params_shape(self, minuend, subtrahend, sub_result):
        minuend_shape = minuend.get("ori_shape")
        if (not isinstance(minuend_shape, (tuple, list))
                or len(minuend_shape) != 4):
            raise RuntimeError("minuend shape is not supported")
        dim_0, dim_1, dim_2, dim_3, dim_4 = minuend.get("shape")
        if (dim_0 * dim_1 * dim_2 * dim_3 * dim_4 > 10 ** 8 or
                dim_0 * dim_1 > 65535):
            raise RuntimeError("input_shape is too larger")
        minuend_format = minuend.get("ori_format")
        if minuend_format == "NCHW":
            channel = minuend_shape[1]
        elif minuend_format == "NHWC":
            channel = minuend_shape[3]
        else:
            raise RuntimeError("minuend format is not supported")
        subtrahend_format = subtrahend.get("ori_format")
        if subtrahend_format == "NCHW":
            subtrahend_shape_ori = (1, channel, 1, 1)
        elif subtrahend_format == "NHWC":
            subtrahend_shape_ori = (1, 1, 1, channel)
        else:
            raise RuntimeError("subtrahend format is not supported")
        subtrahend_shape = subtrahend.get("ori_shape")
        sub_result_shape = sub_result.get("ori_shape")
        if subtrahend_shape != subtrahend_shape_ori:
            raise RuntimeError("subtrahend shape is not supported")
        if sub_result_shape != minuend_shape:
            raise RuntimeError("sub_result shape is not supported")

    def _ceil_div(self, dividend, divisor):
        result = (dividend + divisor - 1) // divisor
        return result

    def _get_loop_info(self, all_data_num, each_loop_num):
        loop_times = self._ceil_div(all_data_num, each_loop_num)
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return loop_times, last_loop_num

    def _get_format_num(self, input_num, format_num, ceil=True):
        if ceil:
            result = self._ceil_div(input_num, format_num) * format_num
        else:
            result = input_num // format_num * format_num
        return result

    def mode_compute(self):
        batch_num = self.batch_num * self.channel_1
        if batch_num < self.ai_core_use:
            self._mode1_compute()
        else:
            self._mode2_compute()
        self.tik_inst.BuildCCE(inputs=[self.minuend, self.subtrahend],
                               outputs=[self.sub_result],
                               kernel_name=self.kernel_name)

    def _mode1_compute(self):
        batch_num = self.batch_num * self.channel_1
        each_batch_core_num = self.ai_core_use // batch_num
        each_core_data_num = self._ceil_div(self.data_num, each_batch_core_num)
        each_core_data_num = self._get_format_num(each_core_data_num,
                                                  self.repeat_block_num)
        each_batch_core_num, last_core_data_num = self._get_loop_info(
            self.data_num, each_core_data_num)
        self.ai_core_use = each_batch_core_num * batch_num

        with self.tik_inst.for_range(0, self.ai_core_use,
                                     block_num=self.ai_core_use) as core_index:
            batch_index = core_index // each_batch_core_num
            batch_core_index = core_index % each_batch_core_num
            data_index_start = batch_core_index * each_core_data_num
            with self.tik_inst.if_scope(
                    batch_core_index != each_batch_core_num - 1):
                self._mode1_compute_each_core(batch_index, data_index_start,
                                              each_core_data_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_core(batch_index, data_index_start,
                                              last_core_data_num)

    def _mode2_compute(self):
        batch_num = self.batch_num * self.channel_1
        with self.tik_inst.for_range(0, batch_num,
                                     block_num=batch_num) as batch_index:
            self._mode2_compute_each_core(batch_index)

    def _get_each_loop_data_num(self, data_num):
        align_num = self.repeat_block_num
        if data_num <= align_num:
            thread_num = 1
        else:
            thread_num = 2
        ub_size = self.ub_size // thread_num - self.channel_0 * self.data_size
        each_data_size = self.channel_0 * self.data_size
        each_loop_data_num = ub_size // each_data_size
        each_loop_data_num = self._get_format_num(
            each_loop_data_num, align_num, False)
        each_loop_data_num_max = data_num // thread_num
        each_loop_data_num_max = self._get_format_num(
            each_loop_data_num_max, align_num)
        each_loop_data_num = min(each_loop_data_num, each_loop_data_num_max,
                                 self.repeat_time_max * align_num)
        return each_loop_data_num

    def _mode1_compute_each_core(self, batch_index, data_index_start,
                                 data_num):
        each_loop_data_num = self._get_each_loop_data_num(data_num)
        loop_num, last_loop_data_num = self._get_loop_info(
            data_num, each_loop_data_num)
        with self.tik_inst.for_range(
                0, loop_num, thread_num=min(2, loop_num)) as loop_index:
            data_index = each_loop_data_num * loop_index + data_index_start
            with self.tik_inst.if_scope(loop_index != loop_num - 1):
                self._mode_compute_each_loop(batch_index, data_index,
                                             each_loop_data_num)
            with self.tik_inst.else_scope():
                self._mode_compute_each_loop(batch_index, data_index,
                                             last_loop_data_num)

    def _mode2_compute_each_core(self, batch_index):
        each_loop_data_num = self._get_each_loop_data_num(self.data_num)
        loop_num, last_loop_data_num = self._get_loop_info(
            self.data_num, each_loop_data_num)
        with self.tik_inst.for_range(
                0, loop_num, thread_num=min(2, loop_num)) as loop_index:
            data_index = each_loop_data_num * loop_index
            with self.tik_inst.if_scope(loop_index != loop_num - 1):
                self._mode_compute_each_loop(
                    batch_index, data_index, each_loop_data_num)
            with self.tik_inst.else_scope():
                self._mode_compute_each_loop(
                    batch_index, data_index, last_loop_data_num)

    def _mode_compute_each_loop(self, batch_index, data_index, data_num):
        channel_1_index = batch_index % self.channel_1
        batch_index = batch_index // self.channel_1
        subtrahend_ub = self.tik_inst.Tensor(
            self.data_type, (self.channel_0,),
            self.tik.scope_ubuf, "subtrahend_ub")
        subtrahend_block_num = self._ceil_div(
            self.channel_0, self.block_data_num)
        self.tik_inst.data_move(
            subtrahend_ub, self.subtrahend[0, channel_1_index, 0, 0],
            0, 1, subtrahend_block_num, 0, 0)
        data_num_format = self._get_format_num(data_num, self.repeat_block_num)
        repeat_num = data_num_format // self.repeat_block_num
        minuend_ub = self.tik_inst.Tensor(
            self.data_type, (data_num_format, self.channel_0),
            self.tik.scope_ubuf, "minuend_ub")
        minuend_block_num = self._ceil_div(
            data_num * self.channel_0, self.block_data_num)
        self.tik_inst.data_move(
            minuend_ub,
            self.minuend[batch_index, channel_1_index, data_index, 0],
            0, 1, minuend_block_num, 0, 0)
        mask = self.repeat_data_num
        self.tik_inst.vsub(
            mask, minuend_ub, minuend_ub, subtrahend_ub,
            repeat_num, 1, 1, 0, 8, 8, 0)
        self.tik_inst.data_move(
            self.sub_result[batch_index, channel_1_index, data_index, 0],
            minuend_ub, 0, 1, minuend_block_num, 0, 0)

    def tik_output_debug(self):
        return self.tik_inst


def broadcast_sub(minuend, subtrahend, sub_result,
                  kernel_name="BroadcastSub", test=False):
    obj = BroadcastSub(minuend, subtrahend, sub_result, kernel_name)
    obj.mode_compute()
    if not test:
        return 0

    obj.tik_output_debug()
    return 0
