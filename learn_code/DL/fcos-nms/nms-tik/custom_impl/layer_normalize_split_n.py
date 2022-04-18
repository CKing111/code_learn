from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from .layer_normalize_split_d import LayerNormalizeBase


class LayerNormalizeSplitN(LayerNormalizeBase):
    """
    layer normalize:
        split mode: spilt n dim, each loop count multi batch
    """

    def mode_compute(self):
        # mode1: data_num: 8
        if self.data_num == self.fp32_block_data_num:
            self._mode1_compute()
        # mode2: data_num bigger than 8
        else:
            self._mode2_compute()

    def _mode1_compute(self):
        """
        start mode1 compute, d equal 8
        """
        each_core_batch_num = self._ceil_div(self.batch_num, self.ai_core_use)
        last_core_batch_num, self.ai_core_use = self._get_loop_info(
            self.batch_num, each_core_batch_num)
        with self.tik_inst.for_range(0, self.ai_core_use,
                                     block_num=self.ai_core_use) as core_index:
            batch_index = each_core_batch_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                self._mode1_compute_each_core(batch_index, each_core_batch_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_core(batch_index, last_core_batch_num)
        self.tik_inst.BuildCCE(
            inputs=[self.input_tensor, self.weight, self.bias],
            outputs=[self.output_tensor],
            kernel_name=self.kernel_name)

    def _mode1_compute_each_core(self, batch_index_start, batch_num):
        if batch_num == 1:
            thread_num = 1
            loop_times = 1
            each_loop_batch_num = 1
            last_loop_batch_num = 1
        else:
            thread_num = 2
            ub_size = self.cont.const_ub_max_byte // thread_num
            max_loop_batch_num_format = self._mode_1_max_batch_num(ub_size)
            each_thread_batch_num = self._ceil_div(batch_num, thread_num)
            each_loop_batch_num = min(max_loop_batch_num_format,
                                      each_thread_batch_num)
            last_loop_batch_num, loop_times = self._get_loop_info(
                batch_num, each_loop_batch_num)
        with self.tik_inst.for_range(0, loop_times,
                                     thread_num=thread_num) as loop_index:
            batch_index = batch_index_start + each_loop_batch_num * loop_index
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._mode1_compute_each_loop(batch_index, each_loop_batch_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_loop(batch_index, last_loop_batch_num)

    def _mode_1_max_batch_num(self, ub_size):
        """
        count each loop batch num
        Args:
            ub_size: each loop ub size
        """
        tensor_num = 3
        repeat_max_num = self.cont.const_vector_proc_max_rpt
        each_loop_max_batch_num = repeat_max_num * self.repeat_block_num
        expand_size = tensor_num * self.cont.const_block_byte
        ub_size_remain = ub_size - expand_size
        each_batch_size = self.fp32_size * 3 + 2 * self.cont.const_block_byte
        max_loop_batch_num = ub_size_remain // each_batch_size
        max_loop_batch_num_format = self._get_format_num(max_loop_batch_num,
                                                         self.repeat_block_num,
                                                         False)
        return min(max_loop_batch_num_format, each_loop_max_batch_num)

    def _mode1_compute_each_loop(self, batch_index_start, batch_num):
        # each loop data info
        repeat_num = self._ceil_div(batch_num, self.repeat_block_num)
        batch_num_format = self._get_format_num(batch_num,
                                                self.repeat_block_num)
        batch_num_format_block = self._get_format_num(batch_num,
                                                      self.fp32_block_data_num)
        each_batch_data_num = self.fp32_block_data_num
        (buf_mean_all, batch_ne_mean_ub, batch_mean_square_ub,
         batch_rec_std_ub) = \
            self._init_mean_tensor(batch_num_format_block)
        buf_data_all, input_data_ub, input_data_square_ub, fp16_data_tensor = \
            self._init_data_tensor(batch_num_format, each_batch_data_num)
        # data move in
        self.data_move(input_data_ub, self.input_tensor[batch_index_start, 0],
                       0, 1, batch_num, 0, 0)
        mask_count = self.fp32_repeat_data_num
        self.tik_inst.vmul(
            mask_count, input_data_square_ub, input_data_ub, input_data_ub,
            repeat_num, 1, 1, 1, 8, 8, 8)
        # count sum(x) sum(x^2)
        self.tik_inst.vcgadd(
            mask_count, batch_ne_mean_ub, input_data_ub, repeat_num, 1, 1, 8)
        self.tik_inst.vcgadd(
            mask_count, batch_mean_square_ub, input_data_square_ub, repeat_num,
            1, 1, 8)
        # count rec_std, ne_mean
        self._count_rec_std_ne_mean(buf_mean_all)
        # count (data add ne_mean) mul rec_std
        self._stand_data(input_data_ub, batch_ne_mean_ub, batch_rec_std_ub,
                         batch_num, each_batch_data_num, 1)
        # count data mul weight add bias
        if self.elementwise:
            self.data_move(input_data_square_ub, self.weight, 0, 1, 1, 0, 0)
            self.tik_inst.vmul(
                mask_count, input_data_ub, input_data_ub, input_data_square_ub,
                repeat_num, 1, 1, 0, 8, 8, 0)
            self.data_move(input_data_square_ub, self.bias, 0, 1, 1, 0, 0)
            self.tik_inst.vadd(
                mask_count, input_data_ub, input_data_ub, input_data_square_ub,
                repeat_num, 1, 1, 0, 8, 8, 0)
        self.data_move(self.output_tensor[batch_index_start, 0],
                       input_data_ub, 0, 1, batch_num, 0, 0)

    def _init_data_tensor(self, batch_num, data_num):
        tensor_type = self.fp32_type
        input_data_ub = self.tik_inst.Tensor(
            tensor_type, (batch_num, data_num), self.tik.scope_ubuf,
            "input_data_ub")
        input_data_square_ub = self.tik_inst.Tensor(
            tensor_type, (batch_num, data_num), self.tik.scope_ubuf,
            "input_data_square_ub")
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        buf_data_all = {
            "input_data_ub": AVecBuf(
                input_data_ub, batch_num * data_num, 0, self.cont, False,
                num_per_cmd),
            "input_data_square_ub": AVecBuf(
                input_data_square_ub, batch_num * data_num, 0, self.cont,
                False, num_per_cmd)}
        fp16_data_tensor = None
        if self.if_fp16:
            fp16_data_type = "float16"
            fp16_data_tensor = self.tik_inst.Tensor(
                fp16_data_type, (batch_num, data_num), self.tik.scope_ubuf,
                "fp16_data_tensor")
            num_per_cmd_fp16 = self.cont.get_vec_proc_num_per_cmd(
                fp16_data_type)
            buf_data_all["fp16_data_tensor"] = AVecBuf(
                fp16_data_tensor, batch_num * data_num, 0, self.cont, False,
                num_per_cmd_fp16)
            buf_data_all["fp16_data_tensor_vconv"] = AVecBuf(
                fp16_data_tensor, batch_num * data_num, 0, self.cont, False,
                num_per_cmd)
            cmd_dup_input_tensor = [
                VecGCmd(cmd_name="vector_dup", dst_name="fp16_data_tensor",
                        scalar=0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_data_all,
                                        cmd_dup_input_tensor,
                                        "fp16_data_tensor")
        cmd_dup_input_tensor = [
            VecGCmd(cmd_name="vector_dup", dst_name="input_data_ub", scalar=0)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_data_all,
                                    cmd_dup_input_tensor, "input_data_ub")
        return (buf_data_all, input_data_ub,
                input_data_square_ub, fp16_data_tensor)

    def _count_rec_std_ne_mean(self, buf_mean_all):
        # count rec_std and ne_mean
        cmd_count_var_data = [
            VecGCmd("vmuls", "batch_ne_mean_ub", "batch_ne_mean_ub",
                    scalar=1.0 / self.data_num),  # E(x)
            VecGCmd("vmuls", "batch_mean_square_ub", "batch_mean_square_ub",
                    scalar=1.0 / self.data_num), ]  # E(x^2)
        cmd_count_var_data.extend(self._get_rec_std_ne_mean_cmd())
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_mean_all,
                                    cmd_count_var_data, "batch_ne_mean_ub")

    def _stand_data(self, input_data_ub, batch_ne_mean_ub, batch_rec_std_ub,
                    batch_num, mask, repeat_num):
        scalar_type = self.fp32_type
        ne_mean_scalar = self.tik_inst.Scalar(scalar_type)
        rec_std_scalar = self.tik_inst.Scalar(scalar_type)
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            ne_mean_scalar.set_as(batch_ne_mean_ub[batch_index])
            rec_std_scalar.set_as(batch_rec_std_ub[batch_index])
            self.tik_inst.vadds(mask, input_data_ub[batch_index, 0],
                                input_data_ub[batch_index, 0],
                                ne_mean_scalar, repeat_num, 1, 1, 8, 8)
            self.tik_inst.vmuls(mask, input_data_ub[batch_index, 0],
                                input_data_ub[batch_index, 0],
                                rec_std_scalar, repeat_num, 1, 1, 8, 8)

    def _mode2_compute(self):
        """
        start mode compute, d smaller
        """
        each_core_batch_num = self._ceil_div(self.batch_num, self.ai_core_use)
        last_core_batch_num, self.ai_core_use = self._get_loop_info(
            self.batch_num, each_core_batch_num)
        with self.tik_inst.for_range(0, self.ai_core_use,
                                     block_num=self.ai_core_use) as core_index:
            batch_index = each_core_batch_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                self._mode2_compute_each_core(batch_index, each_core_batch_num)
            with self.tik_inst.else_scope():
                self._mode2_compute_each_core(batch_index, last_core_batch_num)
        self.tik_inst.BuildCCE(
            inputs=[self.input_tensor, self.weight, self.bias],
            outputs=[self.output_tensor],
            kernel_name=self.kernel_name)

    def _mode2_compute_each_core(self, batch_index_start, batch_num):
        if batch_num == 1:
            thread_num = 1
            loop_times = 1
            each_loop_batch_num = 1
            last_loop_batch_num = 1
        else:
            thread_num = 2
            each_loop_max_batch_num = self.cont.const_vector_proc_max_rpt
            ub_size = self.cont.const_ub_max_byte // thread_num
            each_batch_data_num_format = self._get_format_num(
                self.data_num, self.fp32_repeat_data_num)
            max_loop_data_num = self.mode_n_split_max_num(ub_size,
                                                          self.format_num,
                                                          self.cont)
            max_loop_batch_num = \
                max_loop_data_num // each_batch_data_num_format
            each_thread_batch_num = self._ceil_div(batch_num, thread_num)
            each_loop_batch_num = min(max_loop_batch_num,
                                      each_thread_batch_num,
                                      each_loop_max_batch_num)
            last_loop_batch_num, loop_times = self._get_loop_info(
                batch_num, each_loop_batch_num)
        with self.tik_inst.for_range(0, loop_times,
                                     thread_num=thread_num) as loop_index:
            batch_index = batch_index_start + each_loop_batch_num * loop_index
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._mode2_compute_each_loop(batch_index, each_loop_batch_num)
            with self.tik_inst.else_scope():
                self._mode2_compute_each_loop(batch_index, last_loop_batch_num)

    def _mode2_compute_each_loop(self, batch_index_start, batch_num):
        # fp32 data info
        batch_num_format_block = self._get_format_num(batch_num,
                                                      self.fp32_block_data_num)
        each_batch_repeat_num = self._ceil_div(self.data_num,
                                               self.fp32_repeat_data_num)
        each_batch_data_num = self.fp32_repeat_data_num * each_batch_repeat_num
        work_tensor_data_num = self._get_format_num(each_batch_repeat_num,
                                                    self.fp32_block_data_num)
        # init tensor
        reduce_work_tensor = self.tik_inst.Tensor(
            self.fp32_type, (work_tensor_data_num,), self.tik.scope_ubuf,
            "reduce_work_tensor")
        (buf_mean_all, batch_ne_mean_ub, batch_mean_square_ub,
         batch_rec_std_ub) = \
            self._init_mean_tensor(batch_num_format_block)
        buf_data_all, input_data_ub, input_data_square_ub, fp16_data_ub = \
            self._init_data_tensor(batch_num, each_batch_data_num)
        data_ub_all = (input_data_ub, input_data_square_ub, fp16_data_ub)
        mean_ub_all = (
            batch_ne_mean_ub, batch_mean_square_ub, batch_rec_std_ub)
        # data move in
        self._mode2_data_move_in(data_ub_all, buf_data_all, batch_index_start,
                                 batch_num)
        # count sum
        self._mode2_count_sum(data_ub_all, mean_ub_all, reduce_work_tensor,
                              batch_num)
        # last 32 byte data info
        last_index_start_ub = self._get_format_num(self.data_num,
                                                   self.format_num, False)
        data_num_last_gm = self.data_num - last_index_start_ub
        data_num_last_format = self._get_format_num(data_num_last_gm,
                                                    self.format_num)
        last_index_start_gm = self.data_num - data_num_last_format
        # adjust last 32b data
        self._mode2_adjust_input(data_ub_all, batch_index_start, batch_num,
                                 data_num_last_format, last_index_start_gm,
                                 last_index_start_ub)
        # count rec_std, ne_mean
        self._count_rec_std_ne_mean(buf_mean_all)
        # count (data add ne_mean) mul rec_std
        self._stand_data(input_data_ub, batch_ne_mean_ub, batch_rec_std_ub,
                         batch_num,
                         self.fp32_repeat_data_num, each_batch_repeat_num)
        # data mul gama add beta
        if self.elementwise:
            self._mode2_count_elementwise(
                data_ub_all, batch_num, buf_data_all, data_num_last_format,
                last_index_start_gm, last_index_start_ub)
        self._mode2_data_move_out(
            input_data_ub, fp16_data_ub, buf_data_all, batch_index_start,
            batch_num,
            data_num_last_format, last_index_start_gm, last_index_start_ub)

    def _mode2_data_move_in(self, data_ub_all, buf_data_all, batch_index_start,
                            batch_num):
        input_data_ub, input_data_square_ub, fp16_data_ub = data_ub_all
        # get block format num
        if self.input_tensor_type == "float16":
            each_block_data_num = self.fp16_block_data_num
            ub_tensor = fp16_data_ub
        else:
            each_block_data_num = self.fp32_block_data_num
            ub_tensor = input_data_ub
        # get each batch info
        each_batch_ub_data_num = self._get_format_num(
            self.data_num, self.fp32_repeat_data_num)
        each_batch_ub_block_num = self._ceil_div(each_batch_ub_data_num,
                                                 each_block_data_num)
        each_batch_gm_block_num = self._ceil_div(self.data_num,
                                                 each_block_data_num)
        gm_data_num_floor = each_block_data_num * (
                    self.data_num // each_block_data_num)
        last_num = self.data_num - gm_data_num_floor
        # if data 32byte align
        if last_num == 0:
            self.data_move(
                ub_tensor, self.input_tensor[batch_index_start, 0], 0,
                batch_num,
                each_batch_gm_block_num, 0,
                each_batch_ub_block_num - each_batch_gm_block_num)
        # if data not 32byte align
        else:
            start_index = self.data_num - gm_data_num_floor
            end_index = (each_batch_gm_block_num * each_block_data_num -
                         gm_data_num_floor)
            mask_num = 0
            for i in range(start_index, end_index):
                mask_num += 2 ** i
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self.data_move(
                    ub_tensor[batch_index, 0],
                    self.input_tensor[batch_index + batch_index_start, 0],
                    0, 1, each_batch_gm_block_num, 0, 0)
                self.tik_inst.vector_dup(
                    [0, mask_num], ub_tensor[batch_index, gm_data_num_floor],
                    0.0, 1, 1, 8)

        if self.input_tensor_type == "float16":
            cmd_fp16_2_fp32 = [
                VecGCmd(cmd_name="vconv", dst_name="input_data_ub",
                        src0_name="fp16_data_tensor_vconv", round_mode="")]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_data_all,
                                        cmd_fp16_2_fp32, "input_data_ub")

    def _mode2_count_sum(self, data_ub_all, mean_ub_all, reduce_work_tensor,
                         batch_num):
        input_data_ub, input_data_square_ub, fp16_data_ub = data_ub_all
        batch_ne_mean_ub, batch_mean_square_ub, batch_rec_std_ub = mean_ub_all
        repeat_format_num = self.fp32_repeat_data_num
        each_batch_repeat_num = self._ceil_div(self.data_num,
                                               repeat_format_num)
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self.tik_inst.vmul(
                repeat_format_num, input_data_square_ub[batch_index, 0],
                input_data_ub[batch_index, 0],
                input_data_ub[batch_index, 0], each_batch_repeat_num, 1, 1, 1,
                8, 8, 8)
            self.tik_inst.vec_reduce_add(
                repeat_format_num, batch_ne_mean_ub[batch_index],
                input_data_ub[batch_index, 0],
                reduce_work_tensor, each_batch_repeat_num, 8)
            self.tik_inst.vec_reduce_add(
                repeat_format_num, batch_mean_square_ub[batch_index],
                input_data_square_ub[batch_index, 0],
                reduce_work_tensor, each_batch_repeat_num, 8)

    def _mode2_adjust_input(self, data_ub_all, batch_index_start, batch_num,
                            data_num_last_format,
                            last_index_start_gm, last_index_start_ub):
        if data_num_last_format != 0:
            input_data_ub, input_data_square_ub, fp16_data_ub = data_ub_all
            if self.input_tensor_type == "float16":
                block_num = data_num_last_format // self.fp16_block_data_num
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    self.data_move(
                        fp16_data_ub, self.input_tensor[
                            batch_index_start + batch_index,
                            last_index_start_gm],
                        0, 1, block_num, 0, 0)
                    self.tik_inst.vconv(
                        data_num_last_format, "",
                        input_data_ub[batch_index, last_index_start_ub],
                        fp16_data_ub,
                        1, 1, 1, 8, 8)
            else:
                block_num = data_num_last_format // self.fp32_block_data_num
                with self.tik_inst.for_range(0, batch_num) as batch_index:
                    self.data_move(
                        input_data_ub[batch_index, last_index_start_ub],
                        self.input_tensor[
                            batch_index_start + batch_index,
                            last_index_start_gm],
                        0, 1, block_num, 0, 0)

    def _mode2_count_elementwise(self, data_ub_all, batch_num, buf_data_all,
                                 data_num_last_format,
                                 last_index_start_gm, last_index_start_ub):
        input_data_ub, input_data_square_ub, fp16_data_ub = data_ub_all
        each_batch_repeat_num = self._ceil_div(self.data_num,
                                               self.fp32_repeat_data_num)
        cmd_fp16_2_fp32 = [
            VecGCmd(cmd_name="vconv", dst_name="input_data_square_ub",
                    src0_name="fp16_data_tensor_vconv", round_mode="")]
        count_mask = self.fp32_repeat_data_num
        self._mode2_weight_bias_move_in(self.weight, input_data_square_ub,
                                        fp16_data_ub, buf_data_all,
                                        cmd_fp16_2_fp32,
                                        data_num_last_format,
                                        last_index_start_gm,
                                        last_index_start_ub)
        if each_batch_repeat_num == 1:
            self.tik_inst.vmul(
                count_mask, input_data_ub, input_data_ub, input_data_square_ub,
                batch_num, 1, 1, 1, 8, 8, 0)
        else:
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self.tik_inst.vmul(
                    count_mask, input_data_ub[batch_index, 0],
                    input_data_ub[batch_index, 0],
                    input_data_square_ub, each_batch_repeat_num, 1, 1, 1, 8, 8,
                    8)
        self._mode2_weight_bias_move_in(self.bias, input_data_square_ub,
                                        fp16_data_ub, buf_data_all,
                                        cmd_fp16_2_fp32,
                                        data_num_last_format,
                                        last_index_start_gm,
                                        last_index_start_ub)
        if each_batch_repeat_num == 1:
            self.tik_inst.vadd(
                count_mask, input_data_ub, input_data_ub, input_data_square_ub,
                batch_num, 1, 1, 1, 8, 8, 0)
        else:
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self.tik_inst.vadd(
                    count_mask, input_data_ub[batch_index, 0],
                    input_data_ub[batch_index, 0],
                    input_data_square_ub, each_batch_repeat_num, 1, 1, 1, 8, 8,
                    8)

    def _mode2_weight_bias_move_in(self, gm_tensor, input_data_square_ub,
                                   fp16_data_ub, buf_data_all, cmd_fp16_2_fp32,
                                   data_num_last_format, last_index_start_gm,
                                   last_index_start_ub):
        if gm_tensor.dtype == "float16":
            each_block_data_num = self.fp16_block_data_num
            ub_tensor = fp16_data_ub
        else:
            each_block_data_num = self.fp32_block_data_num
            ub_tensor = input_data_square_ub
        block_num_ub_floor = last_index_start_ub // each_block_data_num

        self.data_move(ub_tensor, gm_tensor, 0, 1, block_num_ub_floor,
                       0, 0)
        if data_num_last_format != 0:
            block_num_last = data_num_last_format // each_block_data_num
            self.data_move(ub_tensor[0, last_index_start_ub],
                           gm_tensor[last_index_start_gm],
                           0, 1, block_num_last, 0, 0)
        if gm_tensor.dtype == "float16":
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_data_all,
                                        cmd_fp16_2_fp32,
                                        "input_data_square_ub")

    def _mode2_data_move_out(self, input_data_square_ub, fp16_data_ub,
                             buf_data_all, batch_index_start,
                             batch_num, data_num_last_format,
                             last_index_start_gm, last_index_start_ub):
        if self.output_tensor_type == "float16":
            cmd_fp32_2_fp16 = [
                VecGCmd(cmd_name="vconv", dst_name="fp16_data_tensor_vconv",
                        src0_name="input_data_ub", round_mode="")]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_data_all,
                                        cmd_fp32_2_fp16, "input_data_ub")
            format_num = self.fp16_block_data_num
            ub_tensor = fp16_data_ub
        else:
            format_num = self.fp32_block_data_num
            ub_tensor = input_data_square_ub
        each_batch_data_num = self._get_format_num(self.data_num,
                                                   self.fp32_repeat_data_num)
        block_num_ub = each_batch_data_num // format_num
        block_num_gm = self.data_num // format_num
        if data_num_last_format == 0:
            self.data_move(
                self.output_tensor[batch_index_start, 0], ub_tensor,
                0, batch_num, block_num_gm, block_num_ub - block_num_gm, 0)
        else:
            block_num_last = data_num_last_format // format_num
            with self.tik_inst.for_range(0, batch_num) as batch_index:
                self.data_move(
                    self.output_tensor[batch_index_start + batch_index, 0],
                    ub_tensor[batch_index, 0],
                    0, 1, block_num_gm, 0, 0)
                self.data_move(
                    self.output_tensor[
                        batch_index_start + batch_index, last_index_start_gm],
                    ub_tensor[batch_index, last_index_start_ub],
                    0, 1, block_num_last, 0, 0)

    def tik_output_debug(self):
        return self.tik_inst
