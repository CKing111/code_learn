from version import get_version
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from ascend import TilingND, TilingInfo
from uti import interface_check


class InstanceNormalization(TilingND):

    def __init__(self, batch_num, channel_num, data_num, elementwise,
                 epsilon, data_type, kernel_name):
        self.batch_num = batch_num
        self.channel_num = channel_num
        self.data_num = data_num
        self.elementwise = elementwise
        self.epsilon = epsilon
        self.kernel_name = kernel_name

        self.cont = get_version.get_aicore_container(("Ascend610",))
        super().__init__(self.cont,
                         self.batch_num * self.channel_num,
                         self.data_num)

        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst
        self.ub_size = self.cont.const_ub_max_byte
        self.block_size = self.cont.const_block_byte
        self.repeat_size = self.cont.const_vector_proc_byte

        self.gm_type = data_type
        self.gm_data_size, self.gm_block_data_num, self.gm_repeat_data_num = \
            self.get_type_const(self.cont, self.gm_type)
        self.ub_type = "float32"
        self.ub_data_size, self.ub_block_data_num, self.ub_repeat_data_num = \
            self.get_type_const(self.cont, self.ub_type)

        self.data_num_align = self.get_align_num(self.data_num,
                                                 self.ub_repeat_data_num)

        input_shape = (self.batch_num, self.channel_num, self.data_num)
        self.input_data = self.tik_inst.Tensor(
            self.gm_type, input_shape, self.tik.scope_gm, "input")
        if self.elementwise:
            weight_shape = (self.channel_num,)
            self.gamma = self.tik_inst.Tensor(
                self.gm_type, weight_shape, self.tik.scope_gm, "scale")
            self.beta = self.tik_inst.Tensor(
                self.gm_type, weight_shape, self.tik.scope_gm, "B")
            if self.gm_type != self.ub_type:
                self.gamma_ub_type = self.tik_inst.Tensor(
                    self.ub_type, weight_shape, self.tik.scope_gm,
                    "gamma_ub_type_gm", is_workspace=True)
                self.beta_ub_type = self.tik_inst.Tensor(
                    self.ub_type, weight_shape, self.tik.scope_gm,
                    "beta_ub_type_gm", is_workspace=True)

        is_atomic_add = (self.data_num % self.gm_block_data_num != 0)
        self.output_data = self.tik_inst.Tensor(
            self.gm_type, input_shape, self.tik.scope_gm, "output",
            is_atomic_add=is_atomic_add)

    def _vector_buffer(self, tensor_all, tensor_name, tensor, element_num):
        tensor_all[tensor_name] = AVecBuf(
            tensor, element_num, 0, self.cont, False, self.ub_repeat_data_num)

    def _count_each_batch_size(self):
        each_batch_size = ((self.data_num_align * 2
                            + self.data_num_align // self.ub_repeat_data_num)
                           * self.ub_data_size
                           + self.block_size * 4)
        if self.elementwise:
            each_batch_size += self.block_size * 2
        if self.gm_type != self.ub_type:
            each_batch_size += self.data_num_align * self.gm_data_size
        extra_size = 0
        return each_batch_size, extra_size

    def mode_compute(self):
        # Tend to use split n
        self._set_loop_batch_num(1)
        self._set_thread_mode(1)
        self._mode_compute()
        if self.elementwise:
            input_gm_tensor = [self.input_data, self.gamma, self.beta]
        else:
            input_gm_tensor = [self.input_data]
        self.tik_inst.BuildCCE(
            inputs=input_gm_tensor,
            outputs=[self.output_data],
            kernel_name=self.kernel_name)

    def _compute_gamma_convert(self):
        each_data_size = (self.gm_data_size + self.ub_data_size) * 2
        data_num_max = self.ub_size // each_data_size
        data_num_align = self.get_align_num(data_num_max,
                                            self.ub_repeat_data_num,
                                            False)
        loop_times, last_loop_data_num = \
            self.get_loop_info(self.channel_num, data_num_align)
        with self.tik_inst.for_range(0, loop_times) as loop_index:
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                gamma_ub_all = self._init_gamma_convert(data_num_align)
                self._gamma_convert(gamma_ub_all, data_num_align)
            with self.tik_inst.else_scope():
                gamma_ub_all = self._init_gamma_convert(last_loop_data_num)
                self._gamma_convert(gamma_ub_all, last_loop_data_num)

    def _init_gamma_convert(self, data_num):
        data_num_align = self.get_align_num(data_num, self.ub_repeat_data_num)
        gamma_shape = (data_num_align,)
        gamma_gm_type = self.tik_inst.Tensor(self.gm_type, gamma_shape,
                                             self.tik.scope_ubuf,
                                             "gamma_gm_type")
        gamma_ub_type = self.tik_inst.Tensor(self.ub_type, gamma_shape,
                                             self.tik.scope_ubuf,
                                             "gamma_ub_type")
        beta_gm_type = self.tik_inst.Tensor(self.gm_type, gamma_shape,
                                            self.tik.scope_ubuf,
                                            "beta_gm_type")
        beta_ub_type = self.tik_inst.Tensor(self.ub_type, gamma_shape,
                                            self.tik.scope_ubuf,
                                            "beta_ub_type")
        gamma_ub_all = {}
        self._vector_buffer(gamma_ub_all, "gamma_gm_type",
                            gamma_gm_type, data_num_align)
        self._vector_buffer(gamma_ub_all, "gamma_ub_type",
                            gamma_ub_type, data_num_align)
        self._vector_buffer(gamma_ub_all, "beta_gm_type",
                            beta_gm_type, data_num_align)
        self._vector_buffer(gamma_ub_all, "beta_ub_type",
                            beta_ub_type, data_num_align)
        return gamma_ub_all

    def _gamma_convert(self, gamma_ub_all, data_num):
        block_num_move_in = self.ceil_div(data_num, self.gm_block_data_num)
        gamma_gm_type = gamma_ub_all.get("gamma_gm_type").const_tensor
        gamma_ub_type = gamma_ub_all.get("gamma_ub_type").const_tensor
        beta_gm_type = gamma_ub_all.get("beta_gm_type").const_tensor
        beta_ub_type = gamma_ub_all.get("beta_ub_type").const_tensor
        self.tik_inst.data_move(gamma_gm_type, self.gamma, 0,
                                1, block_num_move_in, 0, 0)
        self.tik_inst.data_move(beta_gm_type, self.beta, 0,
                                1, block_num_move_in, 0, 0)
        convert_cmd = [
            VecGCmd(cmd_name="vconv", dst_name="gamma_ub_type",
                    src0_name="gamma_gm_type", round_mode=""),
            VecGCmd(cmd_name="vconv", dst_name="beta_ub_type",
                    src0_name="beta_gm_type", round_mode="")]
        VecGExecutor.exec_vec_g_cmd(self.cont, gamma_ub_all,
                                    convert_cmd, "gamma_gm_type")
        block_num_move_out = self.ceil_div(data_num, self.ub_block_data_num)
        self.tik_inst.data_move(self.gamma_ub_type, gamma_ub_type, 0,
                                1, block_num_move_out, 0, 0)
        self.tik_inst.data_move(self.beta_ub_type, beta_ub_type, 0,
                                1, block_num_move_out, 0, 0)

    def _core_compute_before_n(self, tile_info: TilingInfo):
        if self.elementwise and self.gm_type != self.ub_type:
            self._compute_gamma_convert()

    def _core_compute_before_d(self, tile_info: TilingInfo):
        if self.elementwise and self.gm_type != self.ub_type:
            self._compute_gamma_convert()

    def _compute_each_loop_n(self, tile_info: TilingInfo):
        batch_index_start_all_s = tile_info.loop_batch_index_start_s
        batch_index_start_s = batch_index_start_all_s // self.channel_num
        channel_index_start_s = batch_index_start_all_s % self.channel_num
        data_ub_all = self._init_data_n()
        mean_ub_all = self._init_mean()
        if self.elementwise:
            gamma_ub, beta_ub = self._init_gamma_beta()
            self._move_gamma_beta(gamma_ub, beta_ub, channel_index_start_s)
        else:
            gamma_ub, beta_ub = None, None
        self._move_data_n(data_ub_all, batch_index_start_s,
                          channel_index_start_s)

        self._count_sum_n(data_ub_all, mean_ub_all)

        self._count_rec_std_ne_mean(mean_ub_all)

        self._count_stand_n(gamma_ub, beta_ub, data_ub_all, mean_ub_all)
        self._mode_data_move_out(data_ub_all, batch_index_start_s,
                                 channel_index_start_s)

    def _init_data_n(self):
        data_num = self.data_num
        data_num_align = self.data_num_align
        repeat_num = data_num_align // self.ub_repeat_data_num
        data_shape = (1, data_num_align)
        work_tensor_shape = (repeat_num,)
        input_data_ub = self.tik_inst.Tensor(self.ub_type, data_shape,
                                             self.tik.scope_ubuf,
                                             "input_data_ub")
        input_data_square_ub = self.tik_inst.Tensor(self.ub_type, data_shape,
                                                    self.tik.scope_ubuf,
                                                    "input_data_square_ub")
        work_tensor_ub = self.tik_inst.Tensor(self.ub_type, work_tensor_shape,
                                              self.tik.scope_ubuf,
                                              "work_tensor_ub")

        data_ub_all = {}
        self._vector_buffer(data_ub_all, "input_data_ub",
                            input_data_ub, data_num)
        self._vector_buffer(data_ub_all, "input_data_square_ub",
                            input_data_square_ub, data_num)
        self._vector_buffer(data_ub_all, "work_tensor_ub",
                            work_tensor_ub, repeat_num)
        if self.gm_type != self.ub_type:
            input_data_ub_fp16 = self.tik_inst.Tensor(self.gm_type, data_shape,
                                                      self.tik.scope_ubuf,
                                                      "input_data_ub_fp16")
            self._vector_buffer(data_ub_all, "input_data_ub_fp16",
                                input_data_ub_fp16, data_num)
        return data_ub_all

    def _init_mean(self):
        batch_num = 1
        batch_num_align = self.ub_block_data_num
        mean_shape = (batch_num_align,)
        batch_mean_ub = self.tik_inst.Tensor(self.ub_type, mean_shape,
                                             self.tik.scope_ubuf,
                                             "batch_mean_ub")
        batch_mean_square_ub = self.tik_inst.Tensor(self.ub_type, mean_shape,
                                                    self.tik.scope_ubuf,
                                                    "batch_mean_square_ub")
        batch_variance_ub = self.tik_inst.Tensor(self.ub_type, mean_shape,
                                                 self.tik.scope_ubuf,
                                                 "batch_variance_ub")
        mean_ub_all = {}
        self._vector_buffer(mean_ub_all, "batch_mean_ub",
                            batch_mean_ub, batch_num)
        self._vector_buffer(mean_ub_all, "batch_mean_square_ub",
                            batch_mean_square_ub, batch_num)
        self._vector_buffer(mean_ub_all, "batch_variance_ub",
                            batch_variance_ub, batch_num)
        return mean_ub_all

    def _init_gamma_beta(self):
        batch_num_align = self.ub_block_data_num
        tensor_shape = (batch_num_align,)
        gamma_ub = self.tik_inst.Tensor(self.ub_type, tensor_shape,
                                        self.tik.scope_ubuf, "gamma_ub")
        beta_ub = self.tik_inst.Tensor(self.ub_type, tensor_shape,
                                       self.tik.scope_ubuf, "beta_ub")
        return gamma_ub, beta_ub

    def _move_gamma_beta(self, gamma_ub, beta_ub, channel_index_start_s):
        if self.gm_type != self.ub_type:
            gamma_gm = self.gamma_ub_type
            beta_gm = self.beta_ub_type
        else:
            gamma_gm = self.gamma
            beta_gm = self.beta
        block_num = 1
        self.tik_inst.data_move(
            gamma_ub, gamma_gm[channel_index_start_s],
            0, 1, block_num, 0, 0)
        self.tik_inst.data_move(
            beta_ub, beta_gm[channel_index_start_s],
            0, 1, block_num, 0, 0)

    def _move_data_n(self, data_ub_all, batch_index_start_s,
                     channel_index_start_s):
        if self.gm_type != self.ub_type:
            input_data_ub = data_ub_all.get("input_data_ub_fp16").const_tensor
        else:
            input_data_ub = data_ub_all.get("input_data_ub").const_tensor
        each_batch_gm_block_num = self.ceil_div(self.data_num,
                                                self.gm_block_data_num)

        self.tik_inst.data_move(
            input_data_ub,
            self.input_data[batch_index_start_s, channel_index_start_s, 0],
            0, 1, each_batch_gm_block_num, 0, 0)
        cmd_init_data = []
        if self.gm_type != self.ub_type:
            cmd_init_data.append(
                VecGCmd(cmd_name="vconv", dst_name="input_data_ub",
                        src0_name="input_data_ub_fp16", round_mode=""))
        cmd_init_data.append(
            VecGCmd(cmd_name="vmul", dst_name="input_data_square_ub",
                    src0_name="input_data_ub", src1_name="input_data_ub"))
        VecGExecutor.exec_vec_g_cmd(self.cont, data_ub_all,
                                    cmd_init_data, "input_data_ub")

    def _vector_reduce_add(self, result_ub, data_ub, work_ub, data_num):
        mask = self.ub_repeat_data_num
        if data_num <= mask:
            self.tik_inst.vcadd(data_num, result_ub, data_ub, 1, 1, 1, 8)
        else:
            repeat_num = data_num // mask
            self.tik_inst.vec_reduce_add(mask,
                                         result_ub,
                                         data_ub,
                                         work_ub,
                                         repeat_num, 8)
            start_index = repeat_num * mask
            data_last = data_num - start_index
            if data_last > 0:
                self.tik_inst.vcadd(data_last, work_ub,
                                    data_ub[0, start_index],
                                    1, 1, 1, 8)
                self.tik_inst.vadd(1, result_ub, result_ub, work_ub,
                                   1, 1, 1, 1, 8, 8, 8)

    def _count_sum_n(self, data_ub_all, mean_ub_all):
        input_data_ub = data_ub_all.get("input_data_ub").const_tensor
        input_data_square_ub = \
            data_ub_all.get("input_data_square_ub").const_tensor
        work_tensor_ub = data_ub_all.get("work_tensor_ub").const_tensor
        batch_mean_ub = mean_ub_all.get("batch_mean_ub").const_tensor
        batch_mean_square_ub = \
            mean_ub_all.get("batch_mean_square_ub").const_tensor

        self._vector_reduce_add(batch_mean_ub, input_data_ub,
                                work_tensor_ub, self.data_num)

        self._vector_reduce_add(batch_mean_square_ub, input_data_square_ub,
                                work_tensor_ub, self.data_num)

    def _count_rec_std_ne_mean(self, mean_ub_all):
        # count rec_std and ne_mean
        rec_std_ne_mean_cmd = [
            VecGCmd(cmd_name="vmuls", dst_name="batch_mean_ub",
                    src0_name="batch_mean_ub",
                    scalar=1.0 / self.data_num),  # E(x)
            VecGCmd(cmd_name="vmuls", dst_name="batch_mean_square_ub",
                    src0_name="batch_mean_square_ub",
                    scalar=1.0 / self.data_num),  # E(x^2)
            VecGCmd(cmd_name="vmul", dst_name="batch_variance_ub",
                    src0_name="batch_mean_ub",
                    src1_name="batch_mean_ub"),  # E(x)^2
            VecGCmd(cmd_name="vsub", dst_name="batch_variance_ub",
                    src0_name="batch_mean_square_ub",
                    src1_name="batch_variance_ub"),  # E(x^2) - E(x)^2
            VecGCmd(cmd_name="vmaxs", dst_name="batch_variance_ub",
                    src0_name="batch_variance_ub", scalar=0),
            VecGCmd(cmd_name="vmuls", dst_name="batch_mean_ub",
                    src0_name="batch_mean_ub", scalar=-1),  # -E(x)
            VecGCmd(cmd_name="vadds", dst_name="batch_variance_ub",
                    src0_name="batch_variance_ub",
                    scalar=self.epsilon),  # var(x) + eps
            VecGCmd(cmd_name="vsqrt", dst_name="batch_variance_ub",
                    src0_name="batch_variance_ub"),  # std(x)
            VecGCmd(cmd_name="vector_dup", dst_name="batch_mean_square_ub",
                    scalar=1),  # 1
            VecGCmd(cmd_name="vdiv", dst_name="batch_variance_ub",
                    src0_name="batch_mean_square_ub",
                    src1_name="batch_variance_ub"),  # 1/std(x)
        ]
        VecGExecutor.exec_vec_g_cmd(self.cont, mean_ub_all,
                                    rec_std_ne_mean_cmd, "batch_mean_ub")

    def _count_stand_n(self, gamma_ub, beta_ub, data_ub_all, mean_ub_all):
        batch_ne_mean_ub = mean_ub_all.get("batch_mean_ub").const_tensor
        batch_rec_std_ub = mean_ub_all.get("batch_variance_ub").const_tensor

        ne_mean_scalar = self.tik_inst.Scalar(self.ub_type)
        rec_std_scalar = self.tik_inst.Scalar(self.ub_type)
        ne_mean_scalar.set_as(batch_ne_mean_ub[0])
        rec_std_scalar.set_as(batch_rec_std_ub[0])
        cmd_std_data = [
            VecGCmd(cmd_name="vadds", dst_name="input_data_ub",
                    src0_name="input_data_ub", scalar=ne_mean_scalar),
            VecGCmd(cmd_name="vmuls", dst_name="input_data_ub",
                    src0_name="input_data_ub", scalar=rec_std_scalar)]

        if self.elementwise:
            gamma_scalar = self.tik_inst.Scalar(self.ub_type)
            beta_scalar = self.tik_inst.Scalar(self.ub_type)
            gamma_scalar.set_as(gamma_ub[0])
            beta_scalar.set_as(beta_ub[0])
            cmd_std_data.extend([
                VecGCmd(cmd_name="vmuls", dst_name="input_data_ub",
                        src0_name="input_data_ub", scalar=gamma_scalar),
                VecGCmd(cmd_name="vadds", dst_name="input_data_ub",
                        src0_name="input_data_ub", scalar=beta_scalar)])

        if self.gm_type != self.ub_type:
            cmd_std_data.append(
                VecGCmd(cmd_name="vconv", dst_name="input_data_ub_fp16",
                        src0_name="input_data_ub", round_mode=""))
        VecGExecutor.exec_vec_g_cmd(self.cont, data_ub_all,
                                    cmd_std_data, "input_data_ub")

    def _mode_data_move_out(self, data_ub_all, batch_index_start_s,
                            channel_index_start_s):
        if self.gm_type != self.ub_type:
            result_data_ub = data_ub_all.get("input_data_ub_fp16").const_tensor
        else:
            result_data_ub = data_ub_all.get("input_data_ub").const_tensor

        self._data_move_out(result_data_ub, batch_index_start_s,
                            channel_index_start_s, 0, self.data_num)

    def _compute_each_loop_d(self, tile_info: TilingInfo):
        batch_index_start_all_s = tile_info.loop_batch_index_start_s
        batch_index_start_s = batch_index_start_all_s // self.channel_num
        channel_index_start_s = batch_index_start_all_s % self.channel_num

        mean_ub_all = self._init_mean()
        self._count_sum_d(mean_ub_all,
                          batch_index_start_s,
                          channel_index_start_s)
        self._count_rec_std_ne_mean(mean_ub_all)
        self._count_stand_d(mean_ub_all,
                            batch_index_start_s,
                            channel_index_start_s)

    def _count_sum_d(self, mean_ub_all,
                     batch_index_start_s,
                     channel_index_start_s):
        each_loop_data_num = self._get_sum_loop_num()
        loop_times, last_loop_data_num = self.get_loop_info(self.data_num,
                                                            each_loop_data_num)
        batch_index_all = (batch_index_start_s, channel_index_start_s)
        data_num_all = (each_loop_data_num, last_loop_data_num)
        thread_num = min(loop_times, 2)

        cmd_dup_mean_tensor = [
            VecGCmd(cmd_name="vector_dup", dst_name="batch_mean_ub",
                    scalar=0),
            VecGCmd(cmd_name="vector_dup", dst_name="batch_mean_square_ub",
                    scalar=0),
            VecGCmd(cmd_name="vector_dup", dst_name="batch_variance_ub",
                    scalar=0), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, mean_ub_all,
                                    cmd_dup_mean_tensor, "batch_mean_ub")
        with self.tik_inst.for_range(0, loop_times,
                                     thread_num=thread_num) as loop_index:
            start_index = each_loop_data_num * loop_index
            data_ub_all = self._init_sum_data_d(each_loop_data_num,
                                                last_loop_data_num)
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._count_sum_d_each_loop(data_ub_all, mean_ub_all,
                                            batch_index_all, start_index,
                                            data_num_all, False)
            with self.tik_inst.else_scope():
                self._count_sum_d_each_loop(data_ub_all, mean_ub_all,
                                            batch_index_all, start_index,
                                            data_num_all, True)

    def _get_sum_loop_num(self):
        ub_size = self.ub_size - self.block_size * 3
        each_repeat_size = self.repeat_size + self.ub_data_size
        if self.gm_type != self.ub_type:
            each_repeat_size += (self.repeat_size // 2)
        repeat_num = self.data_num_align // self.ub_repeat_data_num
        each_batch_size = each_repeat_size * repeat_num + self.block_size
        if each_batch_size <= ub_size:
            each_loop_data_num = self.data_num_align
        else:
            ub_size = ub_size // 2
            each_loop_repeat_num = \
                (ub_size - self.block_size) // each_repeat_size
            each_loop_data_num = each_loop_repeat_num * self.ub_repeat_data_num
        return each_loop_data_num

    def _init_sum_data_d(self, each_loop_data_num, last_loop_data_num):
        repeat_num = each_loop_data_num // self.ub_repeat_data_num

        data_shape = (1, each_loop_data_num)
        work_tensor_shape = (repeat_num,)
        input_data_ub = self.tik_inst.Tensor(self.ub_type, data_shape,
                                             self.tik.scope_ubuf,
                                             "input_data_ub")
        work_tensor_ub = self.tik_inst.Tensor(self.ub_type, work_tensor_shape,
                                              self.tik.scope_ubuf,
                                              "work_tensor_ub")
        data_ub_all = {}
        self._vector_buffer(data_ub_all, "input_data_ub",
                            input_data_ub, each_loop_data_num)
        self._vector_buffer(data_ub_all, "last_input_data_ub",
                            input_data_ub, last_loop_data_num)
        self._vector_buffer(data_ub_all, "work_tensor_ub",
                            work_tensor_ub, repeat_num)
        if self.gm_type != self.ub_type:
            input_data_ub_fp16 = self.tik_inst.Tensor(self.gm_type, data_shape,
                                                      self.tik.scope_ubuf,
                                                      "input_data_ub_fp16")
            self._vector_buffer(data_ub_all, "input_data_ub_fp16",
                                input_data_ub_fp16, each_loop_data_num)
        return data_ub_all

    def _count_sum_d_each_loop(self, data_ub_all, mean_ub_all, batch_index_all,
                               start_index, data_num_all, if_last):
        batch_index_start_s, channel_index_start_s = batch_index_all
        data_num, drive_buf_name, block_num = self._get_loop_info(data_num_all,
                                                                  if_last)

        input_data_ub = data_ub_all.get("input_data_ub").const_tensor
        work_tensor_ub = data_ub_all.get("work_tensor_ub").const_tensor
        sum_ub = mean_ub_all.get("batch_mean_ub").const_tensor
        square_sum_ub = mean_ub_all.get("batch_mean_square_ub").const_tensor
        temp_ub = mean_ub_all.get("batch_variance_ub").const_tensor

        if self.gm_type != self.ub_type:
            input_data_ub_fp16 = \
                data_ub_all.get("input_data_ub_fp16").const_tensor
            self.tik_inst.data_move(
                input_data_ub_fp16,
                self.input_data[batch_index_start_s, channel_index_start_s,
                                start_index],
                0, 1, block_num, 0, 0)
            cmd_vconv = [
                VecGCmd(cmd_name="vconv", dst_name="input_data_ub",
                        src0_name="input_data_ub_fp16", round_mode="")]
            VecGExecutor.exec_vec_g_cmd(self.cont, data_ub_all,
                                        cmd_vconv, drive_buf_name)
        else:
            self.tik_inst.data_move(
                input_data_ub,
                self.input_data[batch_index_start_s,
                                channel_index_start_s,
                                start_index],
                0, 1, block_num, 0, 0)
        self._vector_reduce_add(temp_ub, input_data_ub,
                                work_tensor_ub, data_num)
        self.tik_inst.vadd(1, sum_ub, sum_ub, temp_ub,
                           1, 1, 1, 1, 8, 8, 8)
        cmd_square = [
            VecGCmd(cmd_name="vmul", dst_name="input_data_ub",
                    src0_name="input_data_ub", src1_name="input_data_ub")]
        VecGExecutor.exec_vec_g_cmd(self.cont, data_ub_all,
                                    cmd_square, drive_buf_name)
        self._vector_reduce_add(temp_ub, input_data_ub, work_tensor_ub,
                                data_num)
        self.tik_inst.vadd(1, square_sum_ub, square_sum_ub, temp_ub,
                           1, 1, 1, 1, 8, 8, 8)

    def _count_stand_d(self, mean_ub_all, batch_index_start_s,
                       channel_index_start_s):
        each_loop_data_num = self._get_stand_loop_num()
        loop_times, last_loop_data_num = self.get_loop_info(self.data_num,
                                                            each_loop_data_num)
        batch_index_all = (batch_index_start_s, channel_index_start_s)
        data_num_all = (each_loop_data_num, last_loop_data_num)

        batch_mean_ub = mean_ub_all.get("batch_mean_ub").const_tensor
        batch_variance_ub = mean_ub_all.get("batch_variance_ub").const_tensor
        ne_mean_s = self.tik_inst.Scalar(self.ub_type)
        rec_std_s = self.tik_inst.Scalar(self.ub_type)
        ne_mean_s.set_as(batch_mean_ub[0])
        rec_std_s.set_as(batch_variance_ub[0])
        scalar_all = {"ne_mean_scalar": ne_mean_s,
                      "rec_std_scalar": rec_std_s}
        if self.elementwise:
            gamma_ub, beta_ub = self._init_gamma_beta()
            self._move_gamma_beta(gamma_ub, beta_ub, channel_index_start_s)
            gamma_s = self.tik_inst.Scalar(self.ub_type)
            beta_s = self.tik_inst.Scalar(self.ub_type)
            gamma_s.set_as(gamma_ub[0])
            beta_s.set_as(beta_ub[0])
            scalar_all["gamma_s"] = gamma_s
            scalar_all["beta_s"] = beta_s

        thread_num = min(loop_times, 2)
        with self.tik_inst.for_range(0, loop_times,
                                     thread_num=thread_num) as loop_index:
            start_index = each_loop_data_num * loop_index
            data_ub_all = self._init_stand_data_d(each_loop_data_num,
                                                  last_loop_data_num)
            with self.tik_inst.if_scope(loop_index != loop_times - 1):
                self._count_stand_d_each_loop(data_ub_all, scalar_all,
                                              batch_index_all, start_index,
                                              data_num_all, False)
            with self.tik_inst.else_scope():
                self._count_stand_d_each_loop(data_ub_all, scalar_all,
                                              batch_index_all, start_index,
                                              data_num_all, True)

    def _get_stand_loop_num(self):
        ub_size = self.ub_size - self.block_size * 3
        if self.elementwise:
            ub_size -= self.block_size * 2
        each_repeat_size = self.repeat_size
        if self.gm_type != self.ub_type:
            each_repeat_size += (self.repeat_size // 2)
        repeat_num = self.data_num_align // self.ub_repeat_data_num
        each_batch_size = each_repeat_size * repeat_num
        if each_batch_size <= ub_size:
            each_loop_data_num = self.data_num_align
        else:
            ub_size = ub_size // 2
            each_loop_repeat_num = ub_size // each_repeat_size
            each_loop_data_num = each_loop_repeat_num * self.ub_repeat_data_num
        return each_loop_data_num

    def _init_stand_data_d(self, each_loop_data_num, last_loop_data_num):
        data_shape = (1, each_loop_data_num)
        input_data_ub = self.tik_inst.Tensor(self.ub_type, data_shape,
                                             self.tik.scope_ubuf,
                                             "input_data_ub")
        data_ub_all = {}
        self._vector_buffer(data_ub_all, "input_data_ub",
                            input_data_ub, each_loop_data_num)
        self._vector_buffer(data_ub_all, "last_input_data_ub",
                            input_data_ub, last_loop_data_num)
        if self.gm_type != self.ub_type:
            input_data_ub_fp16 = self.tik_inst.Tensor(self.gm_type, data_shape,
                                                      self.tik.scope_ubuf,
                                                      "input_data_ub_fp16")
            self._vector_buffer(data_ub_all, "input_data_ub_fp16",
                                input_data_ub_fp16, each_loop_data_num)
        return data_ub_all

    def _count_stand_d_each_loop(self, data_ub_all, scalar_all,
                                 batch_index_all, start_index,
                                 data_num_all, if_last):
        batch_index_start_s, channel_index_start_s = batch_index_all
        data_num, drive_buf_name, block_num = self._get_loop_info(data_num_all,
                                                                  if_last)
        if self.gm_type != self.ub_type:
            input_data_ub = data_ub_all.get("input_data_ub_fp16").const_tensor
        else:
            input_data_ub = data_ub_all.get("input_data_ub").const_tensor
        self.tik_inst.data_move(
            input_data_ub,
            self.input_data[
                batch_index_start_s, channel_index_start_s, start_index],
            0, 1, block_num, 0, 0)

        count_stand_cmd = []
        if self.gm_type != self.ub_type:
            count_stand_cmd.append(
                VecGCmd(cmd_name="vconv", dst_name="input_data_ub",
                        src0_name="input_data_ub_fp16", round_mode=""))
        count_stand_cmd.extend([
            VecGCmd(cmd_name="vadds", dst_name="input_data_ub",
                    src0_name="input_data_ub",
                    scalar=scalar_all.get("ne_mean_scalar")),
            VecGCmd(cmd_name="vmuls", dst_name="input_data_ub",
                    src0_name="input_data_ub",
                    scalar=scalar_all.get("rec_std_scalar"))
        ])

        if self.elementwise:
            count_stand_cmd.extend([
                VecGCmd(cmd_name="vmuls", dst_name="input_data_ub",
                        src0_name="input_data_ub",
                        scalar=scalar_all.get("gamma_s")),
                VecGCmd(cmd_name="vadds", dst_name="input_data_ub",
                        src0_name="input_data_ub",
                        scalar=scalar_all.get("beta_s"))])

        if self.gm_type != self.ub_type:
            count_stand_cmd.append(
                VecGCmd(cmd_name="vconv", dst_name="input_data_ub_fp16",
                        src0_name="input_data_ub", round_mode=""))
        VecGExecutor.exec_vec_g_cmd(self.cont, data_ub_all,
                                    count_stand_cmd, drive_buf_name)
        self._data_move_out(input_data_ub, batch_index_start_s,
                            channel_index_start_s, start_index, data_num)

    def _data_move_out(self, result_data_ub, batch_index_start_s,
                       channel_index_start_s, start_index, data_num):
        if data_num % self.gm_block_data_num == 0:
            block_num = data_num // self.gm_block_data_num
            self.tik_inst.data_move(
                self.output_data[batch_index_start_s,
                                 channel_index_start_s, start_index],
                result_data_ub, 0, 1, block_num, 0, 0)
        else:
            block_num = self.ceil_div(data_num, self.gm_block_data_num)
            mask_h, mask_l, mask_index = self.get_mask(data_num,
                                                       self.gm_repeat_data_num,
                                                       self.gm_block_data_num)

            self.tik_inst.vector_dup([mask_h, mask_l],
                                     result_data_ub[0, mask_index],
                                     0.0, 1, 1, 8)
            add_mode = 1 if self.gm_type == "float32" else 2
            self.tik_inst.set_atomic_add(add_mode)
            self.tik_inst.data_move(
                self.output_data[batch_index_start_s,
                                 channel_index_start_s, start_index],
                result_data_ub, 0,
                1, block_num, 0, 0)

            self.tik_inst.set_atomic_add(0)

    def _get_loop_info(self, data_num_all, if_last):
        each_loop_data_num, last_loop_data_num = data_num_all
        if if_last:
            data_num = last_loop_data_num
            drive_buf_name = "last_input_data_ub"
            block_num = self.ceil_div(data_num, self.gm_block_data_num)
        else:
            data_num = each_loop_data_num
            drive_buf_name = "input_data_ub"
            block_num = data_num // self.gm_block_data_num
        return data_num, drive_buf_name, block_num

    def tik_output_debug(self):
        return self.tik_inst


def check_params_attrs_name(input_data, gamma, beta, output_data, epsilon,
                            kernel_name):
    interface_check.check_kernelname(kernel_name)
    check_params(input_data, gamma, beta, output_data)
    check_attrs(epsilon)
    check_shape(input_data, gamma, beta, output_data)


def check_params(input_data, gamma, beta, output_data):
    support_shape_length = [shape_length for shape_length in range(3, 9)]
    support_dtype = ["float16", "float32"]
    support_format = ["NCHW", "ND"]
    interface_check.check_param(input_data, support_shape_length,
                                support_dtype, support_format)
    support_dtype = [input_data.get("dtype")]
    interface_check.check_param(output_data, support_shape_length,
                                support_dtype, support_format)

    if gamma is not None and beta is not None:
        interface_check.check_param(gamma, [1], support_dtype, support_format)
        interface_check.check_param(beta, [1], support_dtype, support_format)
    elif gamma is not None or beta is not None:
        raise RuntimeError("[ERROR][InstanceNormalization] "
                           "Both gamma and beta must exist or not exist")


def check_attrs(epsilon):
    if not isinstance(epsilon, float) or epsilon <= 0:
        raise RuntimeError(
            "[ERROR][InstanceNormalization] param epsilon is not supported")


def check_shape(input_data, gamma, beta, output_data):
    input_data_shape = input_data.get("shape")
    output_data_shape = output_data.get("shape")
    if output_data_shape != input_data_shape:
        raise RuntimeError(
            "[ERROR][InstanceNormalization] output shape is not supported")
    if gamma is not None and beta is not None:
        gamma_shape = gamma.get("shape")
        beta_shape = beta.get("shape")
        if gamma_shape != input_data_shape[1:2]:
            raise RuntimeError(
                "[ERROR][InstanceNormalization] gamma shape is not supported")
        if beta_shape != gamma_shape:
            raise RuntimeError(
                "[ERROR][InstanceNormalization] beta shape is not supported")


def split_shape(input_tensor):
    shape_split = input_tensor.get("shape")
    batch_num = shape_split[0]
    channel_num = shape_split[1]
    dim_split = 2
    data_num = 1
    for i in range(dim_split, len(shape_split)):
        data_num *= shape_split[i]
    return batch_num, channel_num, data_num


def instance_normalization(input_data, gamma, beta, output_data, epsilon,
                           kernel_name="InstanceNormalization", test=False):
    check_params_attrs_name(input_data, gamma, beta, output_data, epsilon,
                            kernel_name)

    batch_num, channel_num, data_num = split_shape(input_data)
    data_type = input_data.get("dtype")
    if gamma is None:
        elementwise = False
    else:
        elementwise = True
    obj = InstanceNormalization(batch_num, channel_num, data_num, elementwise,
                                epsilon, data_type, kernel_name)
    obj.mode_compute()
    if test:
        obj.tik_output_debug()
