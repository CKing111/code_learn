from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor


class LayerNormalizeBase:

    def __init__(self, input_tensor, weight, bias, output_tensor, eps,
                 elementwise, dims, kernel_name,
                 batch_num, data_num, format_num, if_fp16, ai_core_use, cont):
        self.eps = eps
        self.elementwise = elementwise
        self.dims = dims
        self.kernel_name = kernel_name
        self.batch_num = batch_num
        self.data_num = data_num
        self.format_num = format_num
        self.if_fp16 = if_fp16
        self.ai_core_use = ai_core_use
        self.cont = cont
        self.tik = self.cont.tik
        self.tik_inst = self.cont.tinst

        self.repeat_block_num = \
            (self.cont.const_vector_proc_byte // self.cont.const_block_byte)
        (self.fp16_type, self.fp16_size, self.fp16_block_data_num,
         self.fp16_repeat_data_num) = \
            self._get_type_const(cont, "float16")
        (self.fp32_type, self.fp32_size, self.fp32_block_data_num,
         self.fp32_repeat_data_num) = \
            self._get_type_const(cont, "float32")

        self.input_tensor_type = input_tensor.get("dtype")
        self.output_tensor_type = output_tensor.get("dtype")
        self.weight_type = weight.get("dtype")
        self.bias_type = bias.get("dtype")

        self.input_output_shape = (self.batch_num, self.data_num)
        if elementwise:
            self.weight_shape = (self.data_num,)
            self.bias_shape = (self.data_num,)
        else:
            self.weight_shape = weight.get("shape")
            self.bias_shape = bias.get("shape")

        self.input_tensor = self.tik_inst.Tensor(
            self.input_tensor_type, self.input_output_shape, self.tik.scope_gm,
            "input_tensor")
        self.weight = self.tik_inst.Tensor(
            self.weight_type, self.weight_shape, self.tik.scope_gm, "weight")
        self.bias = self.tik_inst.Tensor(
            self.bias_type, self.bias_shape, self.tik.scope_gm, "bias")
        self.output_tensor = self.tik_inst.Tensor(
            self.output_tensor_type, self.input_output_shape,
            self.tik.scope_gm, "output_tensor")

    @staticmethod
    def _get_type_const(cont, data_type):
        data_size = cont.const_dtype_byte.get(data_type)
        block_data_num = cont.get_vec_proc_num_per_cmd_blk(data_type)
        repeat_data_num = cont.get_vec_proc_num_per_cmd(data_type)
        return data_type, data_size, block_data_num, repeat_data_num

    @staticmethod
    def _ceil_div(dividend, divisor):
        return (dividend + divisor - 1) // divisor

    @staticmethod
    def _get_loop_info(all_data_num, each_loop_num):
        loop_times = LayerNormalizeBase._ceil_div(all_data_num, each_loop_num)
        last_loop_num = all_data_num - each_loop_num * (loop_times - 1)
        return last_loop_num, loop_times

    @staticmethod
    def _get_format_num(input_num, format_num, ceil=True):
        if ceil:
            result = (input_num + format_num - 1) // format_num * format_num
        else:
            result = input_num // format_num * format_num
        return result

    @staticmethod
    def get_ub_format_num(input_tensor, weight, bias, output_tensor,
                          elementwise, cont):
        fp16_block_data_num = \
            LayerNormalizeBase._get_type_const(cont, "float16")[2]
        fp32_block_data_num = \
            LayerNormalizeBase._get_type_const(cont, "float32")[2]
        input_tensor_type = input_tensor.get("dtype")
        output_tensor_type = output_tensor.get("dtype")
        format_num = fp32_block_data_num
        if_fp16 = False
        if input_tensor_type == "float16" or output_tensor_type == "float16":
            format_num = fp16_block_data_num
            if_fp16 = True
        elif elementwise:
            weight_type = weight.get("dtype")
            bias_type = bias.get("dtype")
            if weight_type == "float16" or bias_type == "float16":
                format_num = fp16_block_data_num
                if_fp16 = True
        return format_num, if_fp16

    @staticmethod
    def split_shape(input_tensor, dim_split):
        shape_split = input_tensor.get("shape")
        if dim_split < 0:
            dim_split = dim_split + len(shape_split)
        else:
            dim_split = dim_split
        batch_num = 1
        data_num = 1
        for i in range(dim_split):
            batch_num *= shape_split[i]
        for i in range(dim_split, len(shape_split)):
            data_num *= shape_split[i]
        return batch_num, data_num, dim_split

    @staticmethod
    def mode_n_split_max_num(ub_size, format_num, cont):
        fp16_type = "float16"
        fp32_type = "float32"
        fp32_size = float(cont.const_dtype_byte.get(fp32_type))
        fp16_size = cont.const_dtype_byte.get(fp16_type)

        block_size = cont.const_block_byte
        repeat_max_num = cont.const_vector_proc_max_rpt
        fp32_repeat_data_num = cont.get_vec_proc_num_per_cmd(fp32_type)
        num_max = repeat_max_num * fp32_repeat_data_num
        fp16_block_data_num = block_size // fp16_size

        expand_tensor_num = 4
        expand_size = expand_tensor_num * block_size
        ub_size_remain = ub_size - expand_size
        count_tensor_num = 2
        each_data_size = (fp32_size * count_tensor_num + fp32_size /
                          fp32_repeat_data_num)
        if format_num == fp16_block_data_num:
            each_data_size += fp16_size
        data_num_max = int(ub_size_remain / each_data_size)
        data_num_format = (data_num_max // fp32_repeat_data_num *
                           fp32_repeat_data_num)
        return min(num_max, data_num_format)

    def _init_mean_tensor(self, batch_num_mean):
        tensor_type = self.fp32_type
        num_per_cmd = self.fp32_repeat_data_num
        batch_ne_mean_ub = self.tik_inst.Tensor(
            tensor_type, (batch_num_mean,), self.tik.scope_ubuf,
            "batch_ne_mean_ub")
        batch_mean_square_ub = self.tik_inst.Tensor(
            tensor_type, (batch_num_mean,), self.tik.scope_ubuf,
            "batch_mean_square_ub")
        batch_rec_std_ub = self.tik_inst.Tensor(
            tensor_type, (batch_num_mean,), self.tik.scope_ubuf,
            "batch_rec_std_ub")
        buf_mean_all = {
            "batch_ne_mean_ub": AVecBuf(
                batch_ne_mean_ub, batch_num_mean, 0, self.cont, False,
                num_per_cmd),
            "batch_mean_square_ub": AVecBuf(
                batch_mean_square_ub, batch_num_mean, 0, self.cont, False,
                num_per_cmd),
            "batch_rec_std_ub": AVecBuf(
                batch_rec_std_ub, batch_num_mean, 0, self.cont, False,
                num_per_cmd)}
        cmd_dup_mean_tensor = [
            VecGCmd(cmd_name="vector_dup", dst_name="batch_ne_mean_ub",
                    scalar=0),
            VecGCmd(cmd_name="vector_dup", dst_name="batch_mean_square_ub",
                    scalar=0),
            VecGCmd(cmd_name="vector_dup", dst_name="batch_rec_std_ub",
                    scalar=0), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_mean_all,
                                    cmd_dup_mean_tensor, "batch_ne_mean_ub")
        return (buf_mean_all, batch_ne_mean_ub,
                batch_mean_square_ub, batch_rec_std_ub)

    def _get_rec_std_ne_mean_cmd(self):
        cmd_count_var_data = [
            VecGCmd("vmul", "batch_rec_std_ub", "batch_ne_mean_ub",
                    "batch_ne_mean_ub"),  # E(x)^2
            VecGCmd("vmuls", "batch_ne_mean_ub", "batch_ne_mean_ub",
                    scalar=-1),  # -E(x)
            VecGCmd("vsub", "batch_rec_std_ub", "batch_mean_square_ub",
                    "batch_rec_std_ub"),  # E(x^2) - E(x)^2
            VecGCmd("vadds", "batch_rec_std_ub", "batch_rec_std_ub",
                    scalar=self.eps),  # var(x) + eps
            VecGCmd("vsqrt", "batch_rec_std_ub", "batch_rec_std_ub"),  # std(x)
            VecGCmd("vector_dup", "batch_mean_square_ub", scalar=1),  # 1
            VecGCmd("vdiv", "batch_rec_std_ub", "batch_mean_square_ub",
                    "batch_rec_std_ub"), ]  # 1/std(x)
        return cmd_count_var_data

    def data_move(self, dst, src, sid, nburst, burst, src_stride, dst_stride):
        if isinstance(nburst, int) and isinstance(burst, int):
            if nburst > 0 and burst > 0:
                self.tik_inst.data_move(dst, src, sid, nburst, burst,
                                        src_stride, dst_stride)


class LayerNormalizeSplitD(LayerNormalizeBase):
    """
    layer normalize:
        split mode: spilt d dim, each batch count
    """

    def mode_compute(self):
        """
        start mode compute, d larger
        """
        each_core_batch_num = self._ceil_div(self.batch_num, self.ai_core_use)
        last_core_batch_num, self.ai_core_use = self._get_loop_info(
            self.batch_num, each_core_batch_num)
        with self.tik_inst.for_range(0, self.ai_core_use,
                                     block_num=self.ai_core_use) as core_index:
            batch_index = each_core_batch_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                self._mode_compute_each_core(batch_index, each_core_batch_num)
            with self.tik_inst.else_scope():
                self._mode_compute_each_core(batch_index, last_core_batch_num)
        self.tik_inst.BuildCCE(
            inputs=[self.input_tensor, self.weight, self.bias],
            outputs=[self.output_tensor],
            kernel_name=self.kernel_name)

    def _mode_compute_each_core(self, batch_index_start, batch_num):
        if batch_num == 1:
            thread_num = 1
        else:
            thread_num = 2
        with self.tik_inst.for_range(0, batch_num,
                                     thread_num=thread_num) \
                as batch_index_temp:
            batch_index = batch_index_start + batch_index_temp
            self._mode_compute_each_loop(batch_index, thread_num)

    def _mode_compute_each_loop(self, batch_index, thread_num):
        ub_size = self.cont.const_ub_max_byte // thread_num
        (buf_mean_all, batch_ne_mean_ub, batch_mean_square_ub,
         batch_rec_std_ub) = \
            self._init_mean_tensor(self.fp32_block_data_num)
        mean_ub_all = (batch_ne_mean_ub, batch_mean_square_ub,
                       batch_rec_std_ub)
        # count sum
        with self.tik_inst.new_stmt_scope():
            self._count_mean_each_batch(ub_size, batch_index, mean_ub_all)
        # count rec_std, ne_mean
        self._count_rec_std_ne_mean(buf_mean_all)

        with self.tik_inst.new_stmt_scope():
            self._count_stand_each_batch(ub_size, batch_index, mean_ub_all)

    def _count_mean_each_batch(self, ub_size, batch_index, mean_ub_all):
        each_loop_mean_num = self._get_mean_each_loop_num(ub_size,
                                                          self.fp32_size)
        last_loop_mean_num, mean_loop_num = self._get_loop_info(
            self.data_num, each_loop_mean_num)
        repeat_num = each_loop_mean_num // self.fp32_repeat_data_num
        work_tensor_data_num = self._get_format_num(
            repeat_num, self.fp32_block_data_num)
        src_tensor_ub = self.tik_inst.Tensor(
            self.fp32_type, (each_loop_mean_num,), self.tik.scope_ubuf,
            "src_tensor_ub")
        work_tensor_ub = self.tik_inst.Tensor(
            self.fp32_type, (work_tensor_data_num,), self.tik.scope_ubuf,
            "work_tensor_ub")
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        buf_sum_ub = {
            "src_tensor_ub": AVecBuf(src_tensor_ub, each_loop_mean_num, 0,
                                     self.cont, False, num_per_cmd)}

        fp16_data_tensor = None
        if self.input_tensor_type == "float16":
            fp16_data_type = "float16"
            fp16_data_tensor = self.tik_inst.Tensor(
                fp16_data_type, (each_loop_mean_num,), self.tik.scope_ubuf,
                "fp16_data_tensor")
            num_per_cmd_fp16 = self.cont.get_vec_proc_num_per_cmd(
                fp16_data_type)
            buf_sum_ub["fp16_data_tensor"] = AVecBuf(
                fp16_data_tensor, each_loop_mean_num, 0, self.cont, False,
                num_per_cmd_fp16)
            buf_sum_ub["fp16_data_tensor_vconv"] = AVecBuf(
                fp16_data_tensor, each_loop_mean_num, 0, self.cont, False,
                num_per_cmd)
        count_tensor_all = (src_tensor_ub, work_tensor_ub, fp16_data_tensor)
        with self.tik_inst.for_range(0, mean_loop_num) as loop_index:
            start_index = loop_index * each_loop_mean_num
            with self.tik_inst.if_scope(loop_index != mean_loop_num - 1):
                self._count_mean_each_batch_loop(
                    batch_index, start_index, each_loop_mean_num,
                    count_tensor_all, mean_ub_all, buf_sum_ub)
            with self.tik_inst.else_scope():
                self._count_mean_each_batch_loop(
                    batch_index, start_index, last_loop_mean_num,
                    count_tensor_all, mean_ub_all, buf_sum_ub)

    def _get_mean_each_loop_num(self, ub_size, data_size):
        fp16_data_size = 2.0
        tensor_num = 4
        ub_size_last = ub_size - tensor_num * self.cont.const_block_byte
        data_size = float(data_size)
        if self.input_tensor_type == "float16":
            each_data_size = (data_size + fp16_data_size +
                              data_size / self.fp32_repeat_data_num)
        else:
            each_data_size = data_size + data_size / self.fp32_repeat_data_num
        data_num_max = int(ub_size_last / each_data_size)
        data_num_max_format = self._get_format_num(data_num_max,
                                                   self.fp32_repeat_data_num,
                                                   False)
        return data_num_max_format

    def _count_mean_each_batch_loop(self, batch_index, start_index, data_num,
                                    count_tensor_all, mean_ub_all, buf_sum_ub):
        mean_data_ub, mean_data_square_ub, mean_temp_ub = mean_ub_all
        src_tensor_ub, work_tensor_ub, fp16_data_tensor = count_tensor_all
        repeat_num = self._ceil_div(data_num, self.fp32_repeat_data_num)

        if self.input_tensor_type == "float16":
            block_data_num_fp16 = 16
            block_num = self._ceil_div(data_num, block_data_num_fp16)
            data_num_format = self._get_format_num(data_num,
                                                   block_data_num_fp16)
            cmd_dup_fp16_tensor = [
                VecGCmd(cmd_name="vector_dup", dst_name="fp16_data_tensor",
                        scalar=0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_sum_ub,
                                        cmd_dup_fp16_tensor,
                                        "fp16_data_tensor")
            self.data_move(
                fp16_data_tensor, self.input_tensor[batch_index, start_index],
                0, 1, block_num, 0, 0)
            cmd_vconv = [
                VecGCmd(cmd_name="vconv", dst_name="src_tensor_ub",
                        src0_name="fp16_data_tensor_vconv", round_mode="")]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_sum_ub, cmd_vconv,
                                        "src_tensor_ub")
        else:
            block_num = self._ceil_div(data_num, self.fp32_block_data_num)
            data_num_format = self._get_format_num(data_num, block_num)
            cmd_dup_fp32_tensor = [
                VecGCmd(cmd_name="vector_dup", dst_name="src_tensor_ub",
                        scalar=0)]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_sum_ub,
                                        cmd_dup_fp32_tensor, "src_tensor_ub")
            self.data_move(
                src_tensor_ub, self.input_tensor[batch_index, start_index], 0,
                1, block_num, 0, 0)
        if data_num_format != data_num:
            self._dup_last_data(src_tensor_ub, data_num, data_num_format)

        # count mean(x)
        self._count_mean_each_batch_loop_count(
            mean_temp_ub, src_tensor_ub, work_tensor_ub,
            repeat_num, mean_data_ub)
        # count mean(x^2)
        cmd_square = [
            VecGCmd(cmd_name="vmul", dst_name="src_tensor_ub",
                    src0_name="src_tensor_ub", src1_name="src_tensor_ub")]
        VecGExecutor.exec_vec_g_cmd(
            self.cont, buf_sum_ub, cmd_square, "src_tensor_ub")
        self._count_mean_each_batch_loop_count(
            mean_temp_ub, src_tensor_ub, work_tensor_ub,
            repeat_num, mean_data_square_ub)

    def _dup_last_data(self, src_tensor_ub, data_num, data_num_format):
        block_format_num = self.fp32_block_data_num
        ub_index_start = self._get_format_num(
            data_num, block_format_num, False)
        mask_dup = 0
        mask_start = data_num - ub_index_start
        mask_end = data_num_format - ub_index_start
        for i in range(mask_start, mask_end):
            mask_dup += 2 ** i
        self.tik_inst.vector_dup(
            [0, mask_dup], src_tensor_ub[ub_index_start], 0.0, 1, 1, 8)

    def _count_mean_each_batch_loop_count(self,  mean_temp_ub,
                                          src_tensor_ub, work_tensor_ub,
                                          repeat_num, mean_data_ub):
        mask_sum = self.fp32_repeat_data_num
        mask_mean = self.fp32_block_data_num
        self.tik_inst.vec_reduce_add(mask_sum, mean_temp_ub, src_tensor_ub,
                                     work_tensor_ub, repeat_num, 8)
        self.tik_inst.vmuls(mask_mean, mean_temp_ub, mean_temp_ub,
                            1.0 / self.data_num, 1, 1, 1, 8, 8)
        self.tik_inst.vadd(mask_mean, mean_data_ub, mean_data_ub, mean_temp_ub,
                           1, 1, 1, 1, 8, 8, 8)

    def _count_stand_each_batch(self, ub_size, batch_index, mean_ub_all):
        batch_ne_mean_ub, batch_mean_square_ub, batch_rec_std_ub = mean_ub_all
        ne_mean_scalar = self.tik_inst.Scalar(self.fp32_type)
        rec_std_scalar = self.tik_inst.Scalar(self.fp32_type)
        ne_mean_scalar.set_as(batch_ne_mean_ub[0])
        rec_std_scalar.set_as(batch_rec_std_ub[0])
        scalar_all = (ne_mean_scalar, rec_std_scalar)
        each_loop_stand_num = self._get_count_each_loop_num(ub_size,
                                                            self.fp32_size)
        last_loop_stand_num, stand_loop_num = self._get_loop_info(
            self.data_num, each_loop_stand_num)
        num_per_cmd = self.cont.get_vec_proc_num_per_cmd(self.fp32_type)
        src_stand_ub = self.tik_inst.Tensor(
            self.fp32_type, (each_loop_stand_num,), self.tik.scope_ubuf,
            "src_stand_ub")
        buf_stand_ub = {
            "src_stand_ub": AVecBuf(src_stand_ub, each_loop_stand_num, 0,
                                    self.cont, False, num_per_cmd)}
        gamma_beta_tensor_ub = None
        fp16_data_tensor = None
        if self.elementwise:
            gamma_beta_tensor_ub = self.tik_inst.Tensor(
                self.fp32_type, (each_loop_stand_num,), self.tik.scope_ubuf,
                "gamma_beta_tensor_ub")
            buf_stand_ub["gamma_beta_tensor_ub"] = AVecBuf(
                gamma_beta_tensor_ub, each_loop_stand_num, 0, self.cont, False,
                num_per_cmd)
        if self.if_fp16:
            fp16_data_type = "float16"
            fp16_data_tensor = self.tik_inst.Tensor(
                fp16_data_type, (each_loop_stand_num,), self.tik.scope_ubuf,
                "fp16_data_tensor")
            buf_stand_ub["fp16_data_tensor_vconv"] = AVecBuf(
                fp16_data_tensor, each_loop_stand_num, 0, self.cont, False,
                num_per_cmd)
        stand_ub_all = (src_stand_ub, gamma_beta_tensor_ub, fp16_data_tensor)
        with self.tik_inst.for_range(0, stand_loop_num) as loop_index:
            start_index = loop_index * each_loop_stand_num
            with self.tik_inst.if_scope(loop_index != stand_loop_num - 1):
                self._count_stand_each_batch_loop(
                    batch_index, start_index, each_loop_stand_num, scalar_all,
                    stand_ub_all, buf_stand_ub)
            with self.tik_inst.else_scope():
                self._count_stand_each_batch_loop(
                    batch_index, start_index, last_loop_stand_num, scalar_all,
                    stand_ub_all, buf_stand_ub)

    def _get_count_each_loop_num(self, ub_size, data_size):
        fp16_data_size = 2.0
        tensor_num = 3
        ub_size_last = ub_size - tensor_num * self.cont.const_block_byte
        data_size = float(data_size)
        each_data_size = data_size
        if self.elementwise:
            each_data_size += data_size
        if self.if_fp16:
            each_data_size += fp16_data_size
        data_num_max = int(ub_size_last / each_data_size)
        data_num_max_format = self._get_format_num(data_num_max,
                                                   self.fp32_repeat_data_num,
                                                   False)
        return data_num_max_format

    def _count_stand_each_batch_loop(self, batch_index, start_index, data_num,
                                     scalar_all, stand_ub_all, buf_stand_ub):
        ne_mean_scalar, rec_std_scalar = scalar_all
        src_stand_ub, gamma_beta_tensor_ub, fp16_data_tensor = stand_ub_all
        data_num_align = self._get_format_num(data_num, self.format_num, False)
        data_num_last = self._get_format_num(data_num - data_num_align,
                                             self.format_num)
        cmd_vconv_data = [
            VecGCmd(cmd_name="vconv", dst_name="src_stand_ub",
                    src0_name="fp16_data_tensor_vconv", round_mode="")]
        self._mode_data_move_in(
            src_stand_ub, fp16_data_tensor, buf_stand_ub, cmd_vconv_data,
            batch_index, start_index, data_num, data_num_align, data_num_last)
        cmd_stand = [
            VecGCmd(cmd_name="vadds", dst_name="src_stand_ub",
                    src0_name="src_stand_ub", scalar=ne_mean_scalar),
            VecGCmd(cmd_name="vmuls", dst_name="src_stand_ub",
                    src0_name="src_stand_ub", scalar=rec_std_scalar)]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_stand_ub, cmd_stand,
                                    "src_stand_ub")
        if self.elementwise:
            cmd_vconv_weight_bias = [
                VecGCmd(cmd_name="vconv", dst_name="gamma_beta_tensor_ub",
                        src0_name="fp16_data_tensor_vconv", round_mode="")]
            self._mode_weight_bias_move_in(
                self.weight, gamma_beta_tensor_ub, fp16_data_tensor,
                buf_stand_ub,
                cmd_vconv_weight_bias, start_index, data_num, data_num_align,
                data_num_last)
            VecGExecutor.exec_vec_g_cmd(
                self.cont, buf_stand_ub,
                [VecGCmd(cmd_name="vmul", dst_name="src_stand_ub",
                         src0_name="src_stand_ub",
                         src1_name="gamma_beta_tensor_ub")],
                "src_stand_ub")
            self._mode_weight_bias_move_in(
                self.bias, gamma_beta_tensor_ub, fp16_data_tensor,
                buf_stand_ub, cmd_vconv_weight_bias,
                start_index, data_num, data_num_align, data_num_last)
            VecGExecutor.exec_vec_g_cmd(
                self.cont, buf_stand_ub,
                [VecGCmd(cmd_name="vadd", dst_name="src_stand_ub",
                         src0_name="src_stand_ub",
                         src1_name="gamma_beta_tensor_ub")],
                "src_stand_ub")
        self._mode_data_move_out(
            src_stand_ub, fp16_data_tensor, buf_stand_ub, batch_index,
            start_index,
            data_num, data_num_align, data_num_last)

    def _mode_data_move_in(self, data_tensor_ub, fp16_tensor, buf_stand_ub,
                           cmd_vconv,
                           batch_index, start_index, data_num, data_num_align,
                           data_num_last):
        if self.input_tensor.dtype == "float16":
            block_format_num = self.fp16_block_data_num
            ub_tensor = fp16_tensor
        else:
            block_format_num = self.fp32_block_data_num
            ub_tensor = data_tensor_ub
        block_num_floor = self._ceil_div(data_num_align, block_format_num)
        self.data_move(
            ub_tensor, self.input_tensor[batch_index, start_index], 0, 1,
            block_num_floor, 0, 0)
        if data_num_last != 0:
            block_num_last = self._ceil_div(data_num_last, block_format_num)
            data_num_last_format = block_num_last * block_format_num
            last_data_index_gm = data_num - data_num_last_format
            self.data_move(
                ub_tensor[data_num_align], self.input_tensor[
                    batch_index, start_index + last_data_index_gm],
                0, 1, block_num_last, 0, 0)
        if self.input_tensor.dtype == "float16":
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_stand_ub, cmd_vconv,
                                        "src_stand_ub")

    def _mode_weight_bias_move_in(self, gm_tensor, data_tensor_ub, fp16_tensor,
                                  buf_stand_ub, cmd_vconv,
                                  start_index, data_num, data_num_align,
                                  data_num_last):
        if gm_tensor.dtype == "float16":
            block_format_num = self.fp16_block_data_num
            ub_tensor = fp16_tensor
        else:
            block_format_num = self.fp32_block_data_num
            ub_tensor = data_tensor_ub
        block_num_floor = self._ceil_div(data_num_align, block_format_num)
        self.data_move(
            ub_tensor, gm_tensor[start_index], 0, 1, block_num_floor, 0, 0)
        if data_num_last != 0:
            block_num_last = self._ceil_div(data_num_last, block_format_num)
            data_num_last_format = block_num_last * block_format_num
            last_data_index_gm = data_num - data_num_last_format
            self.data_move(
                ub_tensor[data_num_align],
                gm_tensor[start_index + last_data_index_gm],
                0, 1, block_num_last, 0, 0)
        if gm_tensor.dtype == "float16":
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_stand_ub, cmd_vconv,
                                        "src_stand_ub")

    def _mode_data_move_out(self, data_tensor_ub, fp16_tensor, buf_stand_ub,
                            batch_index, start_index, data_num,
                            data_num_align, data_num_last):
        if self.output_tensor_type == "float16":
            cmd_fp32_2_fp16 = [
                VecGCmd(cmd_name="vconv", dst_name="fp16_data_tensor_vconv",
                        src0_name="src_stand_ub", round_mode="")]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_stand_ub,
                                        cmd_fp32_2_fp16, "src_stand_ub")
            block_format_num = self.fp16_block_data_num
            ub_tensor = fp16_tensor
        else:
            block_format_num = self.fp32_block_data_num
            ub_tensor = data_tensor_ub
        block_num = data_num_align // block_format_num
        self.data_move(self.output_tensor[batch_index, start_index],
                       ub_tensor, 0, 1, block_num, 0, 0)
        if data_num_last != 0:
            block_num_last = self._ceil_div(data_num_last, block_format_num)
            data_num_last_format = block_num_last * block_format_num
            last_data_index_gm = data_num - data_num_last_format
            self.data_move(
                self.output_tensor[
                    batch_index, start_index + last_data_index_gm],
                ub_tensor[data_num_align],
                0, 1, block_num_last, 0, 0)

    def _count_rec_std_ne_mean(self, buf_mean_all):
        cmd_count_var_data = self._get_rec_std_ne_mean_cmd()
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_mean_all,
                                    cmd_count_var_data, "batch_ne_mean_ub")
