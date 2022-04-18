from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor


class RevertIndex():
    def __init__(self, container, padding_mode, align_corners):
        self.tinst = container.tinst
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.cont = container
        self.tik = container.tik
        self.num_per_cmd = self.cont.get_vec_proc_num_per_cmd("float32")

    def cmp_sel(self, process_num, sel_result, former, latter, if_former,
                if_latter):
        '''
        sel_result = former cmp_mode latter ? if_former:if_latter
        :param process_num: the length of input tensor
        :param sel_result: return result
        :param former: former tensor[process_num]
        :param latter: Scalar(mode) or Tensor
        :param if_former: choose if former compare success[process_num]
        :param if_latter: choose if latter compare success[process_num]
        :param cmp_mode: support [">","<","=="]
        '''
        repeat_time = process_num // self.num_per_cmd
        repeat_time_last = process_num - repeat_time * self.num_per_cmd

        rep_offset = 0
        with self.tinst.new_stmt_scope():
            latter_tensor = self.tinst.Tensor("float32",
                                              (self.num_per_cmd,),
                                              self.tik.scope_ubuf,
                                              "latter_tensor")
            self.tinst.vector_dup(self.num_per_cmd, latter_tensor, latter,
                                  1, 1, 8)
            if repeat_time > 0:
                with self.tinst.for_range(0, repeat_time) as rt_time:
                    self._cmp_sel_found(self.num_per_cmd, rt_time,
                                        rep_offset, sel_result, former,
                                        latter_tensor, if_former,
                                        if_latter)
            if repeat_time_last > 0:
                self._cmp_sel_found(repeat_time_last, repeat_time,
                                    rep_offset, sel_result, former,
                                    latter_tensor, if_former, if_latter)
        return sel_result

    def _cmp_sel_found(self, process_num, repeat_time, mode, sel_result,
                       former, latter, if_former, if_latter):
        cmp_mask = self.tinst.vcmp_eq(process_num, former[
            self.num_per_cmd * repeat_time], latter[
                                          self.num_per_cmd * repeat_time
                                          * mode],
                                      1, 1)
        self.tinst.vsel(process_num, 0,
                        sel_result[self.num_per_cmd * repeat_time], cmp_mask,
                        if_former[self.num_per_cmd * repeat_time],
                        if_latter[self.num_per_cmd * repeat_time], 1, 1, 1, 1,
                        1, 1)

    def compute_origin_index(self, process_num, grid, input_size):
        '''
        Main function which select a calculation method based on
        padding_mode&align_corners
        :param process_num:the length of process tensor
        :param grid: input tensor
        :param input_size: original length of current dim
        :return: restoration coordinates
        '''
        coord = self.grid_sampler_unnormalize(process_num, grid, input_size)
        if self.padding_mode == "border":
            coord = self.clip_coordinates(process_num, coord, input_size)
        elif self.padding_mode == "reflection":
            if self.align_corners:
                coord = self.reflect_coordinates(process_num, coord, 0,
                                                 2 * (input_size - 1))
            else:
                coord = self.reflect_coordinates(process_num, coord, -1,
                                                 2 * input_size - 1)
            coord = self.clip_coordinates(process_num, coord, input_size)
        else:
            pass

        return coord

    def grid_sampler_unnormalize(self, grid_num, grid, input_size):
        '''
        1.(grid + 1) / 2 * (input_size - 1) [align_corners is True]
        2.((grid + 1) * input_size - 1) / 2 [align_corners is False]
        '''
        buf_unnormal_all = {
            "grid_ub": AVecBuf(grid, grid_num, 0, self.cont, False,
                               self.num_per_cmd)}
        cmd_unnormal_tensor = [
            VecGCmd(cmd_name="vadds", dst_name="grid_ub", src0_name="grid_ub",
                    scalar=1)]
        if self.align_corners:
            cmd_unnormal_tensor.append(
                VecGCmd(cmd_name="vmuls", dst_name="grid_ub",
                        src0_name="grid_ub", scalar=(input_size - 1) / 2.0))
        else:
            cmd_unnormal_tensor.extend([
                VecGCmd(cmd_name="vmuls", dst_name="grid_ub",
                        src0_name="grid_ub", scalar=input_size),
                VecGCmd(cmd_name="vadds", dst_name="grid_ub",
                        src0_name="grid_ub", scalar=-1),
                VecGCmd(cmd_name="vmuls", dst_name="grid_ub",
                        src0_name="grid_ub", scalar=0.5)])
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_unnormal_all,
                                    cmd_unnormal_tensor, "grid_ub")
        return grid

    def reflect_coordinates(self, grid_num, grid, twice_low, twice_high):
        '''
        if the padding mode is reflection,The location which has beyond the
        boundary is obtained
        through reflection mode.
        :param grid_num: the length of input tensor
        :param grid:input tensor
        :param twice_low:low value
        :param twice_high:high value
        '''
        min_value = twice_low / 2.0
        span = (twice_high - twice_low) / 2.0
        with self.tinst.new_stmt_scope():
            flips = self.tinst.Tensor("float32", (grid_num,),
                                      self.tik.scope_ubuf, "flips")
            extra = self.tinst.Tensor("float32", (grid_num,),
                                      self.tik.scope_ubuf, "flips")
            buf_reflect_all = {
                "grid": AVecBuf(grid, grid_num, 0, self.cont, False,
                                self.num_per_cmd),
                "flips": AVecBuf(flips, grid_num, 0, self.cont, False,
                                 self.num_per_cmd),
                "extra": AVecBuf(extra, grid_num, 0, self.cont, False,
                                 self.num_per_cmd), }

            if twice_low == twice_high:
                cmd_reflect_tensor = [
                    VecGCmd(cmd_name="vmuls", dst_name="grid",
                            src0_name="grid", scalar=0)]
                VecGExecutor.exec_vec_g_cmd(self.cont, buf_reflect_all,
                                            cmd_reflect_tensor, "grid")

            else:
                cmd_reflect_tensor = [
                    VecGCmd(cmd_name="vadds", dst_name="grid",
                            src0_name="grid", scalar=-min_value),
                    VecGCmd(cmd_name="vabs", dst_name="grid",
                            src0_name="grid")]
                VecGExecutor.exec_vec_g_cmd(self.cont, buf_reflect_all,
                                            cmd_reflect_tensor, "grid")
                # extra is grid mod span, flips is grid div span
                flips, extra = self._modulus(grid_num, grid, span, flips,
                                             extra)
                # flips is flips mod 2, grid is flips div 2
                grid, flips = self._modulus(grid_num, flips, 2, grid, flips)

                cmd_reflect_tensor = [
                    VecGCmd(cmd_name="vmuls", dst_name="grid",
                            src0_name="extra", scalar=-1),
                    VecGCmd(cmd_name="vadds", dst_name="grid",
                            src0_name="grid", scalar=min_value + span),
                    VecGCmd(cmd_name="vadds", dst_name="extra",
                            src0_name="extra", scalar=min_value), ]
                VecGExecutor.exec_vec_g_cmd(self.cont, buf_reflect_all,
                                            cmd_reflect_tensor, "grid")

                grid = self.cmp_sel(grid_num, grid, flips, 0, extra, grid)

        return grid

    def clip_coordinates(self, process_num, coord, size):
        # calculate mini(size - 1, maxi(0, coord))
        buf_clip_all = {
            "coord": AVecBuf(coord, process_num, 0, self.cont, False,
                             self.num_per_cmd), }
        cmd_clip_tensor = [
            VecGCmd("vmaxs", dst_name="coord", src0_name="coord",
                    scalar=0),
            VecGCmd("vmins", dst_name="coord", src0_name="coord",
                    scalar=size - 1), ]
        VecGExecutor.exec_vec_g_cmd(self.cont, buf_clip_all,
                                    cmd_clip_tensor, "coord")

        return coord

    def _modulus(self, process_num, dividend, divisor, result_quotient,
                 result_remainder):
        '''
        Modulus operation result_quotient=dividend/divisor,
        result_remainder=dividend%divisor
        :param process_num:the length of input tensor
        :param dividend:input tensor
        :param divisor:scalar
        :param result_quotient:return quotient
        :param result_remainder:return remainder
        '''
        with self.tinst.new_stmt_scope():
            tensor_temp_int32 = self.tinst.Tensor("int32", (process_num,),
                                                  self.tik.scope_ubuf,
                                                  "tensor_temp_int32")

            integer_format = self.tinst.Tensor("float32", (process_num,),
                                               self.tik.scope_ubuf,
                                               "integer_format")
            buf_modulus_all = {
                "result_quotient": AVecBuf(result_quotient, process_num, 0,
                                           self.cont, False, self.num_per_cmd),
                "result_remainder": AVecBuf(result_remainder, process_num, 0,
                                            self.cont, False,
                                            self.num_per_cmd),
                "tensor_temp_int32": AVecBuf(tensor_temp_int32, process_num, 0,
                                             self.cont, False,
                                             self.num_per_cmd),
                "dividend": AVecBuf(dividend, process_num, 0, self.cont, False,
                                    self.num_per_cmd),
                "integer_format": AVecBuf(integer_format, process_num, 0,
                                          self.cont, False,
                                          self.num_per_cmd), }
            cmd_modulus_tensor = [VecGCmd("vmuls", dst_name="result_quotient",
                                          src0_name="dividend",
                                          scalar=1.0 / divisor),
                                  VecGCmd("vconv", "tensor_temp_int32",
                                          "result_quotient",
                                          round_mode="floor"),
                                  VecGCmd("vconv", "result_quotient",
                                          "tensor_temp_int32", round_mode=""),
                                  VecGCmd("vmuls", dst_name="integer_format",
                                          src0_name="result_quotient",
                                          scalar=divisor),
                                  VecGCmd("vsub", dst_name="result_remainder",
                                          src0_name="dividend",
                                          src1_name="integer_format"), ]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_modulus_all,
                                        cmd_modulus_tensor, "integer_format")

        return result_quotient, result_remainder
