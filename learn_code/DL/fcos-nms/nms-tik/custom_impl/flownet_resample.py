import numpy as np
from te import tik
from util import OpLog as log


class InterpLayer:
    """
     resample info class
    """
    def __init__(self, shape, add_num, h_proportion, w_proportion,
                 weight_y, weight_x, kernelname):
        """
        init param
        """
        self.shape = shape
        self.add_num = add_num
        self.h_proportion = h_proportion
        self.w_proportion = w_proportion
        self.weight_y = weight_y
        self.weight_x = weight_x
        self.kernelname = kernelname
        self.tik_inst = tik.Tik(tik.Dprofile())
        self.last_row_out = 512
        self.second_last_row_out = 511
        self.third_last_row_out = 510
        self.data_in = self.tik_inst.Tensor("float16", 
           (1, 1, shape[2], shape[3], 16), name="data_in", scope=tik.scope_gm)
        self.data_out = self.tik_inst.Tensor("float16", 
           [1, 1, shape[2] * self.h_proportion, \
            shape[3] * self.h_proportion, 16],
           name="data_out", scope=tik.scope_gm)
        self.ub_left = self.tik_inst.Tensor("float16",
           (5, shape[3] + 1, 16), name="ub_left", scope=tik.scope_ubuf)
        self.data_ub_left_1 = self.tik_inst.Tensor("float16",
           (1, shape[3] + 1, 16), name="data_ub_left_1", scope=tik.scope_ubuf)
        self.data_ub_right = self.tik_inst.Tensor("float16",
           (5, shape[3] + 1, 16), name="data_ub_right", scope=tik.scope_ubuf)
        self.data_ub_right_1 = self.tik_inst.Tensor("float16", 
           (1, shape[3] + 1, 16), name="data_ub_righ_1", scope=tik.scope_ubuf)
        self.data_ub = self.tik_inst.Tensor("float16",
           (1, shape[3], 16), name="data_ub", scope=tik.scope_ubuf)
        self.eltwise_data_ub = self.tik_inst.Tensor("float16",
           (1, shape[3], 16), name="eltwise_data_ub", scope=tik.scope_ubuf)
        self.data_out_ub = self.tik_inst.Tensor("float16", 
           (shape[3] * 4, 16), name="data_out_ub", scope=tik.scope_ubuf)
        self.data_out_ub_1 = self.tik_inst.Tensor("float16", 
           (shape[3] * 4, 16), name="data_out_ub_1", scope=tik.scope_ubuf)
        self.data_out_one = self.tik_inst.Tensor("float16", 
           (shape[3] * 4 + 4, 16), name="data_out_one", scope=tik.scope_ubuf)
        self.data_out_two = self.tik_inst.Tensor("float16",
           (shape[3] * 4 + 4, 16), name="data_out_two", scope=tik.scope_ubuf)
        self.weight_y_ub = self.tik_inst.Tensor("float16",
           (16,), name="weight_y_ub", scope=tik.scope_ubuf)
        self.weight_x_ub = self.tik_inst.Tensor("float16",
           (16,), name="weight_x_ub", scope=tik.scope_ubuf)
        self.left_add_right = self.tik_inst.Tensor("float16",
           (1, shape[3] + 1, 16), name="left_add_right", scope=tik.scope_ubuf)
        self.weight_y_scalar = self.tik_inst.Scalar(dtype="float16",
                                                    name="weight_y_scalar")
        self.weight_x_scalar = self.tik_inst.Scalar(dtype="float16",
                                                    name="weight_x_scalar")


    def weight_set(self):
        """
        set weight into ub
        """
        for idx in range(8):
            self.weight_y_scalar.set_as(self.weight_y[idx])
            self.weight_y_ub[idx].set_as(self.weight_y_scalar)
            self.weight_x_scalar.set_as(self.weight_x[idx])
            self.weight_x_ub[idx].set_as(self.weight_x_scalar)


    def zero_row_cal_eltwise(self):
        """
        cal eltwise of first row
        """
        with self.tik_inst.for_range(0, 5) as idx:
            self.tik_inst.data_move(self.data_ub[0, 0, 0],
                                    self.data_in[0, 0, idx, 0, 0],
                                    0, 1, self.shape[3], 0, 0)
            self.tik_inst.vmuls(128, self.eltwise_data_ub, self.data_ub,
                                self.add_num, self.shape[3] // 8,
                                1, 1, 8, 8)
            self.tik_inst.data_move(self.ub_left[idx, 1, 0],
                                    self.eltwise_data_ub, 0, 1,
                                    self.shape[3], 0, 0)
            self.tik_inst.data_move(self.data_ub_right[idx, 0, 0],
                                    self.eltwise_data_ub, 0, 1,
                                    self.shape[3], 0, 0)


    def gt_zero_cal_eltwise(self, row):
        """
        cal eltwise of the row which is great than zero
        """
        with self.tik_inst.if_scope(row == 23):
            with self.tik_inst.for_range(0, 4) as idx:
                self.tik_inst.data_move(self.data_ub[0, 0, 0],
                                       self.data_in[0, 0, row * 4 + idx, 0, 0],
                                       0, 1, self.shape[3], 0, 0)
                self.tik_inst.vmuls(128, self.eltwise_data_ub, self.data_ub,
                                    self.add_num, self.shape[3] * 16 // 128,
                                    1, 1, 8, 8)
                self.tik_inst.data_move(self.ub_left[idx, 1, 0],
                                        self.eltwise_data_ub, 0, 1,
                                        self.shape[3], 0, 0)
                self.tik_inst.data_move(self.data_ub_right[idx, 0, 0],
                                        self.eltwise_data_ub, 0, 1,
                                        self.shape[3], 0, 0)
            self.tik_inst.vector_dup(128, self.ub_left[4, 1, 0],
                                     0, 1 * 16, 1, 8)
            self.tik_inst.vector_dup(128, self.data_ub_right[4, 0, 0],
                                     0, 1 * 16, 1, 8)
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, 5) as idx:
                self.tik_inst.data_move(self.data_ub[0, 0, 0], 
                                       self.data_in[0, 0, row * 4 + idx, 0, 0],
                                       0, 1, self.shape[3], 0, 0)
                self.tik_inst.vmuls(128, self.eltwise_data_ub, self.data_ub,
                                    self.add_num,
                                    self.shape[3] * 16 // 128, 1, 1, 8, 8)
                self.tik_inst.data_move(self.ub_left[idx, 1, 0],
                                        self.eltwise_data_ub, 0, 1,
                                        self.shape[3], 0, 0)
                self.tik_inst.data_move(self.data_ub_right[idx, 0, 0],
                                        self.eltwise_data_ub, 0, 1,
                                        self.shape[3], 0, 0)


    def zero_row_cal_repeat_zero_interp(self, temp, temp_y):
        """
        cal the first row
        """
        with self.tik_inst.for_range(0, 4) as w_repeat:
            temp.set_as(self.weight_x_ub[w_repeat * 2 + 1])
            self.tik_inst.vmuls(128, self.data_ub_right_1[0, 0, 0],
                                self.data_ub_right[0, 0, 0], temp,
                                1 * 16, 1, 1, 8, 8)
            temp.set_as(self.weight_x_ub[w_repeat * 2])
            self.tik_inst.vmuls(128, self.data_ub_left_1[0, 1, 0],
                                self.ub_left[0, 1, 0], temp,
                                1 * 16, 1, 1, 8, 8)
            self.tik_inst.vadd(128, self.left_add_right[0, 0, 0],
                               self.data_ub_left_1[0, 0, 0],
                               self.data_ub_right_1[0, 0, 0], 1 * 16,
                               1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(16, self.left_add_right[0, 128, 0],
                               self.data_ub_left_1[0, 128, 0],
                               self.data_ub_right_1[0, 128, 0], 1, 1, 1,
                               1, 1, 1, 1)
            self.tik_inst.data_move(self.data_out_two[w_repeat, 0],
                                    self.left_add_right, 0, 129, 1, 0, 3)
        with self.tik_inst.for_range(0, 2) as h_repeat:
            temp_y.set_as(self.weight_y_ub[h_repeat * 2 + 1 + 4])
            self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                self.data_out_two[2, 0], temp_y,
                                self.last_row_out * 16 // 128, 1, 1, 8, 8)
            with self.tik_inst.if_scope(h_repeat == 0):
                self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                    self.data_out_ub[0, 0], 1 / 0.625,
                                    self.last_row_out * 16 // 128,
                                    1, 1, 8, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                    self.data_out_ub[0, 0],
                                    1 / 0.875, self.last_row_out * 16 // 128,
                                    1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[0, 0],
                                self.data_out_ub[0, 0], 1 / 0.625, 1,
                                1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[1, 0],
                                self.data_out_ub[1, 0], 1 / 0.875,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[self.third_last_row_out,\
                                0], self.data_out_ub[self.third_last_row_out,\
                                0], 1 / 0.875, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[self.second_last_row_out,\
                                0], self.data_out_ub[self.second_last_row_out,\
                                0], 1 / 0.625, 1, 1, 1, 8, 8)
            self.tik_inst.data_move(self.data_out[0, 0, h_repeat, 0, 0],
                                    self.data_out_ub[0, 0], 0, 1,
                                    self.last_row_out, 0, 0)


    def cal_zero_row_cal_repeat_gt_zero_wrepeat(self, repeat, temp, w_repeat):
        """
        cal zero row cal repeat gt zero wrepeat
        """
        with self.tik_inst.for_range(0, 2) as c_1:
            temp.set_as(self.weight_x_ub[w_repeat * 2 + 1])
            self.tik_inst.vmuls(128, self.data_ub_right_1[0, 0, 0],
                                self.data_ub_right[repeat - 1 + c_1, 0, 0],
                                temp, 1 * 16, 1, 1, 8, 8)
            temp.set_as(self.weight_x_ub[w_repeat * 2])
            self.tik_inst.vmuls(128, self.data_ub_left_1[0, 1, 0],
                                self.ub_left[repeat - 1 + c_1, 1, 0],
                                temp, 1 * 16, 1, 1, 8, 8)
            self.tik_inst.vadd(128, self.left_add_right[0, 0, 0],
                               self.data_ub_left_1[0, 0, 0],
                               self.data_ub_right_1[0, 0, 0], 16, 1, 
                               1, 1, 8, 8, 8)
            self.tik_inst.vadd(16, self.left_add_right[0, 128, 0],
                               self.data_ub_left_1[0, 128, 0],
                               self.data_ub_right_1[0, 128, 0],
                               1, 1, 1, 1, 1, 1, 1)
            with self.tik_inst.if_scope(c_1 == 0):
                self.tik_inst.data_move(self.data_out_one[w_repeat, 0],
                                        self.left_add_right,
                                        0, 129, 1, 0, 3)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.data_out_two[w_repeat, 0],
                                        self.left_add_right,
                                        0, 129, 1, 0, 3)


    def zero_row_cal_repeat_gt_zero_interp(self, repeat, temp, temp_y, row):
        """
        cal the first row
        """
        with self.tik_inst.for_range(0, 4) as w_repeat:
            self.cal_zero_row_cal_repeat_gt_zero_wrepeat(repeat, 
                                                         temp, w_repeat)
        with self.tik_inst.for_range(0, 4) as h_repeat:
            temp_y.set_as(self.weight_y_ub[h_repeat * 2])
            self.tik_inst.vmuls(128, self.data_out_ub_1[0, 0],
                                self.data_out_one[2, 0], temp_y,
                                self.last_row_out * 16 // 128, 1, 1, 8, 8)
            temp_y.set_as(self.weight_y_ub[h_repeat * 2 + 1])
            self.tik_inst.vmuls(128, self.data_out_ub[0, 0], 
                                self.data_out_two[2, 0], temp_y,
                                self.last_row_out * 16 // 128, 1, 1, 8, 8)
            self.tik_inst.vadd(128, self.data_out_ub[0, 0], 
                              self.data_out_ub[0, 0], self.data_out_ub_1[0, 0],
                              self.last_row_out * 16 // 128, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[0, 0],
                                self.data_out_ub[0, 0], 1 / 0.625,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[1, 0],
                                self.data_out_ub[1, 0], 1 / 0.875,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[self.third_last_row_out,\
                                0], self.data_out_ub[self.third_last_row_out,\
                                0], 1 / 0.875, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[self.second_last_row_out,\
                                0], self.data_out_ub[self.second_last_row_out,\
                                0], 1 / 0.625, 1, 1, 1, 8, 8)
            self.tik_inst.data_move(self.data_out[0, 0,
                                    (row + repeat) * 4 + h_repeat - 2, 0, 0],
                                    self.data_out_ub, 0, 1, self.last_row_out,
                                    0, 0)


    def zero_row_cal_interp(self, row):
        """
        cal the first row
        """
        with self.tik_inst.for_range(0, 5) as repeat:
            temp = self.tik_inst.Scalar("float16", name="temp")
            temp_y = self.tik_inst.Scalar("float16", name="temp_y")
            with self.tik_inst.if_scope(repeat == 0):
                self.zero_row_cal_repeat_zero_interp(temp, temp_y)
            with self.tik_inst.else_scope():
                self.zero_row_cal_repeat_gt_zero_interp(repeat, temp,
                                                        temp_y, row)


    def last_row_repeat_equal_3(self, row, repeat):
        """
        cal last row repeat equal 3
        """
        with self.tik_inst.for_range(0, 2) as h_repeat:
            temp_y = self.tik_inst.Scalar("float16", name="temp_y")
            temp_y.set_as(self.weight_y_ub[h_repeat * 2])
            self.tik_inst.vmuls(128, self.data_out_ub_1[0, 0],
                                self.data_out_one[2, 0],
                                temp_y, self.last_row_out // 8, 1, 1, 8, 8)
            temp_y.set_as(self.weight_y_ub[h_repeat * 2 + 1])
            self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                self.data_out_two[2, 0],
                                temp_y, self.last_row_out // 8, 1, 1, 8, 8)
            self.tik_inst.vadd(128, self.data_out_ub[0, 0],
                    self.data_out_ub[0, 0], self.data_out_ub_1[0, 0],
                    self.last_row_out * 16 // 128, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[0, 0],
                    self.data_out_ub[0, 0], 1 / 0.625, 1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[1, 0],
                                self.data_out_ub[1, 0], 1 / 0.875,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16, self.data_out_ub[ \
                                self.third_last_row_out, 0],
                                self.data_out_ub[ \
                                self.third_last_row_out, 0], 1 / 0.875,
                                1, 1, 1, 8, 8)
            self.tik_inst.vmuls(16,
                                self.data_out_ub[self.second_last_row_out, 0],
                                self.data_out_ub[self.second_last_row_out, 0],
                                1 / 0.625, 1, 1, 1, 8, 8)
            with self.tik_inst.if_scope(h_repeat == 0):
                self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                    self.data_out_ub[0, 0], 1 / 0.875,
                                    self.last_row_out * 16 // 128, 1, 1, 8, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                   self.data_out_ub[0, 0], 1 / 0.625,
                                   self.last_row_out * 16 // 128, 1, 1, 8, 8)
            self.tik_inst.data_move(self.data_out[0, 0, row * 16 + \
                                    repeat * 4 + h_repeat + 2, 0, 0],
                                    self.data_out_ub[0, 0], 0, 1,
                                    self.last_row_out, 0, 0)


    def last_row_interp(self, row, repeat):
        """
        cal the last row
        """
        with self.tik_inst.if_scope(row == 23):
            with self.tik_inst.if_scope(repeat == 3):
                self.last_row_repeat_equal_3(row, repeat)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, 4) as h_repeat:
                    temp_y = self.tik_inst.Scalar("float16", name="temp_y")
                    temp_y.set_as(self.weight_y_ub[h_repeat * 2])
                    self.tik_inst.vmuls(128, self.data_out_ub_1[0, 0],
                                        self.data_out_one[2, 0], temp_y,
                                        self.last_row_out * 16 // 128,
                                        1, 1, 8, 8)
                    temp_y.set_as(self.weight_y_ub[h_repeat * 2 + 1])
                    self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                        self.data_out_two[2, 0], temp_y,
                                        self.last_row_out * 16 // 128, 1,
                                        1, 8, 8)
                    self.tik_inst.vadd(128, self.data_out_ub[0, 0],
                                       self.data_out_ub[0, 0],
                                       self.data_out_ub_1[0, 0],
                                       self.last_row_out * 16 // 128,
                                       1, 1, 1, 8, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[0, 0],
                                        self.data_out_ub[0, 0], 1 / 0.625,
                                        1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[1, 0],
                                        self.data_out_ub[1, 0], 1 / 0.875,
                                        1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[ \
                                        self.third_last_row_out, 0],
                                        self.data_out_ub[ \
                                        self.third_last_row_out, 0],
                                        1 / 0.875, 1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[ \
                                        self.second_last_row_out, 0],
                                        self.data_out_ub[ \
                                        self.second_last_row_out, 0],
                                        1 / 0.625, 1, 1, 1, 8, 8)
                    self.tik_inst.data_move(self.data_out[0, 0, row * 16 + \
                                            repeat * 4 + h_repeat + 2, 0, 0],
                                            self.data_out_ub, 0, 1, 
                                            self.last_row_out, 0, 0)


    def gt_zero_cal_the_first_two_times(self, w_repeat, repeat):
        """
        cal zero the first two times
        """
        with self.tik_inst.for_range(0, 2) as c_0:
            temp = self.tik_inst.Scalar("float16", name="temp")
            temp.set_as(self.weight_x_ub[w_repeat * 2 + 1])
            self.tik_inst.vmuls(128, self.data_ub_right_1[0, 0, 0],
                                self.data_ub_right[repeat + c_0, 0, 0],
                                temp, 1 * 16, 1, 1, 8, 8)
            temp.set_as(self.weight_x_ub[w_repeat * 2])
            self.tik_inst.vmuls(128, self.data_ub_left_1[0, 1, 0],
                                self.ub_left[repeat + c_0, 1, 0],
                                temp, 1 * 16, 1, 1, 8, 8)
            self.tik_inst.vadd(128, 
                               self.left_add_right[0, 0, 0],
                               self.data_ub_right_1[0, 0, 0],
                               self.data_ub_left_1[0, 0, 0],
                               1 * 16, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vadd(16,
                               self.left_add_right[0, 128, 0],
                               self.data_ub_right_1[0, 128, 0],
                               self.data_ub_left_1[0, 128, 0],
                               1, 1, 1, 1, 1, 1, 1)
            with self.tik_inst.if_scope(c_0 == 0):
                self.tik_inst.data_move(self.data_out_one[w_repeat, 0],
                                        self.left_add_right,
                                        0, 129, 1, 0, 3)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.data_out_two[w_repeat, 0],
                                        self.left_add_right,
                                        0, 129, 1, 0, 3)


    def gt_zero_cal_interp(self, row):
        """
        cal rows which is greater than zero
        """
        with self.tik_inst.for_range(0, 4) as repeat:
            with self.tik_inst.for_range(0, 4) as w_repeat:
                self.gt_zero_cal_the_first_two_times(w_repeat, repeat)
            self.last_row_interp(row, repeat)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(0, 4) as h_repeat:
                    temp_y = self.tik_inst.Scalar("float16", name="temp_y")
                    temp_y.set_as(self.weight_y_ub[h_repeat * 2])
                    self.tik_inst.vmuls(128, self.data_out_ub_1[0, 0],
                                        self.data_out_one[2, 0],
                                        temp_y, self.last_row_out * 16 // 128,
                                        1, 1, 8, 8)
                    temp_y.set_as(self.weight_y_ub[h_repeat * 2 + 1])
                    self.tik_inst.vmuls(128, self.data_out_ub[0, 0],
                                        self.data_out_two[2, 0], temp_y,
                                        self.last_row_out * 16 // 128,
                                        1, 1, 8, 8)
                    self.tik_inst.vadd(128, self.data_out_ub[0, 0],
                                       self.data_out_ub[0, 0],
                                       self.data_out_ub_1[0, 0],
                                       self.last_row_out * 16 // 128,
                                       1, 1, 1, 8, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[0, 0],
                                        self.data_out_ub[0, 0],
                                        1 / 0.625, 1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[1, 0],
                                        self.data_out_ub[1, 0],
                                        1 / 0.875, 1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[ \
                                        self.third_last_row_out, 0],
                                        self.data_out_ub[ \
                                        self.third_last_row_out, 0], 1 / 0.875,
                                        1, 1, 1, 8, 8)
                    self.tik_inst.vmuls(16, self.data_out_ub[ \
                                        self.second_last_row_out, 0],
                                        self.data_out_ub[ \
                                        self.second_last_row_out, 0],
                                        1 / 0.625, 1, 1, 1, 8, 8)
                    self.tik_inst.data_move(self.data_out[0, 0, row * 16 +  \
                                            repeat * 4 + h_repeat + 2, 0, 0],
                                            self.data_out_ub, 0, 1,
                                            self.last_row_out, 0, 0)


    def interp_layer_plugin(self):
        """
        shape:the size of feature map before resample
        add_num:eltwise operation's add number
        h_proportion:The vertical axis direction of expanding scale
        w_proportion:Transverse direction of expanding scale
        weight_y:The weight of the corresponding columns of a single cycle
        weight_x:The weight of line corresponds to a single cycle
        """
        self.tik_inst.vector_dup(16, self.data_ub_left_1[0, 0, 0], 0, 1, 1, 1)
        self.tik_inst.vector_dup(16, self.data_ub_right_1[0, self.shape[3], 0],
                                 0, 1, 1, 1)
        with self.tik_inst.for_range(0, 5) as t_0:
            self.tik_inst.vector_dup(16, self.ub_left[t_0, 0, 0],
                                     0, 1, 1, 1)
            self.tik_inst.vector_dup(16, 
                                     self.data_ub_right[t_0, self.shape[3], 0],
                                     0, 1, 1, 1)
        self.weight_set()
        with self.tik_inst.for_range(0, 24) as row:
            with self.tik_inst.if_scope(row == 0):
                self.zero_row_cal_eltwise()
            with self.tik_inst.else_scope():
                self.gt_zero_cal_eltwise(row)
            with self.tik_inst.if_scope(row == 0):
                self.zero_row_cal_interp(row)
            with self.tik_inst.else_scope():
                self.gt_zero_cal_interp(row)
        self.tik_inst.BuildCCE(self.kernelname, inputs=[self.data_in],
                               outputs=[self.data_out], enable_l2=True)
        return self.tik_inst


def parm_check(add_num, h_proportion, w_proportion, kernelname):
    """
    check param
    add_num: float
    h_proportion:float
    w_proportion:float
    kernelname:string
    """
    if not isinstance(add_num, float):
        raise RuntimeError("add_num need float")
    log.check_eq(h_proportion, 4, "h_proportion should be equal to 4")
    log.check_eq(w_proportion, 4, "w_proportion should be equal to 4")
    log.check_le(len(kernelname), 125, "the length of kernel name \
                 should be less than or equal to 125")


def flownet_resample(input_shape, output_shape, add_num, h_proportion,
                     w_proportion, kernelname="flownet_resample"):
    """
    the entry function of resample
    """
    shape = input_shape.get('shape')
    shape = [shape[0], 2, shape[2], shape[3]]
    parm_check(add_num, h_proportion, w_proportion, kernelname)
    weight_x = [0.875, 0.125, 0.625, 0.375, 0.375, 0.625, 0.125, 0.875]
    weight_y = [0.875, 0.125, 0.625, 0.375, 0.375, 0.625, 0.125, 0.875]
    interp = InterpLayer(shape, add_num, h_proportion, w_proportion, 
                         weight_y, weight_x, kernelname)
    return interp.interp_layer_plugin()

