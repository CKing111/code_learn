import numpy as np
from te import tik
from common import vec_muls
from common import vec_add
from common import vec_adds
from common import vec_exp
from common import vec_dup
from common import vconv_int32tofp16
from util import OpLog as log


class Coordinate:
    def __init__(self, x_start, x_strides, y_start, y_strides, loc_shape,
                 anchor_dims):
        self.INT32 = 'int32'
        self.FLOAT16 = 'float16'
        self.x_start = x_start
        self.y_start = y_start
        self.loc_shape = loc_shape
        self.tik_inst = tik.Tik(tik.Dprofile())
        self.xstep = x_strides
        self.ystep = y_strides
        self.anchor_dims = np.array(list(anchor_dims))
        self.anchor_d = np.sqrt(
            self.anchor_dims[:, 0] * self.anchor_dims[:, 0] + 
            self.anchor_dims[:, 1] * self.anchor_dims[:, 1])
        self.anchor_d = np.concatenate(
            [np.tile(self.anchor_d[:, np.newaxis], [1, 2]), self.anchor_dims],
            axis=-1)
        self.anchor_d[:, 5] = self.anchor_d[:, 5] + self.anchor_d[:, 4] * 0.5
        # float16每个元素占2个字节
        self.MAX_COMPUTE_SIZE = 200 * 1024 // 2
        self.box_num = self.loc_shape[1] // 7
        loc_num = self.loc_shape[0] * self.loc_shape[1] * self.loc_shape[2] * \
                  self.loc_shape[3]
        loc_num_align16 = (self.loc_shape[2] * self.loc_shape[3] + 15) // 16
        self.loc_data_in = self.tik_inst.Tensor(dtype=self.FLOAT16,
                                                shape=(loc_num + 16,),
                                                scope=tik.scope_gm,
                                                name="loc_data_in")
        self.loc_data_out = self.tik_inst.Tensor(
            dtype=self.FLOAT16, shape=(self.box_num, 7, loc_num_align16 * 16),
            scope=tik.scope_gm, name="loc_data_out")

    def compute_x_coor(self, x_data_ub, w_align16, row, box, x_scalar,
                       data_idx_w_ub):
        x_tmp_ub = self.tik_inst.Tensor(self.FLOAT16, (w_align16 * 16,),
                                        name="x_tmp_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(
            x_tmp_ub, self.loc_data_in[box * 7 * self.loc_shape[2] * 
                self.loc_shape[3] + row * self.loc_shape[3]],
            0, 1, w_align16, 1, 1)
        vec_muls(self.tik_inst, self.FLOAT16, x_data_ub, x_tmp_ub, x_scalar)
        vec_add(self.tik_inst, self.FLOAT16, x_data_ub, x_data_ub,
                data_idx_w_ub)
        self.tik_inst.data_move(
            self.loc_data_out[box, 0, row * self.loc_shape[3]], x_data_ub, 0,
            1, w_align16, 1, 1)

    def compute_y_coor(self, y_data_ub, w_align16, row, box, y_scalar,
                       data_idx_h_ub):
        y_tmp_ub = self.tik_inst.Tensor(self.FLOAT16, (w_align16 * 16,),
                                        name="y_tmp_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(
            y_tmp_ub, self.loc_data_in[(box * 7 + 1) * self.loc_shape[2] * 
                self.loc_shape[3] + row * self.loc_shape[3]],
            0, 1, w_align16, 1, 1)
        vec_muls(self.tik_inst, self.FLOAT16, y_data_ub, y_tmp_ub, y_scalar)
        vec_add(self.tik_inst, self.FLOAT16, y_data_ub, y_data_ub,
                data_idx_h_ub)
        self.tik_inst.data_move(
            self.loc_data_out[box, 1, row * self.loc_shape[3]], y_data_ub, 0,
            1, w_align16, 1, 1)

    def compute_z_coor(self, box, z_scalar, addz_scalar):
        hw_align = (self.loc_shape[2] * self.loc_shape[3] + 15) // 16
        if hw_align * 16 <= self.MAX_COMPUTE_SIZE:
            z_tmp_ub = self.tik_inst.Tensor(self.FLOAT16, (hw_align * 16,),
                                            name="z_tmp_ub",
                                            scope=tik.scope_ubuf)
            self.tik_inst.data_move(z_tmp_ub, self.loc_data_in[
                (box * 7 + 2) * self.loc_shape[2] * self.loc_shape[3]], 0, 1,
                                    hw_align, 1, 1)
            vec_muls(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                     z_scalar)
            vec_adds(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                     addz_scalar)
            self.tik_inst.data_move(self.loc_data_out[box, 2, 0], z_tmp_ub,
                                    0, 1, hw_align, 1, 1)
        if hw_align * 16 > self.MAX_COMPUTE_SIZE:
            repeat_time = hw_align * 16 // self.MAX_COMPUTE_SIZE
            tail = (hw_align * 16) % self.MAX_COMPUTE_SIZE
            z_tmp_ub = self.tik_inst.Tensor(self.FLOAT16,
                                            (self.MAX_COMPUTE_SIZE,),
                                            name="z_tmp_ub",
                                            scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, repeat_time) as rep:
                self.tik_inst.data_move(
                    z_tmp_ub, self.loc_data_in[(box * 7 + 2) * 
                        self.loc_shape[2] * self.loc_shape[3] + 
                        rep * self.MAX_COMPUTE_SIZE], 0, 1,
                    self.MAX_COMPUTE_SIZE // 16, 1, 1)
                vec_muls(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         z_scalar)
                vec_adds(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         addz_scalar)
                self.tik_inst.data_move(
                    self.loc_data_out[box, 2, rep * self.MAX_COMPUTE_SIZE],
                    z_tmp_ub, 0, 1, self.MAX_COMPUTE_SIZE // 16, 1, 1)
            if tail > 0:
                self.tik_inst.data_move(
                    z_tmp_ub, self.loc_data_in[(box * 7 + 2) * 
                        self.loc_shape[2] * self.loc_shape[3] + 
                        repeat_time * self.MAX_COMPUTE_SIZE],
                    0, 1, tail // 16, 1, 1)
                vec_muls(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         z_scalar)
                vec_adds(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         addz_scalar)
                self.tik_inst.data_move(
                    self.loc_data_out[
                        box, 2, repeat_time * self.MAX_COMPUTE_SIZE],
                    z_tmp_ub, 0, 1, tail // 16, 1, 1)

    def compute_lwh(self, box, size, lwh_scalar):
        hw_align = (self.loc_shape[2] * self.loc_shape[3] + 15) // 16
        if hw_align * 16 <= self.MAX_COMPUTE_SIZE:
            z_tmp_ub = self.tik_inst.Tensor(self.FLOAT16, (hw_align * 16,),
                                            name="z_tmp_ub",
                                            scope=tik.scope_ubuf)
            self.tik_inst.data_move(z_tmp_ub, self.loc_data_in[
                (box * 7 + 3 + size) * self.loc_shape[2] * self.loc_shape[3]],
                                    0, 1, hw_align, 1, 1)
            vec_exp(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub)
            vec_muls(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                     lwh_scalar)
            self.tik_inst.data_move(self.loc_data_out[box, size + 3, 0],
                                    z_tmp_ub, 0, 1, hw_align, 1, 1)
        if hw_align * 16 > self.MAX_COMPUTE_SIZE:
            repeat_time = hw_align * 16 // self.MAX_COMPUTE_SIZE
            tail = (hw_align * 16) % self.MAX_COMPUTE_SIZE
            z_tmp_ub = self.tik_inst.Tensor(
                self.FLOAT16, (self.MAX_COMPUTE_SIZE,),
                name="z_tmp_ub", scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, repeat_time) as rep:
                self.tik_inst.data_move(
                    z_tmp_ub, self.loc_data_in[
                        (box * 7 + 3 + size) * self.loc_shape[2] *
                        self.loc_shape[3] + rep * self.MAX_COMPUTE_SIZE],
                    0, 1, self.MAX_COMPUTE_SIZE // 16, 1, 1)
                vec_exp(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub)
                vec_muls(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         lwh_scalar)
                self.tik_inst.data_move(
                    self.loc_data_out[
                        box, size + 3, self.loc_shape[2] * self.loc_shape[
                            3] + rep * self.MAX_COMPUTE_SIZE],
                    z_tmp_ub, 0, 1, self.MAX_COMPUTE_SIZE // 16, 1, 1)
            if tail > 0:
                self.tik_inst.data_move(
                    z_tmp_ub, self.loc_data_in[(box * 7 + 3 + size) *
                        self.loc_shape[2] * self.loc_shape[3] + 
                        repeat_time * self.MAX_COMPUTE_SIZE],
                    0, 1, tail // 16, 1, 1)
                vec_exp(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub)
                vec_muls(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         lwh_scalar)
                self.tik_inst.data_move(
                    self.loc_data_out[
                        box, size + 3, repeat_time * self.MAX_COMPUTE_SIZE],
                    z_tmp_ub, 0, 1, tail // 16, 1, 1)

    def compute_angle(self, box, angle_scalar):
        hw_align = (self.loc_shape[2] * self.loc_shape[3] + 15) // 16
        if hw_align * 16 <= self.MAX_COMPUTE_SIZE:
            z_tmp_ub = self.tik_inst.Tensor(self.FLOAT16, (hw_align * 16,),
                                            name="z_tmp_ub",
                                            scope=tik.scope_ubuf)
            self.tik_inst.data_move(
                z_tmp_ub, self.loc_data_in[(box * 7 + 6) * self.loc_shape[2] *
                    self.loc_shape[3]], 0, 1, hw_align, 1, 1)
            vec_adds(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                     angle_scalar)
            self.tik_inst.data_move(self.loc_data_out[box, 6, 0],
                                    z_tmp_ub, 0, 1, hw_align, 1, 1)
        if hw_align * 16 > self.MAX_COMPUTE_SIZE:
            repeat_time = hw_align * 16 // self.MAX_COMPUTE_SIZE
            tail = (hw_align * 16) % self.MAX_COMPUTE_SIZE
            z_tmp_ub = self.tik_inst.Tensor(self.FLOAT16,
                                            (self.MAX_COMPUTE_SIZE,),
                                            name="z_tmp_ub",
                                            scope=tik.scope_ubuf)
            with self.tik_inst.for_range(0, repeat_time) as rep:
                self.tik_inst.data_move(
                    z_tmp_ub, self.loc_data_in[(box * 7 + 6) * 
                        self.loc_shape[2] * self.loc_shape[3] + 
                        rep * self.MAX_COMPUTE_SIZE],
                    0, 1, self.MAX_COMPUTE_SIZE // 16, 1, 1)
                vec_adds(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         angle_scalar)
                self.tik_inst.data_move(
                    self.loc_data_out[box, 6, rep * self.MAX_COMPUTE_SIZE],
                    z_tmp_ub, 0, 1, self.MAX_COMPUTE_SIZE // 16, 1, 1)
            if tail > 0:
                self.tik_inst.data_move(
                    z_tmp_ub, self.loc_data_in[(box * 7 + 6) * 
                        self.loc_shape[2] * self.loc_shape[3] + 
                        repeat_time * self.MAX_COMPUTE_SIZE],
                    0, 1, tail // 16, 1, 1)
                vec_adds(self.tik_inst, self.FLOAT16, z_tmp_ub, z_tmp_ub,
                         angle_scalar)
                self.tik_inst.data_move(self.loc_data_out[
                    box, 6, repeat_time * self.MAX_COMPUTE_SIZE],
                    z_tmp_ub, 0, 1, tail // 16, 1, 1)

    def compute_lwh_and_angle(self, box, anchor_d_ub):
        with self.tik_inst.for_range(0, 3) as size:
            with self.tik_inst.new_stmt_scope():
                lwh_scalar = self.tik_inst.Scalar(self.FLOAT16, 'lwh_scalar')
                lwh_scalar.set_as(anchor_d_ub[box, 2 + size])
                self.compute_lwh(box, size, lwh_scalar)
        with self.tik_inst.new_stmt_scope():
            angle_scalar = self.tik_inst.Scalar(self.FLOAT16, 'angle_scalar')
            angle_scalar.set_as(anchor_d_ub[box, 6])
            self.compute_angle(box, angle_scalar)

    def single_box_compute(self, anchor_d_ub, box, data_idx_w_ub,
                           data_idx_h_ub):
        w = self.loc_shape[3]
        w_align16 = (w + 15) // 16
        h = self.loc_shape[2]
        x_scalar = self.tik_inst.Scalar(self.FLOAT16, 'x_scalar')
        x_scalar.set_as(anchor_d_ub[box, 0])
        y_scalar = self.tik_inst.Scalar(self.FLOAT16, 'y_scalar')
        y_scalar.set_as(anchor_d_ub[box, 1])
        z_scalar = self.tik_inst.Scalar(self.FLOAT16, 'z_scalar')
        z_scalar.set_as(anchor_d_ub[box, 4])
        addz_scalar = self.tik_inst.Scalar(self.FLOAT16, 'addz_scalar')
        addz_scalar.set_as(anchor_d_ub[box, 5])
        tmp_scalar = self.tik_inst.Scalar(self.INT32, 'tmp_scalar')
        row_ub_int32 = self.tik_inst.Tensor(self.INT32, shape=(16,),
            scope=tik.scope_ubuf, name='row_ub_int32')
        row_ub_float16 = self.tik_inst.Tensor(self.FLOAT16, shape=(16,),
            scope=tik.scope_ubuf, name='row_ub_float16')
        row_scalar = self.tik_inst.Scalar(self.FLOAT16, name='row_scalar')
        with self.tik_inst.new_stmt_scope():
            with self.tik_inst.for_range(0, h, thread_num=2) as row:
                x_data_ub = self.tik_inst.Tensor(self.FLOAT16,
                    (w_align16 * 16,), name="x_data_ub", scope=tik.scope_ubuf)
                self.compute_x_coor(x_data_ub, w_align16, row, box, x_scalar,
                                    data_idx_w_ub)
            with self.tik_inst.for_range(0, h, thread_num=2) as row:
                y_data_ub = self.tik_inst.Tensor(self.FLOAT16,
                    (w_align16 * 16,), name="y_data_ub", scope=tik.scope_ubuf)
                data_idx_h_ub_1 = self.tik_inst.Tensor(self.FLOAT16,
                    (w_align16 * 16,), name="data_idx_h_ub_1",
                    scope=tik.scope_ubuf)
                tmp_scalar.set_as(row)
                row_ub_int32.set_as(tmp_scalar)
                vconv_int32tofp16(self.tik_inst, self.INT32, row_ub_int32,
                                  row_ub_float16)
                row_scalar.set_as(row_ub_float16[0])
                vec_muls(self.tik_inst, self.FLOAT16, data_idx_h_ub_1,
                         data_idx_h_ub, row_scalar)
                vec_muls(self.tik_inst, self.FLOAT16, data_idx_h_ub_1,
                         data_idx_h_ub_1, self.ystep)
                vec_adds(self.tik_inst, self.FLOAT16, data_idx_h_ub_1,
                         data_idx_h_ub_1, self.y_start)
                self.compute_y_coor(y_data_ub, w_align16, row, box, y_scalar,
                                    data_idx_h_ub_1)
        with self.tik_inst.new_stmt_scope():
            self.compute_z_coor(box, z_scalar, addz_scalar)
        self.compute_lwh_and_angle(box, anchor_d_ub)
        
    def coordinate_xyz(self, kernel_name):
        anchor_d_ub = self.tik_inst.Tensor(self.FLOAT16, (self.box_num, 16),
                                           name="anchor_d_ub",
                                           scope=tik.scope_ubuf)
        w = self.loc_shape[3]
        w_align16 = (w + 15) // 16
        data_idx_w_ub = self.tik_inst.Tensor(self.FLOAT16, (w_align16 * 16,),
                                              name="data_idx_w_ub",
                                              scope=tik.scope_ubuf)
        data_idx_h_ub = self.tik_inst.Tensor(self.FLOAT16, (w_align16 * 16,),
                                              name="data_idx_h_ub",
                                              scope=tik.scope_ubuf)
        vec_dup(self.tik_inst, data_idx_h_ub.dtype, data_idx_h_ub, 1)
        with self.tik_inst.new_stmt_scope():
            data_idx_w_ub_1 = self.tik_inst.Tensor(self.INT32,
                                                    (w_align16 * 16,),
                                                    name="data_idx_w_ub_1",
                                                    scope=tik.scope_ubuf)
            vec_dup(self.tik_inst, data_idx_w_ub_1.dtype, data_idx_w_ub_1, 0)
            with self.tik_inst.for_range(0, w) as w_index:
                data_idx_w_ub_1[w_index].set_as(w_index)
            vconv_int32tofp16(self.tik_inst, self.INT32, data_idx_w_ub_1,
                              data_idx_w_ub)

        anchor_d_scalar = self.tik_inst.Scalar('float16',
                                               name='anchor_d_scalar')
        for aa in range(0, self.box_num):
            for i in range(0, 7):
                anchor_d_scalar.set_as(self.anchor_d[aa, i])
                anchor_d_ub[aa, i].set_as(anchor_d_scalar)
        vec_muls(self.tik_inst, self.FLOAT16, data_idx_w_ub, data_idx_w_ub,
                 self.xstep)
        vec_adds(self.tik_inst, self.FLOAT16, data_idx_w_ub, data_idx_w_ub,
                 self.x_start)
        with self.tik_inst.for_range(0, self.box_num,
                                     block_num=self.box_num) as box:
            self.single_box_compute(anchor_d_ub, box, data_idx_w_ub,
                                    data_idx_h_ub)
        self.tik_inst.BuildCCE(kernel_name=kernel_name,
                               inputs=[self.loc_data_in],
                               outputs=[self.loc_data_out], enable_l2=True)
        return self.tik_inst


def check_params(loc_shape, input_dtype, box1_size, box2_size, box3_size, 
                 box4_size, z_strides, rotations, kernel_name):
    log.check_eq(len(loc_shape), 4,
                 "the input dims of coord should be equal 4")
    log.check_eq(loc_shape[0], 1, "only support single batch")
    log.check_eq(input_dtype, "float16", "only support float16")
    log.check_eq(len(box1_size), 3, "elem num in param box1_size must be 3")
    log.check_eq(len(box2_size), 3, "elem num in param box2_size must be 3")
    log.check_eq(len(box3_size), 3, "elem num in param box3_size must be 3")
    log.check_eq(len(box4_size), 3, "elem num in param box4_size must be 3")
    log.check_eq(len(z_strides), 4, "elem num in param z_strides must be 4")
    log.check_eq(len(rotations), 2, "elem num in param rotations must be 2")
    log.check_kernelname(kernel_name)


def start_cal_coordinate(coor, decode_coor,
                         x_start, x_strides,
                         y_start, y_strides,
                         box1_size, box2_size,
                         box3_size, box4_size,
                         z_strides, rotations,
                         kernel_name="pointpillar_decode_box"):
    loc_shape = coor.get("shape")
    input_dtype = coor.get("dtype")
    check_params(loc_shape, input_dtype, box1_size, box2_size, box3_size, 
                 box4_size, z_strides, rotations, kernel_name)
    anchor_dims = ((box1_size[0], box1_size[1], box1_size[2],
                    z_strides[0], rotations[0]),
                   (box1_size[0], box1_size[1], box1_size[2],
                    z_strides[0], rotations[1]),
                   (box2_size[0], box2_size[1], box2_size[2],
                    z_strides[1], rotations[0]),
                   (box2_size[0], box2_size[1], box2_size[2],
                    z_strides[1], rotations[1]),
                   (box3_size[0], box3_size[1], box3_size[2],
                    z_strides[2], rotations[0]),
                   (box3_size[0], box3_size[1], box3_size[2],
                    z_strides[2], rotations[1]),
                   (box4_size[0], box4_size[1], box4_size[2],
                    z_strides[3], rotations[0]),
                   (box4_size[0], box4_size[1], box4_size[2],
                    z_strides[3], rotations[1]))
    cal_coordinate = Coordinate(x_start, x_strides, y_start, y_strides,
                                loc_shape, anchor_dims)
    return cal_coordinate.coordinate_xyz(kernel_name)

