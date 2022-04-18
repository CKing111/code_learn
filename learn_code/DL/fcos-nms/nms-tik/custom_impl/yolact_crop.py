# -*- coding: <utf-8> -*-
from te import tik


class YolactCrop():
    def __init__(self, n, h, w, padding=1, kernel_name="YolactCrop"):
        self.n = n
        self.h = h
        self.w = w
        self.padding = padding
        self.masks_out_size = n*w*h+16
        self.max_ub_size = 256*16*29
        self.kernel_name = kernel_name
        self.tik_instance = tik.Tik(tik.Dprofile())

        self.x1_scalar = self.tik_instance.Scalar(dtype="uint32")
        self.x2_scalar = self.tik_instance.Scalar(dtype="uint32")
        self.y1_scalar = self.tik_instance.Scalar(dtype="uint32")
        self.y2_scalar = self.tik_instance.Scalar(dtype="uint32")
        self.n_scalar = self.tik_instance.Scalar(dtype="uint32")
        self.zero_scalar = self.tik_instance.Scalar(dtype="float16")
        self.zero_scalar.set_as(0)
        self.n_scalar.set_as(n)

        self.masks_gm = self.tik_instance.Tensor(
            "float16", (n*h*w+16,), name="masks_gm", scope=tik.scope_gm)
        self.boxes_gm = self.tik_instance.Tensor(
            "float16", (n*4+16,), name="boxes_gm", scope=tik.scope_gm)
        self.masks_out_gm = self.tik_instance.Tensor(
            "float16", (n*h*w+16,), name="masks_out_gm", scope=tik.scope_gm)
        self.apply_ub(n)

    def apply_ub(self, n):
        self.x1_ub = self.tik_instance.Tensor(
            "float16", (n+64,), name="x1_ub", scope=tik.scope_ubuf)
        self.x2_ub = self.tik_instance.Tensor(
            "float16", (n+64,), name="x2_ub", scope=tik.scope_ubuf)
        self.y1_ub = self.tik_instance.Tensor(
            "float16", (n+64,), name="y1_ub", scope=tik.scope_ubuf)
        self.y2_ub = self.tik_instance.Tensor(
            "float16", (n+64,), name="y2_ub", scope=tik.scope_ubuf)
        self.x1_ub_int = self.tik_instance.Tensor(
            "int32", (n+64,), name="x1_ub_int", scope=tik.scope_ubuf)
        self.x2_ub_int = self.tik_instance.Tensor(
            "int32", (n+64,), name="x2_ub_int", scope=tik.scope_ubuf)
        self.y1_ub_int = self.tik_instance.Tensor(
            "int32", (n+64,), name="y1_ub_int", scope=tik.scope_ubuf)
        self.y2_ub_int = self.tik_instance.Tensor(
            "int32", (n+64,), name="y2_ub_int", scope=tik.scope_ubuf)
        self.xmin_ub = self.tik_instance.Tensor(
            "int32", (n+64,), name="xmin_ub", scope=tik.scope_ubuf)
        self.xmax_ub = self.tik_instance.Tensor(
            "int32", (n+64,), name="xmax_ub", scope=tik.scope_ubuf)
        self.ymin_ub = self.tik_instance.Tensor(
            "int32", (n+64,), name="ymin_ub", scope=tik.scope_ubuf)
        self.ymax_ub = self.tik_instance.Tensor(
            "int32", (n+64,), name="ymax_ub", scope=tik.scope_ubuf)
        self.s_0 = self.tik_instance.Tensor(
            "int32", (n+64,), name="s_0", scope=tik.scope_ubuf)
        self.s_w = self.tik_instance.Tensor(
            "int32", (n+64,), name="s_w", scope=tik.scope_ubuf)
        self.s_h = self.tik_instance.Tensor(
            "int32", (n+64,), name="s_h", scope=tik.scope_ubuf)
        self.s_p = self.tik_instance.Tensor(
            "int32", (n+64,), name="s_p", scope=tik.scope_ubuf)

    def compute_param(self, dtype, size):
        if dtype == 16:
            mask = 128
        elif dtype == 32:
            mask = 64
        repeat = size // mask // 255
        tail = size % (mask * 255)
        tail_repeat = tail // mask
        tail_mask = tail % mask
        return mask, repeat, tail_repeat, tail_mask

    def vec_dup(self, dtype, dst, scalar, size):
        mask, repeat, tail_repeat, tail_mask = self.compute_param(dtype, size)
        if repeat > 0:
            for i in range(repeat):
                self.tik_instance.vector_dup(
                    mask, dst[255 * mask * i], scalar, 255, 1, 8)
        if tail_repeat > 0:
            self.tik_instance.vector_dup(
                mask, dst[255 * mask * repeat], scalar,
                tail_repeat, 1, 8)
        if tail_mask > 0:
            self.tik_instance.vector_dup(
                tail_mask, dst[(255 * repeat + tail_repeat) * mask], scalar,
                1, 1, 8)

    def sigmoid(self, data_ub, length):
        scalar_negative_1 = self.tik_instance.Scalar("float16")
        scalar_negative_1.set_as(-1.0)
        scalar_1 = self.tik_instance.Scalar("float16")
        scalar_1.set_as(1.0)
        repeat_num = (length) // 128
        self.tik_instance.vmuls(
            128, data_ub, data_ub,
            scalar_negative_1, repeat_num, 1, 1, 8, 8)
        self.tik_instance.vexp(
            128, data_ub, data_ub, repeat_num, 1, 1, 8, 8)
        self.tik_instance.vadds(
            128, data_ub, data_ub, scalar_1,
            repeat_num, 1, 1, 8, 8)
        self.tik_instance.vrec(
            128, data_ub, data_ub, repeat_num, 1, 1, 8, 8)

    def magnify_boxes(self):
        self.vec_dup(16, self.x1_ub, 0, self.n+64)
        self.vec_dup(16, self.x2_ub, 0, self.n+64)
        self.vec_dup(16, self.y1_ub, 0, self.n+64)
        self.vec_dup(16, self.y2_ub, 0, self.n+64)
        self.vec_dup(32, self.x1_ub_int, 0, self.n+64)
        self.vec_dup(32, self.x2_ub_int, 0, self.n+64)
        self.vec_dup(32, self.y1_ub_int, 0, self.n+64)
        self.vec_dup(32, self.y2_ub_int, 0, self.n+64)
        self.tik_instance.data_move(
            self.x1_ub, self.boxes_gm, 0, 1, (self.n+15)//16, 1, 0)
        self.tik_instance.data_move(
            self.x2_ub, self.boxes_gm[self.n*2], 0, 1, (self.n+15)//16, 1, 0)
        self.tik_instance.data_move(
            self.y1_ub, self.boxes_gm[self.n], 0, 1, (self.n+15)//16, 1, 0)
        self.tik_instance.data_move(
            self.y2_ub, self.boxes_gm[self.n*3], 0, 1, (self.n+15)//16, 1, 0)
        self.tik_instance.vmuls(
            self.n, self.x1_ub, self.x1_ub, self.w, 1, 1, 1, 8, 8)
        self.tik_instance.vmuls(
            self.n, self.x2_ub, self.x2_ub, self.w, 1, 1, 1, 8, 8)
        self.tik_instance.vmuls(
            self.n, self.y1_ub, self.y1_ub, self.h, 1, 1, 1, 8, 8)
        self.tik_instance.vmuls(
            self.n, self.y2_ub, self.y2_ub, self.h, 1, 1, 1, 8, 8)
        self.tik_instance.vconv(
            self.n_scalar, "ceil", self.x1_ub_int,
            self.x1_ub, (self.n+63)//64, 1, 1, 8, 4)
        self.tik_instance.vconv(
            self.n_scalar, "ceil", self.x2_ub_int,
            self.x2_ub, (self.n+63)//64, 1, 1, 8, 4)
        self.tik_instance.vconv(
            self.n_scalar, "ceil", self.y1_ub_int,
            self.y1_ub, (self.n+63)//64, 1, 1, 8, 4)
        self.tik_instance.vconv(
            self.n_scalar, "ceil", self.y2_ub_int,
            self.y2_ub, (self.n+63)//64, 1, 1, 8, 4)

    def cal_boxes(self):

        self.tik_instance.vmin(
            64, self.xmin_ub, self.x1_ub_int,
            self.x2_ub_int, (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(
            64, self.xmax_ub, self.x1_ub_int,
            self.x2_ub_int, (
                self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmin(
            64, self.ymin_ub, self.y1_ub_int,
            self.y2_ub_int, (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(
            64, self.ymax_ub, self.y1_ub_int,
            self.y2_ub_int, (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.vec_dup(32, self.s_0, 0, self.n)
        self.vec_dup(32, self.s_w, self.w, self.n)
        self.vec_dup(32, self.s_h, self.h, self.n)
        self.vec_dup(32, self.s_p, self.padding, self.n)
        self.tik_instance.vsub(
            64, self.xmin_ub, self.xmin_ub, self.s_p,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(
            64, self.xmax_ub, self.xmax_ub, self.s_p,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(
            64, self.ymin_ub, self.ymin_ub, self.s_p,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(
            64, self.ymax_ub, self.ymax_ub, self.s_p,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(
            64, self.x1_ub_int, self.xmin_ub, self.s_0,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmin(
            64, self.x2_ub_int, self.xmax_ub, self.s_w,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmax(
            64, self.y1_ub_int, self.ymin_ub, self.s_0,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmin(
            64, self.y2_ub_int, self.ymax_ub, self.s_h,
            (self.n+63)//64, 1, 1, 1, 8, 8, 8)

    def init_transport(self):
        if self.masks_out_size <= self.max_ub_size:
            masks_out_ub = self.tik_instance.Tensor(
                "float16", (self.masks_out_size,), name="masks_out_ub",
                scope=tik.scope_ubuf)
            self.vec_dup(16, masks_out_ub, 0, self.masks_out_size)
            self.tik_instance.data_move(
                self.masks_out_gm, masks_out_ub,
                0, 1, self.masks_out_size//16, 1, 0)
        else:
            num = self.masks_out_size//self.max_ub_size
            masks_out_ub = self.tik_instance.Tensor(
                "float16", (self.max_ub_size,), name="masks_out_ub",
                scope=tik.scope_ubuf)
            self.vec_dup(16, masks_out_ub, 0, self.max_ub_size)
            for i in range(num):
                self.tik_instance.data_move(
                    self. masks_out_gm[i*self.max_ub_size], masks_out_ub, 0, 1,
                    (self.max_ub_size+15)//16, 1, 0)
            burst = (self.masks_out_size-self.max_ub_size*num)//16
            self.tik_instance.data_move(
                self.masks_out_gm[num*self.max_ub_size], masks_out_ub, 0, 1,
                burst, 1, 0)

    def frame_transport(self):
        temp_ub = self.tik_instance.Tensor(
            "float16", (((self.w+127)//128*128),),
            name="temp_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.n) as i:
            self.x1_scalar.set_as(self.x1_ub_int[i])
            self.x2_scalar.set_as(self.x2_ub_int[i])
            self.y1_scalar.set_as(self.y1_ub_int[i])
            self.y2_scalar.set_as(self.y2_ub_int[i])
            with self.tik_instance.if_scope(
                    self.x2_scalar > self.x1_scalar+self.padding
                    and self.y2_scalar > self.y1_scalar+self.padding):
                with self.tik_instance.for_range(
                        self.y1_scalar, self.y2_scalar) as y2:
                    self.vec_dup(16, temp_ub, 0, (self.w+127)//128*128)
                    offset = i*self.w * self.h + y2*self.w+self.x1_scalar
                    burst = (self.x2_scalar-self.x1_scalar+15)//16
                    self.tik_instance.data_move(
                        temp_ub, self.masks_gm[offset],
                        0, 1, burst, 1, 0)
                    self.sigmoid(temp_ub, (self.w+127)//128*128)
                    begint = self.x2_scalar-self.x1_scalar
                    endt = self.x2_scalar-self.x1_scalar+1+16 - \
                        (self.x2_scalar+1-self.x1_scalar) % 16
                    with self.tik_instance.for_range(begint, endt) as k:
                        temp_ub[k].set_as(self.zero_scalar)
                    offset = i*self.w*self.h + y2 * self.w+self.x1_scalar
                    burst = (self.x2_scalar-1-self.x1_scalar+15)//16
                    self.tik_instance.data_move(
                        self.masks_out_gm[offset],
                        temp_ub, 0, 1, burst, 1, 0)

    def crop(self):
        self.magnify_boxes()
        self.cal_boxes()
        self.init_transport()
        self.frame_transport()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[
            self.masks_gm, self.boxes_gm], outputs=[self.masks_out_gm],
            enable_l2=True)
        return self.tik_instance


def yolact_crop(masks, boxes, output, h, w, padding, kernel_name="YolactCrop"):
    masks_shape = masks.get("shape")
    n = masks_shape[0]
    obj = YolactCrop(n, h, w, padding, kernel_name)
    return obj.crop()
