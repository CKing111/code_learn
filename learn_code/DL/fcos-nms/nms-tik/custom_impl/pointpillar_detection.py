import numpy as np
import nms
from te import tik
from common import cos
from common import sin
from common import vec_mul
from common import vec_add
from common import vec_sub
from common import vec_muls
from common import vec_dup
from util import OpLog as log


class Pointpillar:
    def __init__(self, loc_shape, cls_shape, head_shape,
                 thresh, per_class_num):
        self.INT16 = "int16"
        self.FLOAT16 = "float16"
        self.tik_inst = tik.Tik(tik.Dprofile())
        self.loc_shape = loc_shape
        self.cls_shape = cls_shape
        self.head_shape = head_shape
        self.box_num = self.loc_shape[1]
        self.single_cycle_num = 1024
        self.thresh = thresh
        self.per_class_num = per_class_num
        loc_num = self.loc_shape[0] * self.loc_shape[1] * self.loc_shape[2] * \
                  self.loc_shape[3]
        class_num = self.cls_shape[0] * self.cls_shape[1] * self.cls_shape[
            2] * self.cls_shape[3]
        head_num = self.head_shape[0] * self.head_shape[1] * self.head_shape[
            2] * self.head_shape[3]
        self.loc_data_in = self.tik_inst.Tensor(dtype=self.FLOAT16,
                                                shape=(loc_num + 16,),
                                                scope=tik.scope_gm,
                                                name="loc_data_in")
        self.cls_data_in = self.tik_inst.Tensor(dtype=self.FLOAT16,
                                                shape=(class_num + 16,),
                                                scope=tik.scope_gm,
                                                name="cls_data_in")
        self.head_data_in = self.tik_inst.Tensor(dtype=self.FLOAT16,
                                                 shape=(head_num + 16,),
                                                 scope=tik.scope_gm,
                                                 name="head_data_in")
        self.data_out = self.tik_inst.Tensor(self.FLOAT16,
            (self.cls_shape[2] * self.per_class_num, 16), scope=tik.scope_gm,
            name="data_out")

    def sorted_score(self, loc_x_ub, loc_y_ub, size_w_ub, size_l_ub, cls_ub,
                     top_proposal_ub_xywl, tmp_top_proposal_ub, trans_dst):
        self.tik_inst.vconcat(trans_dst, loc_x_ub,
            self.single_cycle_num // 16, 0) 
        self.tik_inst.vconcat(trans_dst, loc_y_ub,
            self.single_cycle_num // 16, 1)  
        self.tik_inst.vconcat(trans_dst, size_w_ub,
            self.single_cycle_num // 16, 2)  
        self.tik_inst.vconcat(trans_dst, size_l_ub,
            self.single_cycle_num // 16, 3)  
        self.tik_inst.vconcat(
            trans_dst, cls_ub, self.single_cycle_num // 16, 4)
        sort4_16_proposal_ub = self.tik_inst.Tensor(self.FLOAT16, 
            (self.single_cycle_num, 8), name="sort4_16_proposal_ub",
            scope=tik.scope_ubuf)
        self.tik_inst.vrpsort16(sort4_16_proposal_ub, trans_dst,
                                self.single_cycle_num // 16)
        sort4_64_proposal_ub = self.tik_inst.Tensor(self.FLOAT16,
            (self.single_cycle_num, 8), name="sort4_64_proposal_ub",
            scope=tik.scope_ubuf)
        rep_1 = self.single_cycle_num // 64
        with self.tik_inst.for_range(0, rep_1) as idx:
            offset = idx * 64
            self.tik_inst.vmrgsort4(sort4_64_proposal_ub[idx * 64, 0],
                                    [sort4_16_proposal_ub[offset, 0],
                                     sort4_16_proposal_ub[offset + 16, 0],
                                     sort4_16_proposal_ub[offset + 32, 0],
                                     sort4_16_proposal_ub[offset + 48, 0]],
                                    [16, 16, 16, 16], False, 15, 1, None)
        sort4_256_proposal_ub = self.tik_inst.Tensor(self.FLOAT16,
            (self.single_cycle_num, 8), name="sort4_256_proposal_ub",
            scope=tik.scope_ubuf)
        rep_2 = self.single_cycle_num // 256
        with self.tik_inst.for_range(0, rep_2) as idx:
            offset = idx * 256
            self.tik_inst.vmrgsort4(sort4_256_proposal_ub[idx * 256, 0],
                [sort4_64_proposal_ub[offset, 0], 
                sort4_64_proposal_ub[offset + 64, 0],
                sort4_64_proposal_ub[offset + 64 * 2, 0],
                sort4_64_proposal_ub[offset + 64 * 3, 0]],
                [64, 64, 64, 64], False, 15, 1, None)
        self.tik_inst.vmrgsort4(sort4_64_proposal_ub,
            [sort4_256_proposal_ub[0], sort4_256_proposal_ub[256, 0],
            sort4_256_proposal_ub[256 * 2, 0], sort4_256_proposal_ub[256 * 3,
                0]], [256, 256, 256, 256], False, 15, 1, None)
        self.tik_inst.vmrgsort4(tmp_top_proposal_ub, [sort4_64_proposal_ub,
            top_proposal_ub_xywl, top_proposal_ub_xywl, top_proposal_ub_xywl],
            [self.single_cycle_num, 256, 0, 0], False, 3, 1, None)
        self.tik_inst.data_move(top_proposal_ub_xywl, tmp_top_proposal_ub, 0,
                                1, 256 * 8 // 16, 0, 0)

    def cal_new_coor_process_one(self, tmp_angle_ub_cos, tmp_angle_ub_sin,
            tmp_x_ub, tmp_y_ub, tmp_w_ub, tmp_l_ub, box_nms):
        tmp_new_w = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="tmp_new_w", scope=tik.scope_ubuf)
        tmp_new_h = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="tmp_new_h", scope=tik.scope_ubuf)
        new_w = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="new_w", scope=tik.scope_ubuf)
        new_h = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="new_h", scope=tik.scope_ubuf)
        vec_mul(self.tik_inst, self.FLOAT16, tmp_new_w, tmp_w_ub,
                tmp_angle_ub_cos)
        vec_mul(self.tik_inst, self.FLOAT16, tmp_new_h, tmp_l_ub,
                tmp_angle_ub_sin)
        vec_add(self.tik_inst, self.FLOAT16, new_w, tmp_new_w, tmp_new_h)
        vec_mul(self.tik_inst, self.FLOAT16, tmp_new_w, tmp_w_ub,
                tmp_angle_ub_sin)
        vec_mul(self.tik_inst, self.FLOAT16, tmp_new_h, tmp_l_ub,
                tmp_angle_ub_cos)
        vec_add(self.tik_inst, self.FLOAT16, new_h, tmp_new_w, tmp_new_h)
        trans_x1_ub = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="trans_x1_ub", scope=tik.scope_ubuf)
        trans_y1_ub = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="trans_y1_ub", scope=tik.scope_ubuf)
        trans_x2_ub = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="trans_x2_ub", scope=tik.scope_ubuf)
        trans_y2_ub = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="trans_y2_ub", scope=tik.scope_ubuf)
        half_new_w = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="half_new_w", scope=tik.scope_ubuf)
        half_new_h = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="half_new_h", scope=tik.scope_ubuf)
        vec_muls(self.tik_inst, self.FLOAT16, half_new_w, new_w, 0.5)
        vec_muls(self.tik_inst, self.FLOAT16, half_new_h, new_h, 0.5)
        vec_sub(self.tik_inst, self.FLOAT16, trans_x1_ub, tmp_x_ub,
                half_new_w)
        vec_add(self.tik_inst, self.FLOAT16, trans_x2_ub, tmp_x_ub,
                half_new_w)
        vec_sub(self.tik_inst, self.FLOAT16, trans_y1_ub, tmp_y_ub,
                half_new_h)
        vec_add(self.tik_inst, self.FLOAT16, trans_y2_ub, tmp_y_ub,
                half_new_h)
        self.tik_inst.vconcat(box_nms[0], trans_x1_ub[0], 16, 0)
        self.tik_inst.vconcat(box_nms[0], trans_y1_ub[0], 16, 1)
        self.tik_inst.vconcat(box_nms[0], trans_x2_ub[0], 16, 2)
        self.tik_inst.vconcat(box_nms[0], trans_y2_ub[0], 16, 3)

    def cal_new_coor(self, box_nms, top_proposal_ub_xywl,
                     top_proposal_ub_zah):
        self.tik_inst.data_move(box_nms, top_proposal_ub_xywl,
                                0, 1, 128, 0, 0)
        tmp_x_ub = self.tik_inst.Tensor(self.FLOAT16, shape=(256,),
                                        name="tmp_x_ub", scope=tik.scope_ubuf)
        tmp_y_ub = self.tik_inst.Tensor(self.FLOAT16, shape=(256,),
                                        name="tmp_y_ub", scope=tik.scope_ubuf)
        tmp_w_ub = self.tik_inst.Tensor(self.FLOAT16, shape=(256,),
                                        name="tmp_w_ub", scope=tik.scope_ubuf)
        tmp_l_ub = self.tik_inst.Tensor(self.FLOAT16, shape=(256,),
                                        name="tmp_l_ub", scope=tik.scope_ubuf)
        tmp_angle_ub = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="tmp_angle_ub", scope=tik.scope_ubuf)
        self.tik_inst.vextract(tmp_x_ub, top_proposal_ub_xywl[0], 16, 0)
        self.tik_inst.vextract(tmp_y_ub, top_proposal_ub_xywl[0], 16, 1)
        self.tik_inst.vextract(tmp_l_ub, top_proposal_ub_zah[0], 16, 0)
        self.tik_inst.vextract(tmp_w_ub, top_proposal_ub_xywl[0], 16, 3)
        self.tik_inst.vextract(tmp_angle_ub, top_proposal_ub_zah[0], 16, 2)
        self.tik_inst.vabs(128, tmp_angle_ub, tmp_angle_ub, 2, 1, 1, 8, 8)
        tmp_angle_ub_cos = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="tmp_angle_ub_cos", scope=tik.scope_ubuf)
        tmp_angle_ub_cos = cos(self.tik_inst, tmp_angle_ub_cos, tmp_angle_ub)
        self.tik_inst.vabs(128, tmp_angle_ub_cos, tmp_angle_ub_cos, 2, 1, 1,
                           8, 8)
        tmp_angle_ub_sin = self.tik_inst.Tensor(self.FLOAT16,
            shape=(256,), name="tmp_angle_ub_sin", scope=tik.scope_ubuf)
        tmp_angle_ub_sin = sin(self.tik_inst, tmp_angle_ub_sin, tmp_angle_ub)
        self.tik_inst.vabs(128, tmp_angle_ub_sin, tmp_angle_ub_sin, 2, 1, 1,
                           8, 8)
        self.cal_new_coor_process_one(tmp_angle_ub_cos, tmp_angle_ub_sin,
            tmp_x_ub, tmp_y_ub, tmp_w_ub, tmp_l_ub, box_nms)

    def one_class_all_box_sorted_repeat_num(self, box, repeat_num, loc_x_ub,
        loc_y_ub, loc_z_ub, size_w_ub, size_h_ub, size_l_ub, cls_ub,
        aicore_use, core_num, angle_ub, head_ub, trans_dst,
        top_proposal_ub_xywl, tmp_top_proposal_ub_xywl, 
        top_proposal_ub_zah, tmp_top_proposal_ub_zah):
        with self.tik_inst.for_range(0, repeat_num) as rep:
            self.tik_inst.data_move(loc_x_ub, self.loc_data_in[
                box * 7 * self.loc_shape[
                    3] + rep * self.single_cycle_num], 0, 1,
                                    self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(loc_y_ub, self.loc_data_in[
                (box * 7 + 1) * self.loc_shape[
                    3] + rep * self.single_cycle_num], 0, 1,
                                    self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(loc_z_ub, self.loc_data_in[
                (box * 7 + 2) * self.loc_shape[
                    3] + rep * self.single_cycle_num], 0, 1,
                                    self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(size_w_ub, self.loc_data_in[
                (box * 7 + 3) * self.loc_shape[
                    3] + rep * self.single_cycle_num], 0, 1,
                                    self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(size_h_ub, self.loc_data_in[
                (box * 7 + 5) * self.loc_shape[
                    3] + rep * self.single_cycle_num], 0, 1,
                                    self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(size_l_ub, self.loc_data_in[
                (box * 7 + 4) * self.loc_shape[
                    3] + rep * self.single_cycle_num], 0, 1,
                                    self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(cls_ub, self.cls_data_in[
                (box * aicore_use + core_num) * self.cls_shape[
                    3] + rep * self.single_cycle_num], 0, 1,
                                    self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(angle_ub, self.loc_data_in[
                (box * 7 + 6) * self.loc_shape[3] + rep * 
                self.single_cycle_num], 0, 1,
                self.single_cycle_num // 16, 0, 0)
            self.tik_inst.data_move(head_ub, self.head_data_in[
                box * self.head_shape[3] + rep * self.single_cycle_num],
                0, 1, self.single_cycle_num // 16, 0, 0)
            with self.tik_inst.new_stmt_scope():
                self.sorted_score(loc_x_ub, loc_y_ub, loc_z_ub, size_w_ub,
                                  cls_ub, top_proposal_ub_xywl, 
                                  tmp_top_proposal_ub_xywl, trans_dst)
                self.sorted_score(size_l_ub, size_h_ub, angle_ub, head_ub,
                                  cls_ub, top_proposal_ub_zah,
                                  tmp_top_proposal_ub_zah, trans_dst)

    def one_class_all_box_sorted_res_num(self, box, repeat_num, res_num,
        cls_ub, loc_x_ub, loc_y_ub, loc_z_ub, size_w_ub, size_h_ub,
        size_l_ub, aicore_use, core_num, angle_ub, head_ub, trans_dst,
        top_proposal_ub_xywl, tmp_top_proposal_ub_xywl,
        top_proposal_ub_zah, tmp_top_proposal_ub_zah):
        vec_dup(self.tik_inst, self.FLOAT16, cls_ub, 0)
        self.tik_inst.data_move(loc_x_ub, self.loc_data_in[
            box * 7 * self.loc_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(loc_y_ub, self.loc_data_in[
            (box * 7 + 1) * self.loc_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(loc_z_ub, self.loc_data_in[
            (box * 7 + 2) * self.loc_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(size_w_ub, self.loc_data_in[
            (box * 7 + 3) * self.loc_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(size_h_ub, self.loc_data_in[
            (box * 7 + 5) * self.loc_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(size_l_ub, self.loc_data_in[
            (box * 7 + 4) * self.loc_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(cls_ub, self.cls_data_in[
            (box * aicore_use + core_num) * self.cls_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(angle_ub, self.loc_data_in[
            (box * 7 + 6) * self.loc_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        self.tik_inst.data_move(head_ub, self.head_data_in[
            box * self.head_shape[
                3] + repeat_num * self.single_cycle_num], 0, 1,
                                res_num // 16, 0, 0)
        with self.tik_inst.new_stmt_scope():
            self.sorted_score(loc_x_ub, loc_y_ub, loc_z_ub, size_w_ub,
                                cls_ub, top_proposal_ub_xywl,
                                tmp_top_proposal_ub_xywl, trans_dst)
            self.sorted_score(size_l_ub, size_h_ub, angle_ub, head_ub,
                                cls_ub, top_proposal_ub_zah,
                                tmp_top_proposal_ub_zah, trans_dst)

    def one_class_all_box_sorted(self, aicore_use, core_num,
                                top_proposal_ub_zah, top_proposal_ub_xywl):
        with self.tik_inst.for_range(0, self.box_num) as box:
            repeat_num = self.loc_shape[3] // self.single_cycle_num
            res_num = self.loc_shape[3] % self.single_cycle_num
            loc_x_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'loc_x_ub')
            loc_y_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'loc_y_ub')
            loc_z_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'loc_z_ub')  
            size_w_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'size_w_ub')
            size_h_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'size_h_ub')
            size_l_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'size_l_ub')
            cls_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'cls_ub')
            angle_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'angle_ub')
            head_ub = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num,), tik.scope_ubuf, 'head_ub')
            trans_dst = self.tik_inst.Tensor(self.FLOAT16,
                (self.single_cycle_num, 8), tik.scope_ubuf, 'trans_dst')
            tmp_top_proposal_ub_xywl = self.tik_inst.Tensor(self.FLOAT16, (
                self.single_cycle_num * 8 + 256 * 8,),
                name="tmp_top_proposal_ub_xywl", scope=tik.scope_ubuf)
            vec_dup(self.tik_inst, self.FLOAT16, tmp_top_proposal_ub_xywl, 0)
            tmp_top_proposal_ub_zah = self.tik_inst.Tensor(self.FLOAT16, (
                self.single_cycle_num * 8 + 256 * 8,),
                name="tmp_top_proposal_ub_zah", scope=tik.scope_ubuf)
            vec_dup(self.tik_inst, self.FLOAT16, tmp_top_proposal_ub_zah, 0)
            self.one_class_all_box_sorted_repeat_num(box, repeat_num, loc_x_ub,
                loc_y_ub, loc_z_ub, size_w_ub, size_h_ub, size_l_ub, cls_ub,
                aicore_use, core_num, angle_ub, head_ub, trans_dst,
                top_proposal_ub_xywl, tmp_top_proposal_ub_xywl, 
                top_proposal_ub_zah, tmp_top_proposal_ub_zah)

            if res_num > 0:
                self.one_class_all_box_sorted_res_num(box, repeat_num, res_num,
                    cls_ub, loc_x_ub, loc_y_ub, loc_z_ub, size_w_ub, size_h_ub,
                    size_l_ub, aicore_use, core_num, angle_ub, head_ub,
                    trans_dst, top_proposal_ub_xywl, tmp_top_proposal_ub_xywl,
                    top_proposal_ub_zah, tmp_top_proposal_ub_zah)

    def get_ret_ub(self, total_input_proposal_num, sup_vector,
                   selected_proposal, total_out_proposal,
                   ret_ub, top_proposal_ub_xywl, top_proposal_ub_zah):
        with self.tik_inst.for_range(0, total_input_proposal_num) as i:
            with self.tik_inst.if_scope(
                tik.all(sup_vector[i] == 0,
                selected_proposal < total_out_proposal)):
                with self.tik_inst.for_range(0, 4) as j:
                    ret_ub[selected_proposal, j].set_as(
                        top_proposal_ub_xywl[i * 8 + j])
                with self.tik_inst.for_range(0, 5) as t:
                    ret_ub[selected_proposal, t + 4].set_as(
                        top_proposal_ub_zah[i * 8 + t])
                selected_proposal.set_as(selected_proposal + 1)

    def pointpillar(self, kernel_name):
        aicore_use = self.cls_shape[2]
        total_input_proposal_num = 256
        total_out_proposal = self.per_class_num
        burst_input_proposal_num = 256
        down_factor = 1
        ret_ub = self.tik_inst.Tensor(self.FLOAT16,
            (total_out_proposal, 16), tik.scope_ubuf, name="ret_ub")
        self.tik_inst.vector_dup(16, ret_ub, 0, total_out_proposal, 1, 1)
        with self.tik_inst.for_range(0, aicore_use,
                                     block_num=aicore_use) as core_num:
            top_proposal_ub_xywl = self.tik_inst.Tensor(self.FLOAT16,
                (256 * 8,), name="top_proposal_ub_xywl", scope=tik.scope_ubuf)
            vec_dup(self.tik_inst, self.FLOAT16, top_proposal_ub_xywl, 0)
            top_proposal_ub_zah = self.tik_inst.Tensor(self.FLOAT16,
                (256 * 8,), name="top_proposal_ub_zah", scope=tik.scope_ubuf)
            vec_dup(self.tik_inst, self.FLOAT16, top_proposal_ub_zah, 0)
            self.one_class_all_box_sorted(aicore_use, core_num,
                top_proposal_ub_zah, top_proposal_ub_xywl)
            box_nms = self.tik_inst.Tensor(self.FLOAT16, (256, 8),
                name="box_nms", scope=tik.scope_ubuf)
            self.cal_new_coor(box_nms, top_proposal_ub_xywl,
                              top_proposal_ub_zah)
            one_core_nms = nms.OneCoreNMS(self.tik_inst, (
                total_input_proposal_num, total_out_proposal,
                burst_input_proposal_num, down_factor))
            sup_vector = one_core_nms.nms_single_core(box_nms, self.thresh)
            selected_proposal = self.tik_inst.Scalar("uint16")
            selected_proposal.set_as(0)
            self.tik_inst.vector_dup(16, ret_ub, 0, total_out_proposal, 1, 1)
            self.get_ret_ub(total_input_proposal_num, sup_vector,
                            selected_proposal, total_out_proposal,
                            ret_ub, top_proposal_ub_xywl, top_proposal_ub_zah)
            self.tik_inst.data_move(
                self.data_out[core_num * total_out_proposal, 0],
                ret_ub, 0, 1, total_out_proposal, 0, 0, 0)
            self.tik_inst.vector_dup(16, ret_ub, 0, total_out_proposal, 1, 1)
        self.tik_inst.BuildCCE(kernel_name=kernel_name,
                               inputs=[self.loc_data_in, self.cls_data_in,
                                       self.head_data_in],
                               outputs=[self.data_out], enable_l2=True)
        return self.tik_inst


def check_params(loc_shape, cls_shape, head_shape,
                 thresh, per_class_num, kernel_name):
    log.check_eq(len(loc_shape), 4,
                 "the input dims of coor should be equal 4")
    log.check_eq(len(cls_shape), 4,
                 "the input dims of class score should be equal 4")
    log.check_eq(cls_shape[2], 4,
                 "class num should be equal 4")
    log.check_eq(len(head_shape), 4,
                 "the input dims of head should be equal 4")
    log.check_lt(thresh, 1, "the values of  thresh shou1d be less than 1")
    log.check_gt(thresh, 0, "the values of  thresh shou1d be greater than 0")
    log.check_le(per_class_num, 60,
                 "the values of  per_class_num shou1d be less than 60")
    log.check_ge(per_class_num, 20,
                 "the values of  per_class_num shou1d be greater than 20")
    log.check_kernelname(kernel_name)


def pointpillar_detection(decode_coor, class_prob, head_direct, box,
                          thresh, per_class_num,
                          kernel_name="pointpillar_detection"):
    loc_shape = decode_coor.get("shape")
    cls_shape = class_prob.get("shape")
    head_shape = head_direct.get("shape")
    check_params(loc_shape, cls_shape, head_shape,
                 thresh, per_class_num, kernel_name)
    pointpillar_nms = Pointpillar(loc_shape, cls_shape, head_shape, 
                                  thresh, per_class_num)
    return pointpillar_nms.pointpillar(kernel_name)
