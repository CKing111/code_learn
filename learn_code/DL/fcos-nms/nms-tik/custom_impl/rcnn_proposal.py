# -*- coding: utf-8 -*-
from te import tik
import nms
from util import OpLog as log
from util import vec_dup
from util import vec_exp
from util import vec_maxs
from util import vec_mins
from util import vec_adds
from util import vec_sub
from util import vec_add
from util import vec_mul
from util import vec_muls
from util import vec_div
from util import vec_cmp


class PreProcess:
    def __init__(self, tik_instance, attributes, input_data):
        self.num_rois = attributes[0]
        self.shape = attributes[1]  # (height, width)
        self.allow_border = attributes[2]
        self.allow_border_ratio = attributes[3]
        self.min_size = attributes[4]  # (height, width)
        self.min_size_mode = attributes[5]
        self.threshold_background = 1 - attributes[6]
        self.bsz01 = attributes[7]  # bbox尺寸增加值
        self.do_bbox_norm = attributes[8]
        self.bbox_mean = attributes[9]
        self.bbox_std = attributes[10]
        self.refine_out_of_map_bbox = attributes[11]
        self.regress_agnostic = attributes[12]
        self.num_class = attributes[13]
        self.threshold_class = attributes[14]
        # blobs
        self.rois_ub = input_data[0]
        self.score_ub = input_data[1]
        self.bbox_pred_ub = input_data[2]
        # compute repeat and tail
        self.length = nms.ceil_div(self.num_rois, 16) * 16
        self.repeat = self.length // 128
        self.tail = self.length % 128

        self.tik_instance = tik_instance

    def mini_size_filter(self, conf_data):
        threshold_min_size_w = self.tik_instance.Tensor("float16", (128,),
            name="threshold_min_size_w", scope=tik.scope_ubuf)
        threshold_min_size_h = self.tik_instance.Tensor("float16", (128,),
            name="threshold_min_size_h", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(128, threshold_min_size_w,
                                     self.min_size[1] - self.bsz01, 1, 1, 8)
        self.tik_instance.vector_dup(128, threshold_min_size_h,
                                     self.min_size[0] - self.bsz01, 1, 1, 8)

        size_w = self.tik_instance.Tensor("float16", (1, self.length),
                                          name="size_w", scope=tik.scope_ubuf)
        size_h = self.tik_instance.Tensor("float16", (1, self.length),
                                          name="size_h", scope=tik.scope_ubuf)

        vec_sub(self.tik_instance, "float16", size_w, conf_data, conf_data, 0,
                2, 0)
        vec_sub(self.tik_instance, "float16", size_h, conf_data, conf_data, 0,
                3, 1)
        self.mini_size_filter_compute(threshold_min_size_w, 
                                    threshold_min_size_h,
                                    size_w, size_h, conf_data)

    def mini_size_filter_compute(self, threshold_min_size_w,
                                threshold_min_size_h,
                                size_w, size_h, conf_data):    
        if self.min_size_mode:
            vec_cmp(self.tik_instance, "LT", size_w, conf_data,
                    self.min_size[1] - self.bsz01, 0, 4)
            vec_cmp(self.tik_instance, "LT", size_h, conf_data,
                    self.min_size[0] - self.bsz01, 0, 4)
        else:
            conf_tmp = self.tik_instance.Tensor("float16", (self.length,),
                                                name="conf_tmp",
                                                scope=tik.scope_ubuf)
            self.tik_instance.tensor_mov(conf_tmp, conf_data[4, 0], '', 1,
                                         self.length // 16, 0, 0)
            vec_dup(self.tik_instance, "float16", conf_data, 0, 4)

            with self.tik_instance.for_range(0, self.repeat) as i:
                cmpmask = self.tik_instance.vcmp_ge(128, size_w[i * 128],
                                                    threshold_min_size_w, 1,
                                                    1)
                self.tik_instance.vsel(128, 0, conf_data[4, i * 128], cmpmask,
                                       conf_tmp[i * 128],
                                       conf_data[4, i * 128],
                                       1, 1, 1, 1, 8, 8, 8)
                cmpmask = self.tik_instance.vcmp_ge(128, size_h[i * 128],
                                                    threshold_min_size_h, 1,
                                                    1)
                self.tik_instance.vsel(128, 0, conf_data[4, i * 128], cmpmask,
                                       conf_tmp[i * 128],
                                       conf_data[4, i * 128],
                                       1, 1, 1, 1, 8, 8, 8)
            cmpmask = self.tik_instance.vcmp_ge(self.tail,
                                                size_w[self.repeat * 128],
                                                threshold_min_size_w, 1, 1)
            self.tik_instance.vsel(self.tail, 0,
                                   conf_data[4, self.repeat * 128], cmpmask,
                                   conf_tmp[self.repeat * 128],
                                   conf_data[4, self.repeat * 128], 1, 1, 1,
                                   1, 8, 8, 8)
            cmpmask = self.tik_instance.vcmp_ge(self.tail,
                                                size_h[self.repeat * 128],
                                                threshold_min_size_h, 1, 1)
            self.tik_instance.vsel(self.tail, 0,
                                   conf_data[4, self.repeat * 128], cmpmask,
                                   conf_tmp[self.repeat * 128],
                                   conf_data[4, self.repeat * 128], 1, 1, 1,
                                   1, 8, 8, 8)

    def select_coordinate(self, class_max, conf_data):
        # compute coordinate offset according to class
        coord_occup = 1
        cdst = self.tik_instance.Tensor("float16", (1, self.length),
                                        name="cdst", scope=tik.scope_ubuf)
        if self.regress_agnostic:
            vec_dup(self.tik_instance, "float16", cdst, coord_occup, 0)
        else:
            vec_adds(self.tik_instance, "float16", cdst, class_max,
                     coord_occup, 0, 0)

        for i in range(4):
            vec_dup(self.tik_instance, "float16", conf_data, 0, i)
        # select coordinates according to class with max confidence
        loop_coord = 1 if self.regress_agnostic else self.num_class
        col_idx_tmp = self.tik_instance.Tensor("int32", (128,),
                                               name="col_idx_tmp",
                                               scope=tik.scope_ubuf)
        col_idx = self.tik_instance.Tensor("float16", (128,), name="col_idx",
                                           scope=tik.scope_ubuf)
        with self.tik_instance.for_range(1, loop_coord + 1) as c_idx:
            self.tik_instance.vector_dup(64, col_idx_tmp, c_idx, 2, 1, 8)
            self.tik_instance.vconv(64, "", col_idx, col_idx_tmp, 2, 1, 1, 4,
                                    8, 1.0)
            with self.tik_instance.for_range(0, self.repeat) as i:
                cmpmask = self.tik_instance.vcmp_eq(128, cdst[i * 128],
                                                    col_idx, 1, 1)
                with self.tik_instance.for_range(0, 4) as n_idx:
                    self.tik_instance.vsel(128, 0, conf_data[n_idx, i * 128],
                                           cmpmask,
                                           self.bbox_pred_ub[
                                               c_idx * 4 + n_idx, i * 128],
                                           conf_data[n_idx, i * 128], 1, 1, 
                                           1, 1,
                                           8, 8, 8)
            # tail process
            cmpmask = self.tik_instance.vcmp_eq(self.tail,
                                                cdst[self.repeat * 128],
                                                col_idx, 1, 1)
            with self.tik_instance.for_range(0, 4) as n_idx:
                self.tik_instance.vsel(self.tail, 0,
                                       conf_data[n_idx, self.repeat * 128],
                                       cmpmask,
                                       self.bbox_pred_ub[
                                           c_idx * 4 + n_idx, 
                                           self.repeat * 128],
                                       conf_data[n_idx, self.repeat * 128],
                                       1, 1, 1, 1, 8, 8, 8)

        # do bbox normalize
        if self.do_bbox_norm:
            for n_idx in range(0, 4):
                vec_muls(self.tik_instance, "float16", conf_data, conf_data,
                         self.bbox_std[n_idx], n_idx, n_idx)
                vec_adds(self.tik_instance, "float16", conf_data, conf_data,
                         self.bbox_mean[n_idx], n_idx, n_idx)

    def adjust_bbox_coor(self, class_max, conf_data):
        rois_w = self.tik_instance.Tensor("float16", (1, self.length),
                                          name="rois_w", scope=tik.scope_ubuf)
        rois_h = self.tik_instance.Tensor("float16", (1, self.length),
                                          name="rois_h", scope=tik.scope_ubuf)
        rois_ctr_x = self.tik_instance.Tensor("float16", (1, self.length),
                                              name="rois_ctr_x",
                                              scope=tik.scope_ubuf)
        rois_ctr_y = self.tik_instance.Tensor("float16", (1, self.length),
                                              name="rois_ctr_y",
                                              scope=tik.scope_ubuf)

        vec_sub(self.tik_instance, "float16", rois_w, self.rois_ub,
                self.rois_ub, 0, 2, 0)
        vec_muls(self.tik_instance, "float16", rois_ctr_x, rois_w, 0.5, 0, 0)
        vec_adds(self.tik_instance, "float16", rois_w, rois_w, self.bsz01, 0,
                 0)
        vec_add(self.tik_instance, "float16", rois_ctr_x, rois_ctr_x,
                self.rois_ub, 0, 0, 0)

        vec_sub(self.tik_instance, "float16", rois_h, self.rois_ub,
                self.rois_ub, 0, 3, 1)
        vec_muls(self.tik_instance, "float16", rois_ctr_y, rois_h, 0.5, 0, 0)
        vec_adds(self.tik_instance, "float16", rois_h, rois_h, self.bsz01, 0,
                 0)
        vec_add(self.tik_instance, "float16", rois_ctr_y, rois_ctr_y,
                self.rois_ub, 0, 0, 1)

        # select coordinates according to class with max confidence,
        # and normalize coordinates
        self.select_coordinate(class_max, conf_data)
        self.compute_tw_th_ctx_cty_ltx_lty_rbx_rby(rois_w, rois_h,
            rois_ctr_x, rois_ctr_y, conf_data)

    def compute_tw_th_ctx_cty_ltx_lty_rbx_rby(self, rois_w, rois_h,
            rois_ctr_x, rois_ctr_y, conf_data):
        # compute tw, th, ctx, cty, ltx, lty, rbx, rby
        # tw
        vec_exp(self.tik_instance, "float16", conf_data, conf_data, 2, 2)
        vec_mul(self.tik_instance, "float16", conf_data, conf_data, rois_w, 2,
                2, 0)
        vec_adds(self.tik_instance, "float16", conf_data, conf_data,
                -self.bsz01, 2, 2)
        vec_muls(self.tik_instance, "float16", conf_data, conf_data, 0.5, 2,
                2)
        
        # th
        vec_exp(self.tik_instance, "float16", conf_data, conf_data, 3, 3)
        vec_mul(self.tik_instance, "float16", conf_data, conf_data, rois_h, 3,
                3, 0)
        vec_adds(self.tik_instance, "float16", conf_data, conf_data,
                 -self.bsz01, 3, 3)
        vec_muls(self.tik_instance, "float16", conf_data, conf_data, 0.5, 3, 3)

        vec_mul(self.tik_instance, "float16", conf_data, conf_data, rois_w, 0,
                0, 0)  # ctx
        vec_add(self.tik_instance, "float16", rois_ctr_x, rois_ctr_x,
                conf_data, 0, 0, 0)
        vec_mul(self.tik_instance, "float16", conf_data, conf_data, rois_h, 1,
                1, 0)  # cty
        vec_add(self.tik_instance, "float16", rois_ctr_y, rois_ctr_y,
                conf_data, 0, 0, 1)

        vec_sub(self.tik_instance, "float16", conf_data, rois_ctr_x,
                conf_data, 0, 0, 2)  # ltx
        vec_sub(self.tik_instance, "float16", conf_data, rois_ctr_y,
                conf_data, 1, 0, 3)  # lty
        vec_add(self.tik_instance, "float16", conf_data, rois_ctr_x,
                conf_data, 2, 0, 2)  # rbx
        vec_add(self.tik_instance, "float16", conf_data, rois_ctr_y,
                conf_data, 3, 0, 3)  # rby

        if self.refine_out_of_map_bbox:
            vec_maxs(self.tik_instance, "float16", conf_data, conf_data, 0, 0,
                     0)  # ltx
            vec_mins(self.tik_instance, "float16", conf_data, conf_data,
                     self.shape[1], 0, 0)

            vec_maxs(self.tik_instance, "float16", conf_data, conf_data, 0, 1,
                     1)  # lty
            vec_mins(self.tik_instance, "float16", conf_data, conf_data,
                     self.shape[0], 1, 1)

            vec_maxs(self.tik_instance, "float16", conf_data, conf_data, 0, 2,
                     2)  # rbx
            vec_mins(self.tik_instance, "float16", conf_data, conf_data,
                     self.shape[1], 2, 2)

            vec_maxs(self.tik_instance, "float16", conf_data, conf_data, 0, 3,
                     3)  # rby
            vec_mins(self.tik_instance, "float16", conf_data, conf_data,
                     self.shape[0], 3, 3)

    def filter_according_border(self, conf_data):
        if self.allow_border >= 0.0:
            shape = self.tik_instance.Tensor("float16", (2,), name="shape",
                                             scope=tik.scope_ubuf)
            shape[0].set_as(self.shape[0])
            shape[1].set_as(self.shape[1])
            self.tik_instance.vadds(2, shape, shape, self.allow_border, 1, 1,
                                    1, 8, 8)
            height_add_border = self.tik_instance.Scalar("float16")
            height_add_border.set_as(shape[0])
            width_add_border = self.tik_instance.Scalar("float16")
            width_add_border.set_as(shape[1])
            vec_cmp(self.tik_instance, "LT", self.rois_ub, conf_data,
                    -self.allow_border, 0, 4)
            vec_cmp(self.tik_instance, "LT", self.rois_ub, conf_data,
                    -self.allow_border, 1, 4)
            vec_cmp(self.tik_instance, "GT", self.rois_ub, conf_data,
                    width_add_border, 2, 4)
            vec_cmp(self.tik_instance, "GT", self.rois_ub, conf_data,
                    height_add_border, 3, 4)

        self.filter_according_border_process(conf_data)

    def filter_according_border_process(self, conf_data):
        if self.allow_border_ratio >= 0.0:
            coord = self.tik_instance.Tensor("float16",
                (4, nms.ceil_div(self.num_rois, 16) * 16),
                name="coord", scope=tik.scope_ubuf)
            vec_maxs(self.tik_instance, "float16", coord, self.rois_ub, 0, 0,
                     0)
            vec_maxs(self.tik_instance, "float16", coord, self.rois_ub, 0, 1,
                     1)
            vec_mins(self.tik_instance, "float16", coord, self.rois_ub,
                     self.shape[1], 2, 2)
            vec_mins(self.tik_instance, "float16", coord, self.rois_ub,
                     self.shape[0], 3, 3)
            # area with border
            vec_sub(self.tik_instance, "float16", coord, coord, coord, 3, 3,
                    1)
            vec_adds(self.tik_instance, "float16", coord, coord, self.bsz01,
                     3, 3)
            vec_sub(self.tik_instance, "float16", coord, coord, coord, 2, 2,
                    0)
            vec_adds(self.tik_instance, "float16", coord, coord, self.bsz01,
                     2, 2)
            vec_mul(self.tik_instance, "float16", coord, coord, coord, 3, 3,
                    2)
            # area without border
            vec_sub(self.tik_instance, "float16", coord, self.rois_ub,
                    self.rois_ub, 1, 3, 1)
            vec_adds(self.tik_instance, "float16", coord, coord, self.bsz01,
                     1, 1)
            vec_sub(self.tik_instance, "float16", coord, self.rois_ub,
                    self.rois_ub, 0, 2, 0)
            vec_adds(self.tik_instance, "float16", coord, coord, self.bsz01,
                     0, 0)
            vec_mul(self.tik_instance, "float16", coord, coord, coord, 1, 1,
                    0)
            vec_div(self.tik_instance, "float16", coord, coord, coord, 3, 3,
                    1)
            vec_cmp(self.tik_instance, "LT", coord, conf_data,
                    1.0 - self.allow_border_ratio, 3, 4)

    def compute_argmax(self, conf_data, class_max):
        threshold_ub = self.tik_instance.Tensor("float16", (128,),
                                                name="threshold_ub",
                                                scope=tik.scope_ubuf)
        prob_tmp = self.tik_instance.Tensor("float16", (128,),
                                            name="prob_tmp",
                                            scope=tik.scope_ubuf)
        class_tmp = self.tik_instance.Tensor("float16", (128,),
                                             name="class_tmp",
                                             scope=tik.scope_ubuf)

        # initialize max score and class index
        score_max = self.tik_instance.Tensor("float16", (1, self.length),
                                             name="score_max",
                                             scope=tik.scope_ubuf)
        vec_dup(self.tik_instance, "float16", score_max, -1, 0)
        vec_dup(self.tik_instance, "float16", class_max, -1, 0)
        vec_dup(self.tik_instance, "float16", conf_data, 0, 4)

        self.compute_argmax_process(conf_data, class_max, threshold_ub,
            prob_tmp, class_tmp, score_max)

    def compute_argmax_process(self, conf_data, class_max, threshold_ub,
            prob_tmp, class_tmp, score_max):
        # compute argmax
        for c_num in range(1, self.num_class + 1):
            self.tik_instance.vector_dup(128, class_tmp, c_num - 1, 1, 1, 8)
            self.tik_instance.vector_dup(128, threshold_ub,
                                        self.threshold_class[c_num - 1],
                                        1, 1, 8)
            with self.tik_instance.for_range(0, self.repeat) as i:
                # prob sub threshold
                self.tik_instance.vsub(128, prob_tmp,
                                       self.score_ub[c_num, i * 128],
                                       threshold_ub, 1, 1, 1, 1, 8, 8, 8)
                # compare prob with max score
                cmpmask = self.tik_instance.vcmp_gt(128, prob_tmp,
                                                    score_max[i * 128], 1, 1)
                # max score
                self.tik_instance.vsel(128, 0, score_max[i * 128], cmpmask,
                                       prob_tmp, score_max[i * 128],
                                       1, 1, 1, 1, 8, 8, 8)
                # class_index
                self.tik_instance.vsel(128, 0, class_max[i * 128], cmpmask,
                                       class_tmp, class_max[i * 128],
                                       1, 1, 1, 1, 8, 8, 8)
                # update confidence
                self.tik_instance.vsel(128, 0, conf_data[4, i * 128], cmpmask,
                                       self.score_ub[c_num, i * 128],
                                       conf_data[4, i * 128], 1, 1, 1, 1, 8,
                                       8, 8)
            # prob sub threshold
            self.tik_instance.vsub(self.tail, prob_tmp,
                                   self.score_ub[c_num, self.repeat * 128],
                                   threshold_ub,
                                   1, 1, 1, 1, 8, 8, 8)
            # compare prob with max score
            cmpmask = self.tik_instance.vcmp_gt(self.tail, prob_tmp,
                                                score_max[self.repeat * 128],
                                                1, 1)
            # max score
            self.tik_instance.vsel(self.tail, 0, score_max[self.repeat * 128],
                                   cmpmask, prob_tmp,
                                   score_max[self.repeat * 128], 1, 1, 1, 1,
                                   8, 8, 8)
            # class_index
            self.tik_instance.vsel(self.tail, 0, class_max[self.repeat * 128],
                                   cmpmask, class_tmp,
                                   class_max[self.repeat * 128], 1, 1, 1, 1,
                                   8, 8, 8)
            # update confidence
            self.tik_instance.vsel(self.tail, 0,
                                   conf_data[4, self.repeat * 128], cmpmask,
                                   self.score_ub[c_num, self.repeat * 128],
                                   conf_data[4, self.repeat * 128],
                                   1, 1, 1, 1, 8, 8, 8)
        # if max score less than threshold, set confidence to zero
        vec_cmp(self.tik_instance, "LT", score_max, conf_data, 0, 0, 4)

    def rcnn_cmp_conf_bbox(self, conf_and_bbox_data):
        class_max = self.tik_instance.Tensor("float16",
            (1, nms.ceil_div(self.num_rois, 16) * 16),
            name="class_max",
            scope=tik.scope_ubuf)
        self.compute_argmax(conf_and_bbox_data, class_max)
        # filter according to background probabity
        vec_cmp(self.tik_instance, "GT", self.score_ub, conf_and_bbox_data,
                self.threshold_background, 0, 4)
        # filter according to border
        if self.allow_border >= 0.0 or self.allow_border_ratio >= 0.0:
            self.filter_according_border(conf_and_bbox_data)
        # adjust bbox according to coord_pred
        self.adjust_bbox_coor(class_max, conf_and_bbox_data)
        # filter according to minimum size constraint
        self.mini_size_filter(conf_and_bbox_data)
        # remove bbox whose confidence is zero
        for i in range(4):
            vec_cmp(self.tik_instance, "EQ", conf_and_bbox_data,
                    conf_and_bbox_data, 0, 4, i)


class NMS:
    def __init__(self, tik_instance, input_data, nms_param):
        self.tik_instance = tik_instance
        self.conf_and_bbox = input_data[0]
        self.class_score = input_data[1]
        self.num_rois = input_data[2]
        self.num_class = input_data[3]
        self.bsz01 = input_data[4]
        self.image_id = input_data[5]
        self.nms_max_candidate_n = nms_param[0]
        self.nms_top_n = nms_param[1]
        self.nms_overlap = nms_param[2]

        # proposals is made of x1, y1, x2, y2, score, xx, xx, xx
        self.proposals = tik_instance.Tensor("float16",
            (nms.ceil_div(self.num_rois, 16) * 16 * 2, 8),
            name="proposals",
            scope=tik.scope_ubuf)
        # class_score is made of back_ground, cls1, cls2, cls3, score, idx
        self.sorted_score = tik_instance.Tensor("float16",
            (nms.ceil_div(self.num_rois, 16) * 16 * 2, 8),
            name="sorted_score",
            scope=tik.scope_ubuf)

    def vec_trans(self, data_in, data_out):
        loop_num = nms.ceil_div(self.num_rois, 16)
        dst_ub = self.tik_instance.Tensor("float16", (16, 16), tik.scope_ubuf,
                                          "dst_ub")
        src_ub = self.tik_instance.Tensor("float16", (16, 16), tik.scope_ubuf,
                                          "src_ub")
        self.tik_instance.vector_dup(128, src_ub, 0, 2, 1, 8)
        with self.tik_instance.for_range(0, loop_num) as i:
            self.tik_instance.tensor_mov(src_ub, data_in[i * 16], "",
                                         data_in.shape[0], 1, 0, loop_num - 1)
            self.tik_instance.vtranspose(dst_ub, src_ub)
            with self.tik_instance.for_range(0, 16) as j:
                self.tik_instance.tensor_mov(data_out[16 * 2 * i + 2 * j, 0],
                                             dst_ub[j, 0], "", 1, 1, 0, 0)

    def sort_by_score(self, in_proposals):
        vrpsort_repeat = nms.ceil_div(self.num_rois, 16) * 2
        output_proposals = self.tik_instance.Tensor("float16",
                                                    in_proposals.shape,
                                                    name="output_proposals",
                                                    scope=tik.scope_ubuf)
        self.tik_instance.vrpsort16(output_proposals, in_proposals,
                                    vrpsort_repeat)

        src_list = []
        src_list_lengths = [16 for i in range(4)]
        for i in range(0, 4):
            src_list.append(output_proposals[i * 16, 0])
        self.tik_instance.vmrgsort4(in_proposals, src_list, src_list_lengths,
                                    False, 15, vrpsort_repeat // 4)

        src_list = []
        src_list_lengths = [64 for i in range(4)]
        for i in range(0, 4):
            src_list.append(in_proposals[i * 64, 0])
        self.tik_instance.vmrgsort4(output_proposals, src_list,
                                    src_list_lengths, False, 15,
                                    vrpsort_repeat // 4 // 4)

        src_list = []
        src_list_lengths = [512, 64, 16, 16]
        src_list.append(output_proposals[0, 0])
        src_list.append(output_proposals[512, 0])
        src_list.append(output_proposals[576, 0])
        src_list.append(output_proposals[592, 0])
        self.tik_instance.vmrgsort4(in_proposals, src_list, src_list_lengths,
                                    False, 15, 1)

    def perform_nms(self, out_proposals):
        # transpose
        self.vec_trans(self.conf_and_bbox, self.proposals)
        # sort by score
        with self.tik_instance.new_stmt_scope():
            self.sort_by_score(self.proposals)

        # temp class_score
        temp_class_score = self.tik_instance.Tensor("float16",
                                                    (self.num_class + 2,
                                                     nms.ceil_div(
                                                         self.num_rois,
                                                         16) * 16),
                                                    name="temp_class_score",
                                                    scope=tik.scope_ubuf)
        # class score
        with self.tik_instance.for_range(0, self.num_class + 1) as i:
            self.tik_instance.tensor_mov(temp_class_score[i, 0],
                                         self.class_score[i, 0], "", 1,
                                         temp_class_score.shape[1] // 16, 0,
                                         0)
        # max_class score
        self.tik_instance.tensor_mov(temp_class_score[4, 0],
                                     self.conf_and_bbox[4, 0], "", 1,
                                     temp_class_score.shape[1] // 16, 0, 0)

        # remove class score whose confidence is zero
        vec_cmp(self.tik_instance, "EQ", self.conf_and_bbox, temp_class_score,
                0, 4, 0)
        vec_cmp(self.tik_instance, "EQ", self.conf_and_bbox, temp_class_score,
                0, 4, 1)
        vec_cmp(self.tik_instance, "EQ", self.conf_and_bbox, temp_class_score,
                0, 4, 2)
        vec_cmp(self.tik_instance, "EQ", self.conf_and_bbox, temp_class_score,
                0, 4, 3)

        # transpose
        self.vec_trans(temp_class_score, self.sorted_score)
        # sort by score
        with self.tik_instance.new_stmt_scope():
            self.sort_by_score(self.sorted_score)

        # count max_class no-zero number
        self.count_max_class_nozero_number(out_proposals, temp_class_score)

    def count_max_class_nozero_number(self, out_proposals, temp_class_score):
        mask = self.tik_instance.Tensor("float16",
                                        (1, temp_class_score.shape[1]),
                                        name="mask", scope=tik.scope_ubuf)
        vec_dup(self.tik_instance, "float16", mask, 1, 0)
        vec_cmp(self.tik_instance, "EQ", temp_class_score, mask, 0, 4, 0)
        nozeros = self.tik_instance.Tensor("float16",
                                           ((mask.shape[1] + 127) // 128,),
                                           name="nozeros",
                                           scope=tik.scope_ubuf)
        self.tik_instance.vcadd(128, nozeros, mask,
                                (mask.shape[1] + 127) // 128 - 1, 1, 1, 8)
        self.tik_instance.vcadd(mask.shape[1] % 128,
                                nozeros[(mask.shape[1] + 127) // 128 - 1],
                                mask[
                                    128 * ((mask.shape[1] + 127) // 128 - 1)],
                                1, 1, 1, 8)

        self.tik_instance.vcadd((mask.shape[1] + 127) // 128, nozeros,
                                nozeros, 1, 1, 1, 8)
        nozeros_int = self.tik_instance.Tensor("int32",
            ((mask.shape[1] + 127) // 128,),
            name="nozeros_int", scope=tik.scope_ubuf)
        self.tik_instance.vconv(1, "round", nozeros_int, nozeros, 1, 1, 1, 8,
                                4)
        num_remain = self.tik_instance.Scalar("int32")
        num_remain.set_as(nozeros_int[0])

        down_factor = 1
        burst_input_proposal_num = min(16 * 16, self.nms_max_candidate_n)
        with self.tik_instance.if_scope(
                num_remain > self.nms_max_candidate_n):
            num_remain.set_as(self.nms_max_candidate_n)

        # perform nms
        one_core_nms = nms.OneCoreNMS(self.tik_instance,
                                      (num_remain, self.nms_top_n,
                                       burst_input_proposal_num, down_factor))
        sup_vector = one_core_nms.nms_single_core(self.proposals,
                                                  self.nms_overlap)

        # unsuppressed proposals according to suppression vector
        selected_proposal = self.tik_instance.Scalar("uint16")
        selected_proposal.set_as(0)
        vec_dup(self.tik_instance, "float16", out_proposals, 0, 0)
        with self.tik_instance.for_range(0, num_remain) as i:
            with self.tik_instance.if_scope(
                    tik.all(sup_vector[i] == 0,
                            selected_proposal < self.nms_top_n)):
                # image_index
                out_proposals[selected_proposal * 9].set_as(self.image_id)
                # bbox
                with self.tik_instance.for_range(0, 4) as j:
                    out_proposals[selected_proposal * 9 + (j + 1)].set_as(
                        self.proposals[i, j])
                # class score
                with self.tik_instance.for_range(0, self.num_class + 1) as j:
                    out_proposals[selected_proposal * 9 + (j + 5)].set_as(
                        self.sorted_score[i, j])
                # update counter
                selected_proposal.set_as(selected_proposal + 1)


class RCNNProposal:
    """
    RCNNProposal
    """

    def __init__(self, cls_score_softmax, bbox_pred, rois, im_info, bboxes,
                 tik_instance, attr):
        self.batch = attr[0]
        self.channel = attr[1]
        self.bbox_mean = attr[2]
        self.bbox_std = attr[3]
        self.num_class = attr[4]
        self.rpn_proposal_output_score = attr[5]
        self.regress_agnostic = attr[6]
        self.min_size_h = attr[7]
        self.min_size_w = attr[8]
        self.min_size_mode_and_else_or = attr[9]
        self.threshold_objectness = attr[10]
        self.threshold = attr[11]
        self.refine_out_of_map_bbox = attr[12]
        self.nms_overlap = attr[13]
        self.nms_top_n = attr[14]
        self.max_candidate_n = attr[15]
        self.bsz01 = attr[16]
        self.all_border = attr[17]
        self.all_border_ratio = attr[18]
        self.do_bbox_norm = attr[19]

        self.tik_instance = tik_instance
        self.ask_for_gm()

    def ask_for_gm(self):
        # class score (n, num_rois, class + 1, 1)
        padding = 16 - (self.num_class + 1)
        occupation = self.batch * self.channel * (
                    self.num_class + 1) + padding
        self.score_gm = self.tik_instance.Tensor("float16", (occupation,),
                                                 name="score_gm",
                                                 scope=tik.scope_gm)
        # bbox pred (n, num_rois, 8 or 16, 1)
        self.cords_dim = 2 * 4 if self.regress_agnostic else \
            (self.num_class + 1) * 4
        padding = 16 - self.cords_dim
        occupation = self.batch * self.channel * self.cords_dim + padding
        self.bbox_pred_gm = self.tik_instance.Tensor("float16", (occupation,),
                                                     name="bbox_pred_gm",
                                                     scope=tik.scope_gm)
        # rois shape is (n, num_rois, 5+3, 1)
        padding = 16 - 8
        occupation = self.batch * self.channel * 8 + padding
        self.rois_gm = self.tik_instance.Tensor("float16", (occupation,),
                                                name="rois_gm",
                                                scope=tik.scope_gm)
        # im_info shape is (n, 6, 1, 1)
        padding = 16 - 6
        occupation = self.batch * 6 + padding
        self.im_info_gm = self.tik_instance.Tensor("float16", (occupation,),
                                                   name="im_info_gm",
                                                   scope=tik.scope_gm)
        # bboxes shape is (n, top_k, 9, 1)
        padding = nms.ceil_div(self.nms_top_n * (6 + self.num_class),
                               16) * 16 - self.nms_top_n * (
                              6 + self.num_class)
        occupation = self.batch * self.nms_top_n * \
                     (6 + self.num_class) + padding
        self.bboxes_gm = self.tik_instance.Tensor("float16", (occupation,),
                                                  name="bboxes_gm",
                                                  scope=tik.scope_gm)

    def trans_data(self, tensor_gm, tensor_ub, block_id, num_rois):
        for i in range(tensor_ub.shape[0]):
            vec_dup(self.tik_instance, "float16", tensor_ub, 0, i)

        dst_ub = self.tik_instance.Tensor("float16", (16, 16), name="dst_ub",
                                          scope=tik.scope_ubuf)
        src_ub = self.tik_instance.Tensor("float16", (16, 16), name="src_ub",
                                          scope=tik.scope_ubuf)
        loop_num = num_rois // 16
        tail_num = num_rois % 16
        channel_stride = tensor_gm.shape[0] // (self.batch * self.channel)
        batch_stride = channel_stride * self.channel
        with self.tik_instance.for_range(0, loop_num) as i:
            with self.tik_instance.for_range(0, 16) as j:
                self.tik_instance.tensor_mov(src_ub[j, 0],
                                             tensor_gm[
                                                 block_id * batch_stride + (
                                                             i * 16 + j) *
                                                 channel_stride],
                                             "", 1, 1, 0, 0)
            self.tik_instance.vtranspose(dst_ub, src_ub)
            self.tik_instance.tensor_mov(tensor_ub[0, i * 16], dst_ub, "",
                                         tensor_ub.shape[0], 1,
                                         tensor_ub.shape[1] // 16 - 1, 0)
        # tail process
        self.tik_instance.vector_dup(128, src_ub, 0, 2, 1, 8)
        with self.tik_instance.if_scope(tail_num != 0):
            with self.tik_instance.for_range(0, tail_num) as j:
                self.tik_instance.tensor_mov(src_ub[j, 0],
                                             tensor_gm[
                                                 block_id * batch_stride + (
                                                         loop_num * 16 + j) *
                                                 channel_stride],
                                             "", 1, 1, 0, 0)
            self.tik_instance.vtranspose(dst_ub, src_ub)
            self.tik_instance.tensor_mov(tensor_ub[0, loop_num * 16], dst_ub,
                                         "", tensor_ub.shape[0], 1,
                                         tensor_ub.shape[1] // 16 - 1, 0)

    def decode_rois_and_bbox(self, block_id, channel_align_up):
        img_height = self.tik_instance.Scalar(dtype="float16")
        img_height.set_as(self.im_info_ub[0])
        img_width = self.tik_instance.Scalar(dtype="float16")
        img_width.set_as(self.im_info_ub[1])
        # order: background, class 1, class 2, class 3, confidence
        conf_and_bbox = self.tik_instance.Tensor("float16",
                                                    (5, channel_align_up),
                                                    name="conf_and_bbox",
                                                    scope=tik.scope_ubuf)
        with self.tik_instance.new_stmt_scope():
            one_core_preprocess = PreProcess(self.tik_instance,
                (self.channel, (img_height, img_width),
                    self.all_border, self.all_border_ratio,
                    (self.min_size_h, self.min_size_w),
                    self.min_size_mode_and_else_or, self.threshold_objectness,
                    self.bsz01,
                    self.do_bbox_norm, self.bbox_mean, self.bbox_std,
                    self.refine_out_of_map_bbox,
                    self.regress_agnostic, self.num_class, self.threshold),
                (self.rois_ub, self.score_ub, self.bbox_pred_ub))
            one_core_preprocess.rcnn_cmp_conf_bbox(conf_and_bbox)

        # do nms
        img_id_tensor = self.tik_instance.Tensor("int32", (1,),
                                                    name="img_id_tensor",
                                                    scope=tik.scope_ubuf)
        img_id_tensor_half = self.tik_instance.Tensor("float16", (1,),
            name="img_id_tensor_half",
            scope=tik.scope_ubuf)
        img_id_tensor.set_as(block_id)
        self.tik_instance.vconv(1, "", img_id_tensor_half, img_id_tensor,
                                1, 1, 1, 4, 8, 1.0)
        img_id = self.tik_instance.Scalar("float16")
        img_id.set_as(img_id_tensor_half)
        with self.tik_instance.new_stmt_scope():
            one_core_nms = NMS(self.tik_instance,
                (conf_and_bbox, self.score_ub, self.channel, self.num_class,
                self.bsz01, img_id), (self.max_candidate_n, self.nms_top_n,
                self.nms_overlap))
            one_core_nms.perform_nms(self.bboxes_ub)

        # move bbox to gm
        self.tik_instance.tensor_mov(self.bboxes_gm[
                                            block_id * self.nms_top_n * (
                                                        6 + self.num_class)],
                                        self.bboxes_ub, "", 1,
                                        self.bboxes_ub.shape[1] // 16, 0, 0)

    def cce_process(self, kernel_name):
        with self.tik_instance.for_range(0, self.batch) as block_id:
            # get num_rois
            rois_get_num = self.tik_instance.Tensor("float16", (16,),
                                                    name="rois_get_num",
                                                    scope=tik.scope_ubuf)
            rois_get_num_int = self.tik_instance.Tensor("int32", (16,),
                name="rois_get_num_int",
                scope=tik.scope_ubuf)
            self.tik_instance.tensor_mov(rois_get_num, self.rois_gm[
                block_id * self.channel * 8], "", 1, 1, 0, 0)
            self.tik_instance.vconv(16, "round", rois_get_num_int,
                                    rois_get_num, 1, 1, 1, 8, 4)
            num_rois = self.tik_instance.Scalar("int32")
            num_rois.set_as(rois_get_num_int[5])

            channel_align_up = nms.ceil_div(self.channel, 16) * 16
            self.score_ub = self.tik_instance.Tensor("float16",
                (self.num_class + 1, channel_align_up), name="score_ub",
                scope=tik.scope_ubuf)
            self.bbox_pred_ub = self.tik_instance.Tensor("float16",
                (self.cords_dim, channel_align_up), name="bbox_pred_ub",
                scope=tik.scope_ubuf)
            self.rois_ub = self.tik_instance.Tensor("float16",
                                                    (4, channel_align_up),
                                                    name="rois_ub",
                                                    scope=tik.scope_ubuf)
            self.im_info_ub = self.tik_instance.Tensor("float16", (16,),
                                                       name="im_info_ub",
                                                       scope=tik.scope_ubuf)
            self.bboxes_ub = self.tik_instance.Tensor("float16", (
            1, nms.ceil_div(self.nms_top_n * (6 + self.num_class), 16) * 16),
                                                      name="bboxes_ub",
                                                      scope=tik.scope_ubuf)

            # tensor mov from gm to ub
            self.trans_data(self.score_gm, self.score_ub, block_id, num_rois)
            self.trans_data(self.bbox_pred_gm, self.bbox_pred_ub, block_id,
                            num_rois)
            self.trans_data(self.rois_gm, self.rois_ub, block_id, num_rois)
            # im_info
            self.tik_instance.tensor_mov(self.im_info_ub, self.im_info_gm, "",
                                         1, 1, 0, 0)
            self.tik_instance.vadds(2, self.im_info_ub, self.im_info_ub, -1,
                                    1, 1, 1, 8, 8)

            # decode rois and bbox
            self.decode_rois_and_bbox(block_id, channel_align_up)

        self.tik_instance.BuildCCE(kernel_name,
                                   inputs=[self.score_gm, self.bbox_pred_gm,
                                           self.rois_gm, self.im_info_gm],
                                   outputs=[self.bboxes_gm])
        return self.tik_instance


def param_check(batch, channel, num_class, rpn_proposal_output_score,
    threshold, bbox_reg_mean, bbox_reg_std, nms_overlap, threshold_objectness,
    nms_top_n, min_size_h, min_size_w, allow_border, allow_border_ratio,
    max_candidate_n, bsz01):    
    log.check_gt(batch, 0, "batch should be greater than 0")
    log.check_eq(channel, 300, "channel should be equal to 300")
    log.check_eq(num_class, 3, "num_class should be equal to 3")
    log.check(not rpn_proposal_output_score,
              "rpn_proposal_output_score must be true")
    log.check_eq(len(threshold), num_class,
                 "threshold size should be equal to num_class")
    log.check_gt(nms_overlap, 0, "nms_overlap should be greater than 0")
    log.check_lt(nms_overlap, 1, "nms_overlap should be less than 1")
    log.check_gt(threshold_objectness, 0,
                 "threshold_objectness should be greater than 0")
    log.check_lt(threshold_objectness, 1,
                 "threshold_objectness should be less than 1")
    for thres in threshold:
        log.check_gt(thres, 0, "threshold should be greater than 0")
        log.check_lt(thres, 1, "threshold should be less than 1")
    log.check_gt(nms_top_n, 0, "nms_top_n should be greater than 0")
    log.check_gt(min_size_h, 0, "min_size_h should be greater than 0")
    log.check_gt(min_size_w, 0, "min_size_w should be greater than 0")
    log.check_ge(allow_border, 0,
                 "allow_border should be greater than or equal to 0")
    log.check_ge(allow_border_ratio, 0,
                 "allow_border_ratio should be greater than or equal to 0")
    log.check_ge(max_candidate_n, 0,
                 "max_candidate_n should be greater than or equal to 0")
    log.check_ge(bsz01, 0, "bsz01 should be greater than or equal to 0")


def rcnn_proposal(cls_score_softmax, bbox_pred, rois, im_info, bboxes,
                  batch=3, channel=300, bbox_reg_mean=(0, 0, 0, 0),
                  bbox_reg_std=(0.1, 0.1, 0.2, 0.2), num_class=3,
                  rpn_proposal_output_score=True, regress_agnostic=False,
                  min_size_h=8.800800, min_size_w=8.800800,
                  min_size_mode_and_else_or=True, threshold_objectness=0.1,
                  threshold=(0.1, 0.1, 0.1),
                  refine_out_of_map_bbox=True, nms_overlap=0.5, nms_top_n=5,
                  max_candidate_n=300, bsz01=1,
                  allow_border=10.0, allow_border_ratio=0.20,
                  kernel_name="rcnn_proposal"):
    # compile operator
    tik_instance = tik.Tik(tik.Dprofile())

    min_size_mode_and_else_or = False if \
        min_size_mode_and_else_or == "HEIGHT_OR_WIDTH" else True
    nms_overlap = nms_overlap[0]
    nms_top_n = nms_top_n[0]
    max_candidate_n = max_candidate_n[0]
    # parameter check
    param_check(batch, channel, num_class, rpn_proposal_output_score,
        threshold, bbox_reg_mean, bbox_reg_std, nms_overlap, 
        threshold_objectness, nms_top_n, min_size_h, min_size_w, 
        allow_border, allow_border_ratio, max_candidate_n, bsz01)
    do_bbox_norm = False
    if len(bbox_reg_mean) > 0 and len(bbox_reg_std) > 0:
        do_bbox_norm = True
        log.check_eq(len(bbox_reg_mean), 4,
                     "bbox_reg_mean size should be equal to 4 or 0")
        log.check_eq(len(bbox_reg_std), 4,
                     "bbox_reg_std size should be equal to 4 or 0")

    rcnn_proposal_result = RCNNProposal(cls_score_softmax, bbox_pred, rois,
                                        im_info, bboxes, tik_instance, (
                                        batch, channel, bbox_reg_mean,
                                        bbox_reg_std, num_class,
                                        rpn_proposal_output_score,
                                        regress_agnostic, min_size_h,
                                        min_size_w,
                                        min_size_mode_and_else_or,
                                        threshold_objectness, threshold,
                                        refine_out_of_map_bbox, nms_overlap,
                                        nms_top_n, max_candidate_n, bsz01,
                                        allow_border, allow_border_ratio,
                                        do_bbox_norm))

    return rcnn_proposal_result.cce_process(kernel_name)
