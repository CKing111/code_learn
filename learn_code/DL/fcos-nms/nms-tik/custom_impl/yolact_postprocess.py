#!/usr/bin/env python
# coding: utf-8
from te import tik
import numpy as np 
import multi_class_nms
from util import OpLog as log


class YolactPostprocess():
    def __init__(self, box_shape, score_shape, mask_shape, anchor_shape, 
                 merge_shape, width, height, num_class, score_threshold,
                 iou_threshold, variance_xy, variance_wh, topk,
                 max_output_size, kernel_name):
        self.dtype = "float16"
        self.tik_inst = tik.Tik(tik.Dprofile())

        self.num_class = num_class
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.variance_xy = variance_xy
        self.variance_wh = variance_wh
        self.down_filter = 120
        self.width = width 
        self.height = height
        self.kernel_name = kernel_name
        self.mask_shape = mask_shape
        self.max_output_size = max_output_size
        self.slice_len = 1024 * 4
        self.topk = topk
        self.cls_out_num = 32

        #input_tensor
        self.box_gm = self.tik_inst.Tensor(self.dtype,
                      (self.get_shape_size(box_shape) + 16, ),
                      name="box_gm", scope=tik.scope_gm)
        self.score_gm = self.tik_inst.Tensor(self.dtype,
                        (self.get_shape_size(score_shape) + 16, ),
                        name="score_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik_inst.Tensor(self.dtype, 
                       (self.get_shape_size(mask_shape) + 16, ),
                       name="mask_gm", scope=tik.scope_gm)
        self.anchor_gm = self.tik_inst.Tensor(self.dtype,
                         (self.get_shape_size(anchor_shape) + 16, ),
                         name="anchor_gm", scope=tik.scope_gm)

        self.all_element = box_shape[3]
        self.all_element_align_16 = self.ceil_div_offline(self.all_element, 16)

        self.merge_gm = self.tik_inst.Tensor(self.dtype,
                        (merge_shape[0] + 16, ),
                        name="merge_gm", scope=tik.scope_gm)

        self.output_proposal_gm = self.tik_inst.Tensor(self.dtype,
             (self.max_output_size * 8, ), 
             name="output_proposal_gm", scope=tik.scope_gm)
        self.output_mask_gm = self.tik_inst.Tensor(self.dtype,
                              (mask_shape[1] * self.max_output_size + 16, ), 
                              name="output_mask_gm", scope=tik.scope_gm)
        self.output_coord_gm = self.tik_inst.Tensor(self.dtype,
                               (4 * self.max_output_size + 16, ),
                               name="output_coord_gm", scope=tik.scope_gm)

        self.multi_class_nms_obj = multi_class_nms.MultiClassNms(
             self.tik_inst, self.merge_gm, self.dtype,
             self.all_element_align_16 * 16, 
             self.num_class, self.width, self.height, self.max_output_size,
             self.iou_threshold, self.topk, self.cls_out_num, self.down_filter)

    def ceil_div_offline(self, value, factor):
        result = (value + (factor - 1)) // factor
        return result

    def get_shape_size(self, shape):
        return shape[0] * shape[1] * shape[2] * shape[3]

    def gen_ub(self, data_type, length, ub_name):
        return self.tik_inst.Tensor(data_type, (length, ), name=ub_name,
                                        scope=tik.scope_ubuf)

    def cal_coord(self, input_data, output_data, anchor_data, variance_xy,
                  variance_wh, coord_ub, anchor0_ub, anchor1_ub, tmp0_ub,
                  tmp1_ub, coord_flag, offset, loop_length_align_128,
                  loop_length_align_16):
        """
        flag: 0 process w, x; flag: 1 process h, y
        """
        tmp_scalar = self.tik_inst.Scalar(self.dtype)
        tmp_scalar.set_as(0.5)
        input_offset = self.tik_inst.Scalar("int32")

        input_offset.set_as(coord_flag * self.all_element + offset)
        self.tik_inst.data_move(anchor0_ub, anchor_data[(coord_flag) *
             self.all_element + offset], 0, 1, loop_length_align_16, 1, 1)
        self.tik_inst.data_move(anchor1_ub, anchor_data[(coord_flag + 2) *
             self.all_element + offset], 0, 1, loop_length_align_16, 1, 1)
        self.tik_inst.data_move(coord_ub, input_data[input_offset], 0, 1,
             loop_length_align_16, 1, 1)

        self.tik_inst.vmuls(128, tmp0_ub, coord_ub, variance_xy,
                            loop_length_align_128, 1, 1, 8, 8)
        self.tik_inst.vmul(128, tmp0_ub, tmp0_ub, anchor1_ub,
                           loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(128, tmp0_ub, tmp0_ub, anchor0_ub,
                           loop_length_align_128, 1, 1, 1, 8, 8, 8)

        input_offset.set_as((coord_flag + 2) * self.all_element + offset)
        self.tik_inst.data_move(coord_ub, input_data[input_offset],
                                0, 1, loop_length_align_16, 1, 1)
        self.tik_inst.vmuls(128, tmp1_ub, coord_ub, variance_wh,
                            loop_length_align_128, 1, 1, 8, 8)
        self.tik_inst.vexp(128, tmp1_ub, tmp1_ub, loop_length_align_128,
                           1, 1, 8, 8)
        self.tik_inst.vmul(128, tmp1_ub, tmp1_ub, anchor1_ub,
                           loop_length_align_128, 1, 1, 1, 8, 8, 8)

        # xywh --> x1y1x2y2
        self.tik_inst.vmuls(128, tmp1_ub, tmp1_ub, tmp_scalar,
                            loop_length_align_128, 1, 1, 8, 8)
        self.tik_inst.vsub(128, coord_ub, tmp0_ub, tmp1_ub,
                           loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(128, tmp0_ub, tmp0_ub, tmp1_ub,
                           loop_length_align_128, 1, 1, 1, 8, 8, 8)

        #move to gm  ---> x1, y1, x2, y2
        output_offset = coord_flag * self.all_element_align_16 * 16 + offset
        self.tik_inst.data_move(output_data[output_offset], coord_ub, 0, 1,
                                loop_length_align_16, 1, 1)
        output_offset = output_offset + 2 * self.all_element_align_16 * 16
        self.tik_inst.data_move(output_data[output_offset], tmp0_ub, 0, 1,
                                loop_length_align_16, 1, 1)

    def score_filter(self, output_data, prob_max_ub, prob_thres_ub, zero_ub,
                    class_ub, loop_idx, loop_times, loop_length,
                    loop_length_align_128, loop_length_align_16, offset):
        zero_scalar = self.tik_inst.Scalar(self.dtype)
        zero_scalar.set_as(0.0)
        class_bg_ub = self.gen_ub(self.dtype, 128, "class_background_ub")
        class_bg_scalar = self.tik_inst.Scalar(self.dtype)
        class_bg_scalar.set_as(self.num_class + 1)
        self.tik_inst.vector_dup(128, class_bg_ub, class_bg_scalar, 1, 1, 8)

        #filt with score thres 
        with self.tik_inst.for_range(0, loop_length_align_128) as i:
            cmpack = self.tik_inst.vcmp_ge(128, prob_max_ub[128 * i],
                                           prob_thres_ub, 1, 1)
            self.tik_inst.vsel(128, 0, prob_max_ub[128 * i], cmpack,
                 prob_max_ub[128 * i], zero_ub, 1, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vsel(128, 0, class_ub[128 * i], cmpack,
                               class_ub[128 * i], class_bg_ub, 1,
                               1, 1, 1, 8, 8, 8) 
        with self.tik_inst.if_scope(loop_idx == (loop_times-1)):
            remain_element = self.all_element_align_16 * 16 - self.all_element
            with self.tik_inst.if_scope(remain_element > 0):
                with self.tik_inst.for_range(0, remain_element) as rem_idx:
                    prob_max_ub[loop_length+rem_idx].set_as(zero_scalar)
                    class_ub[loop_length+rem_idx].set_as(class_bg_scalar)

        #data move to output_gm 
        output_offset = 4 * self.all_element_align_16 * 16 + offset 
        self.tik_inst.data_move(output_data[output_offset], prob_max_ub,
                                    0, 1, loop_length_align_16, 1, 1)
        output_offset = output_offset + self.all_element_align_16 * 16
        self.tik_inst.data_move(output_data[output_offset], class_ub, 0, 1,
                                loop_length_align_16, 1, 1)

    def process_score(self, input_data, output_data, loop_times, loop_length,
                      input_offset, loop_idx):
        prob_ub = self.gen_ub(self.dtype, self.slice_len, "proob_ub")
        prob_max_ub = self.gen_ub(self.dtype, self.slice_len, "prob_max_ub")
        prob_thres_ub = self.gen_ub(self.dtype, 128, "prob_thres_ub")
        zero_ub = self.gen_ub(self.dtype, 128, "zero_ub")
        class_ub = self.gen_ub(self.dtype, self.slice_len, "class_ub")
        cur_class_ub = self.gen_ub(self.dtype, self.slice_len, "cur_class_ub")

        offset = loop_idx * self.slice_len
        loop_length.set_as(self.slice_len)
        with self.tik_inst.if_scope(loop_idx == (loop_times-1)):
            loop_length.set_as(self.all_element - offset)
        loop_length_align_128 = self.ceil_div_offline(loop_length, 128)
        loop_length_align_16 = self.ceil_div_offline(loop_length, 16)

        self.tik_inst.vector_dup(128, prob_thres_ub, 
                                 self.score_threshold, 1, 1, 8)
        self.tik_inst.vector_dup(128, zero_ub, 0.0, 1, 1, 8)
        self.tik_inst.vector_dup(128, class_ub, 0, loop_length_align_128, 1, 8)
        self.tik_inst.vector_dup(128, cur_class_ub, 0,
                                 loop_length_align_128, 1, 8)
        self.tik_inst.vector_dup(128, prob_max_ub, 0,
                                 loop_length_align_128, 1, 8)

        with self.tik_inst.for_range(1, (self.num_class + 1)) as cls_idx:
            input_offset.set_as(cls_idx * self.all_element + offset)
            with self.tik_inst.if_scope(cls_idx == 1):
                self.tik_inst.data_move(prob_max_ub, 
                    input_data[input_offset], 0, 1, loop_length_align_16, 1, 1)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(prob_ub, input_data[input_offset],
                                        0, 1, loop_length_align_16, 1, 1)
                self.tik_inst.vadds(128, cur_class_ub, cur_class_ub, 1.0,
                                    loop_length_align_128, 1, 1, 8, 8)

                with self.tik_inst.for_range(0, loop_length_align_128) as idx:
                    cmpack = self.tik_inst.vcmp_ge(128, prob_max_ub[128 * idx],
                                                   prob_ub[128 * idx], 1, 1)
                    self.tik_inst.vsel(128, 0, prob_max_ub[128 * idx], cmpack,
                         prob_max_ub[128 * idx], prob_ub[128 * idx],
                         1, 1, 1, 1, 8, 8, 8)
                    self.tik_inst.vsel(128, 0, class_ub[128 * idx], cmpack,
                         class_ub[128 * idx], cur_class_ub[128 * idx],
                         1, 1, 1, 1, 8, 8, 8)

        self.score_filter(output_data, prob_max_ub, prob_thres_ub, zero_ub,
                     class_ub, loop_idx, loop_times, loop_length,
                     loop_length_align_128, loop_length_align_16, offset)

    def process_coord_score(self, box_data, score_data,
                            output_data, anchor_data):
        loop_times = self.ceil_div_offline(self.all_element, self.slice_len)
        loop_length = self.tik_inst.Scalar("int32")
        input_offset1 = self.tik_inst.Scalar("int32")

        with self.tik_inst.for_range(
                  0, loop_times, block_num=loop_times) as loop_idx:
            offset = loop_idx * self.slice_len
            loop_length.set_as(self.slice_len)

            anchor0_ub = self.gen_ub(self.dtype, self.slice_len, "anchor0_ub")
            anchor1_ub = self.gen_ub(self.dtype, self.slice_len, "anchor1_ub")
            coord_ub = self.gen_ub(self.dtype, self.slice_len, "coord_ub")
            tmp0_ub = self.gen_ub(self.dtype, self.slice_len, "tmp0_ub")
            tmp1_ub = self.gen_ub(self.dtype, self.slice_len, "tmp1_ub")

            with self.tik_inst.if_scope(loop_idx == (loop_times-1)):
                loop_length.set_as(self.all_element - offset)

            loop_length_align_128 = self.ceil_div_offline(loop_length, 128)
            loop_length_align_16 = self.ceil_div_offline(loop_length, 16)

            #x1, x2
            self.cal_coord(box_data, output_data, anchor_data,
                           self.variance_xy, self.variance_wh, coord_ub,
                           anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub, 0, offset,
                           loop_length_align_128, loop_length_align_16)

            #y1, y2
            self.cal_coord(box_data, output_data, anchor_data,
                           self.variance_xy, self.variance_wh, coord_ub,
                           anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub, 1, offset,
                           loop_length_align_128, loop_length_align_16)

            self.process_score(score_data, output_data, loop_times,
                 loop_length, input_offset1, loop_idx)

    def get_mask_data(self, input_data, output_data, index_ub, mask_dim):
        mask_output_length = self.max_output_size * mask_dim
        mask_ub = self.tik_inst.Tensor(self.dtype, 
                       (self.ceil_div_offline(mask_output_length, 16) * 16, ),
                       name="mask_ub", scope=tik.scope_ubuf)

        in_index = self.tik_inst.Scalar(dtype="int32", name="in_index")
        out_index = self.tik_inst.Scalar(dtype="int32", name="out_index")
        offset = self.tik_inst.Scalar(dtype="int32", name="offset")
        tmp_ub = self.tik_inst.Tensor(self.dtype, (16, ),
                      name="tmp_ub", scope=tik.scope_ubuf)

        with self.tik_inst.for_range(0, mask_dim) as i:
            offset.set_as(self.all_element * i)
            with self.tik_inst.for_range(0, self.max_output_size) as j:
                in_index.set_as(index_ub[j])
                in_index.set_as(in_index + offset)

                out_index.set_as(j * mask_dim + i)

                self.tik_inst.data_move(tmp_ub, input_data[in_index],
                                            0, 1, 1, 0, 0)
                mask_ub[out_index].set_as(tmp_ub[0])

        self.tik_inst.data_move(output_data, mask_ub, 0, 1, 
             self.ceil_div_offline(mask_output_length, 16), 0, 0)

    def compile(self):
        self.process_coord_score(self.box_gm, self.score_gm,
             self.merge_gm, self.anchor_gm)

        #do each class nms
        ret, out_coord, out_idx_ub = self.multi_class_nms_obj.do_yolact_nms()

        self.tik_inst.data_move(self.output_proposal_gm, ret, 0, 1,
                                self.max_output_size * 8 // 16, 0, 0, 0)

        self.tik_inst.data_move(self.output_coord_gm, out_coord, 0, 1, 
                                self.max_output_size * 4 // 16, 0, 0, 0)
        self.get_mask_data(self.mask_gm, self.output_mask_gm, out_idx_ub,
                           self.mask_shape[1])

        #build
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, 
             inputs=(self.box_gm, self.score_gm, self.mask_gm,
             self.anchor_gm, self.merge_gm),
             outputs=(self.output_proposal_gm, self.output_mask_gm,
             self.output_coord_gm))
        return self.tik_inst


def param_check(box_shape, score_shape, mask_shape, anchor_shape, merge_shape,
                width, height, num_class, score_threshold, iou_threshold,
                variance_xy, variance_wh, topk, max_output_size, kernel_name):
    log.check_gt(num_class, 0, "num_class should be greater than 0")
    log.check_le(num_class, 100,
                "num_class should be less than or equal to 100")
    log.check_eq(box_shape[0], 1, "box_shape[0] should be equal to 1")
    log.check_eq(anchor_shape[0], 1, "anchor_shape[0] should be equal to 1")
    log.check_eq(anchor_shape[2], 1, "anchor_shape[2] should be equal to 1")
    log.check_eq(score_shape[0], 1, "score_shape[0] should be equal to 1")
    log.check_eq(score_shape[2], 1, "score_shape[2] should be equal to 1")
    log.check_eq(box_shape[1], 4, "box_shape[1] should be equal to 4")
    log.check_eq(anchor_shape[1], 4, "anchor_shape[1] should be equal to 4")
    log.check_eq(box_shape[3], score_shape[3],
                 "box_shape[3] should be equal to score_shape[3]")
    merge_size = (box_shape[3] + 15) // 16 * 16 * 6
    log.check_eq(merge_shape[0], merge_size,
        "merge_shape[0] should be equal (box_shape[3] + 15) // 16 * 16 * 6")
    #mask shape check
    log.check_eq(mask_shape[3], box_shape[3],
                "mask_shape[3] should be equal to box_shape[3]")
    log.check_gt(width, 0, "width should be greater than 0")
    log.check_gt(height, 0, "height should be greater than 0")
    log.check_gt(score_threshold, 0,
                "score_threshold should be greater than or equal to 0")
    log.check_lt(score_threshold, 1,
                "score_threshold should be less than or equal to 1")
    log.check_gt(iou_threshold, 0, "iou_threshold should be greater than 0")
    log.check_lt(iou_threshold, 1,
                "iou_threshold should be less than or equal to 1")
    log.check_gt(topk, 0, "topk should be greater than 0")
    log.check_le(topk, 512, "topk should be less than or equal to 512")
    log.check_gt(max_output_size, 0,
                "max_output_size should be greater than 0")
    log.check_le(max_output_size, 256,
        "max_output_size should be less than or equal to 256")
    log.check_gt(variance_xy, 0, "variance_xy should be greater than 0")
    log.check_gt(variance_wh, 0, "variance_wh should be greater than 0")

    log.check_le(len(kernel_name), 200,
        "the length of kernel_name should be less than or equal to 200")


def yolact_postprocess(boxes, scores, masks, anchors, merge_buffer,
           out_proporsals, out_masks, out_coords, width, height,
           score_threshold, iou_threshold, variance_xy, variance_wh,
           topk, max_output_size, kernel_name="yolact_postprocess"):
    """
    the compile function of yolact postprocess
    """
    box_shape = boxes.get('shape')
    score_shape = scores.get('shape')
    mask_shape = masks.get('shape')
    anchor_shape = anchors.get('shape')
    merge_shape = merge_buffer.get('shape')
    num_class = score_shape[1] - 1 
    param_check(box_shape, score_shape, mask_shape, anchor_shape, merge_shape,
                width, height, num_class, score_threshold, iou_threshold,
                variance_xy, variance_wh, topk, max_output_size, kernel_name)

    yolact_obj = YolactPostprocess(box_shape, score_shape, mask_shape,
                 anchor_shape, merge_shape, width, height, num_class,
                 score_threshold, iou_threshold, variance_xy, variance_wh,
                 topk, max_output_size, kernel_name)
    tik_inst = yolact_obj.compile()
    return tik_inst
