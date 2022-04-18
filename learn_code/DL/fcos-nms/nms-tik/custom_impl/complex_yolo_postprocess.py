#!/usr/bin/env python
# coding: utf-8
from te import tik
import numpy as np 
import multi_class_nms
from util import OpLog as log


class ComplexYoloPostprocess():
    def __init__(self, pred_shape, anchor_shape, merge_shape, width_stride,
                 height_stride, num_class, score_threshold, iou_threshold,
                 topk, max_output_size, kernel_name):
        self.tik_inst = tik.Tik(tik.Dprofile())
        self.dtype = "float16"

        self.pred_gm = self.tik_inst.Tensor(self.dtype, 
                       (self.get_shape_size(pred_shape) + 16, ),
                       name="pred_gm", scope=tik.scope_gm)
        self.anchor_gm = self.tik_inst.Tensor(self.dtype, 
                         (self.get_shape_size(anchor_shape) + 16, ),
                         name="anchor_gm", scope=tik.scope_gm)

        self.all_element = pred_shape[3]
        self.all_element_align_16 = self.ceil_div_offline(self.all_element, 16)
        self.merge_gm = self.tik_inst.Tensor(self.dtype,
                        (merge_shape[0] + 16, ), 
                        name="merge_gm", scope=tik.scope_gm)
        self.num_class = num_class
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.down_filter = 0.18

        self.width_stride = width_stride
        self.height_stride = height_stride
        self.kernel_name = kernel_name

        self.max_output_size = max_output_size
        self.slice_len = 512
        self.topk = topk
        self.cls_out_num = 32

        # retrun res: (x, y, w, h, im, re, obj_conf, class_score, class_id)
        self.output_proposal_gm = self.tik_inst.Tensor(self.dtype,
                                  (self.max_output_size * 9 + 16, ),
                                  name="output_proposal_gm",
                                  scope=tik.scope_gm)

        self.multi_class_nms_obj = multi_class_nms.MultiClassNms(
             self.tik_inst, self.merge_gm, self.dtype,
             self.all_element_align_16 * 16, self.num_class,
             self.width_stride, self.height_stride, self.max_output_size,
             self.iou_threshold, self.topk, self.cls_out_num, self.down_filter)

    def get_shape_size(self, shape):
        return shape[0] * shape[1] * shape[2] * shape[3]

    def ceil_div_offline(self, value, factor):
        result = (value + (factor - 1)) // factor
        return result

    def sigmoid(self, data_ub, length):
        scalar_negative_1 = self.tik_inst.Scalar(self.dtype)
        scalar_negative_1.set_as(-1.0)
        scalar_1 = self.tik_inst.Scalar(self.dtype)
        scalar_1.set_as(1.0)

        repeat_num = length // 128
        remain = length % 128
        offset = repeat_num * 128

        with self.tik_inst.if_scope(remain == 0):
            self.tik_inst.vmuls(128, data_ub, data_ub, scalar_negative_1,
                                    repeat_num, 1, 1, 8, 8)
            self.tik_inst.vexp(128, data_ub, data_ub, repeat_num,
                                   1, 1, 8, 8)
            self.tik_inst.vadds(128, data_ub, data_ub, scalar_1,
                                    repeat_num, 1, 1, 8, 8)
            self.tik_inst.vrec(128, data_ub, data_ub, repeat_num,
                                   1, 1, 8, 8)
        with self.tik_inst.else_scope():
            self.tik_inst.vmuls(128, data_ub, data_ub, scalar_negative_1,
                                    repeat_num, 1, 1, 8, 8)
            self.tik_inst.vmuls(remain, data_ub[offset], data_ub[offset],
                                    scalar_negative_1, 1, 1, 1, 8, 8)
            self.tik_inst.vexp(128, data_ub, data_ub, repeat_num,
                                   1, 1, 8, 8)
            self.tik_inst.vexp(remain, data_ub[offset], data_ub[offset],
                                   1, 1, 1, 8, 8)
            self.tik_inst.vadds(128, data_ub, data_ub, scalar_1,
                                    repeat_num, 1, 1, 8, 8)
            self.tik_inst.vadds(remain, data_ub[offset], data_ub[offset],
                                    scalar_1, 1, 1, 1, 8, 8)
            self.tik_inst.vrec(128, data_ub, data_ub, repeat_num,
                                   1, 1, 8, 8)
            self.tik_inst.vrec(remain, data_ub[offset], data_ub[offset],
                                   1, 1, 1, 8, 8)

    def cal_coord(self, input_data, output_data, anchor_data, coord_ub,
                  anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub, coord_flag,
                  offset, stride, loop_length_align_128, loop_length_align_16):
        """
        flag: 0 process w, x 
        flag: 1 process h, y 
        """
        input_offset = self.tik_inst.Scalar("int32")
        tmp_scalar = self.tik_inst.Scalar(self.dtype)
        tmp_scalar.set_as(0.5)

        input_offset.set_as(coord_flag * self.all_element + offset)
        self.tik_inst.data_move(anchor0_ub, anchor_data[(coord_flag) *
             self.all_element + offset], 0, 1, loop_length_align_16, 1, 1)
        self.tik_inst.data_move(anchor1_ub, anchor_data[(coord_flag + 2) *
             self.all_element + offset], 0, 1, loop_length_align_16, 1, 1)
        self.tik_inst.data_move(coord_ub, input_data[input_offset], 0, 1,
             loop_length_align_16, 1, 1)

        self.sigmoid(coord_ub, loop_length_align_128 * 128)

        self.tik_inst.vadd(128, tmp0_ub, coord_ub, anchor0_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmuls(128, tmp0_ub, tmp0_ub, stride,
                               loop_length_align_128, 1, 1, 8, 8)
        input_offset.set_as((coord_flag + 2) * self.all_element + offset)
        self.tik_inst.data_move(coord_ub, input_data[input_offset], 0, 1,
                                    loop_length_align_16, 1, 1)
        self.tik_inst.vexp(128, tmp1_ub, coord_ub, loop_length_align_128,
                               1, 1, 8, 8)
        self.tik_inst.vmul(128, tmp1_ub, tmp1_ub, anchor1_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmuls(128, tmp1_ub, tmp1_ub, stride,
                               loop_length_align_128, 1, 1, 8, 8)

        # xywh --> x1y1x2y2
        self.tik_inst.vmuls(128, tmp1_ub, tmp1_ub, tmp_scalar,
                                loop_length_align_128, 1, 1, 8, 8)
        self.tik_inst.vsub(128, coord_ub, tmp0_ub, tmp1_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(128, tmp0_ub, tmp0_ub, tmp1_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)

        # move to gm
        out_offset = coord_flag * self.all_element_align_16 * 16 + offset
        self.tik_inst.data_move(output_data[out_offset], coord_ub, 0, 1,
                                    loop_length_align_16, 1, 1)
        out_offset = out_offset + 2 * self.all_element_align_16 * 16
        self.tik_inst.data_move(output_data[out_offset], tmp0_ub, 0, 1,
                                    loop_length_align_16, 1, 1)

    def argmax_class_prob(self, input_offset, offset, zero_scalar, input_data,
                          obj_prob_ub, prob_ub, prob_max_ub, class_ub,
                          cur_class_ub, loop_length_align_16, 
                          loop_length_align_128, loop_length, prob_thres_ub,
                          zero_ub, loop_times, loop_idx):
        input_offset.set_as(self.all_element * 6 + offset)
        self.tik_inst.data_move(obj_prob_ub, input_data[input_offset],
                                    0, 1, loop_length_align_16, 1, 1)
        self.sigmoid(obj_prob_ub, loop_length)
        with self.tik_inst.for_range(0, loop_length_align_128) as i:
            cmpack = self.tik_inst.vcmp_ge(128, obj_prob_ub[128 * i],
                                               prob_thres_ub, 1, 1)
            self.tik_inst.vsel(128, 0, obj_prob_ub[128 * i], cmpack,
                 obj_prob_ub[128 * i], zero_ub, 1, 1, 1, 1, 8, 8, 8)
        with self.tik_inst.if_scope(loop_idx == (loop_times-1)):
            remain_element = self.all_element_align_16 * 16 - self.all_element
            with self.tik_inst.if_scope(remain_element > 0):
                with self.tik_inst.for_range(0, remain_element) as rem_idx:
                    obj_prob_ub[loop_length+rem_idx].set_as(zero_scalar)

        with self.tik_inst.for_range(0, self.num_class) as cls_idx:
            input_offset.set_as(7 + cls_idx)
            input_offset.set_as(input_offset * self.all_element + offset)
            with self.tik_inst.if_scope(cls_idx == 0):
                self.tik_inst.data_move(prob_ub, input_data[input_offset],
                                            0, 1, loop_length_align_16, 1, 1)
                self.sigmoid(prob_ub, loop_length)
                self.tik_inst.vmul(128, prob_max_ub, prob_ub, obj_prob_ub, 
                                       loop_length_align_128, 1, 1, 1, 8, 8, 8)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(prob_ub, input_data[input_offset],
                                            0, 1, loop_length_align_16, 1, 1)
                self.sigmoid(prob_ub, loop_length)
                self.tik_inst.vmul(128, prob_ub, prob_ub, obj_prob_ub, 
                                       loop_length_align_128, 1, 1, 1, 8, 8, 8)
                self.tik_inst.vadds(128, cur_class_ub, cur_class_ub, 1.0,
                                        loop_length_align_128, 1, 1, 8, 8)
                with self.tik_inst.for_range(0, loop_length_align_128) as idx:
                    cmpack = self.tik_inst.vcmp_ge(128, 
                             prob_max_ub[128 * idx], prob_ub[128 * idx], 1, 1)
                    self.tik_inst.vsel(128, 0, prob_max_ub[128 * idx], 
                         cmpack, prob_max_ub[128 * idx], prob_ub[128 * idx], 
                         1, 1, 1, 1, 8, 8, 8)
                    self.tik_inst.vsel(128, 0, class_ub[128 * idx], cmpack,
                         class_ub[128 * idx], cur_class_ub[128 * idx], 
                         1, 1, 1, 1, 8, 8, 8)

    def process_score(self, input_data, output_data, loop_times, loop_length,
                      input_offset, zero_scalar, loop_idx):
        obj_prob_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                      name="conf_prob_ub", scope=tik.scope_ubuf)
        zero_ub = self.tik_inst.Tensor(self.dtype, (128, ), name="zero_ub",
                                           scope=tik.scope_ubuf)

        prob_thres_ub = self.tik_inst.Tensor(self.dtype, (128, ),
                        name="prob_thres_ub", scope=tik.scope_ubuf)
        prob_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                       name="prob_ub", scope=tik.scope_ubuf)
        prob_max_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                      name="prob_max_ub", scope=tik.scope_ubuf)
        class_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                   name="class_ub", scope=tik.scope_ubuf)
        cur_class_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                       name="cur_class_ub", scope=tik.scope_ubuf)

        offset = loop_idx * self.slice_len
        loop_length.set_as(self.slice_len)
        with self.tik_inst.if_scope(loop_idx == (loop_times-1)):
            loop_length.set_as(self.all_element - offset)
        loop_length_align_128 = self.ceil_div_offline(loop_length, 128)
        loop_length_align_16 = self.ceil_div_offline(loop_length, 16)

        self.tik_inst.vector_dup(128, zero_ub, 0.0, 1, 1, 8)
        self.tik_inst.vector_dup(128, prob_thres_ub, self.score_threshold,
                                     1, 1, 8)
        self.tik_inst.vector_dup(128, class_ub, 0, loop_length_align_128, 1, 8)
        self.tik_inst.vector_dup(128, cur_class_ub, 0, loop_length_align_128,
                                     1, 8)
        self.tik_inst.vector_dup(128, prob_max_ub, 0, loop_length_align_128,
                                 1, 8)

        self.argmax_class_prob(input_offset, offset, zero_scalar, input_data,
             obj_prob_ub, prob_ub, prob_max_ub, class_ub, cur_class_ub,
             loop_length_align_16, loop_length_align_128, loop_length,
             prob_thres_ub, zero_ub, loop_times, loop_idx)

        #data move to output_gm 
        output_offset = 4 * self.all_element_align_16 * 16 + offset 
        self.tik_inst.data_move(output_data[output_offset], prob_max_ub, 0, 1,
                                    loop_length_align_16, 1, 1)
        output_offset = output_offset + self.all_element_align_16 * 16
        self.tik_inst.data_move(output_data[output_offset], class_ub, 0, 1,
                                    loop_length_align_16, 1, 1)

    def process_coord_score(self, pred_data, output_data, anchor_data):
        loop_times = self.ceil_div_offline(self.all_element, self.slice_len)
        loop_length = self.tik_inst.Scalar("int32")
        input_offset1 = self.tik_inst.Scalar("int32")
        zero_scalar = self.tik_inst.Scalar(self.dtype)
        zero_scalar.set_as(0.0)

        with self.tik_inst.for_range(
                  0, loop_times, block_num=loop_times) as loop_idx:
            offset = loop_idx * self.slice_len
            loop_length.set_as(self.slice_len)

            anchor0_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                         name="anchor0_ub", scope=tik.scope_ubuf)
            anchor1_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                         name="anchor1_ub", scope=tik.scope_ubuf)
            coord_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                       name="coord_ub", scope=tik.scope_ubuf)
            tmp0_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                      name="tmp0_ub", scope=tik.scope_ubuf)
            tmp1_ub = self.tik_inst.Tensor(self.dtype, (self.slice_len, ),
                      name="tmp1_ub", scope=tik.scope_ubuf)

            with self.tik_inst.if_scope(loop_idx == (loop_times-1)):
                loop_length.set_as(self.all_element - offset)

            loop_length_align_128 = self.ceil_div_offline(loop_length, 128)
            loop_length_align_16 = self.ceil_div_offline(loop_length, 16)

            #calculate x1, x2
            self.cal_coord(pred_data, output_data, anchor_data, coord_ub,
                           anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub, 0, offset,
                           self.width_stride, loop_length_align_128,
                           loop_length_align_16)
            self.cal_coord(pred_data, output_data, anchor_data, coord_ub,
                           anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub, 1, offset,
                           self.height_stride, loop_length_align_128,
                           loop_length_align_16)

            self.process_score(pred_data, output_data, loop_times, loop_length,
                               input_offset1, zero_scalar, loop_idx)

    def get_re_im_objprob(self, out_ub, index_ub, input_data):
        tmp_ub = self.tik_inst.Tensor(self.dtype, (128, ), name="tmp_ub",
                                          scope=tik.scope_ubuf)
        index = self.tik_inst.Scalar(dtype="int32", name="index")
        offset = self.tik_inst.Scalar(dtype="int32", name="offset")
        out_index = self.tik_inst.Scalar(dtype="int32", name="out_index")
        length = self.tik_inst.Scalar(dtype="int32")
        length.set_as(128)

        with self.tik_inst.for_range(0, self.max_output_size) as idx:
            index.set_as(index_ub[idx])
            offset.set_as(4 * self.all_element)
            offset.set_as(offset + index)
            out_index.set_as(4 + 9 * idx)
            self.tik_inst.data_move(tmp_ub, input_data[offset], 0, 1, 1, 0, 0)
            out_ub[out_index].set_as(tmp_ub[0])

            offset.set_as(offset + self.all_element)
            out_index.set_as(5 + 9 * idx)
            self.tik_inst.data_move(tmp_ub, input_data[offset], 0, 1, 1, 0, 0)
            out_ub[out_index].set_as(tmp_ub[0])

            offset.set_as(offset + self.all_element)
            out_index.set_as(6 + 9 * idx)
            self.tik_inst.data_move(tmp_ub, input_data[offset], 0, 1, 1, 0, 0)
            self.sigmoid(tmp_ub, length)
            out_ub[out_index].set_as(tmp_ub[0])

    def compile(self):
        self.process_coord_score(self.pred_gm, self.merge_gm, self.anchor_gm)

        #do nms
        ret, out_index_ub = self.multi_class_nms_obj.do_complex_yolo_nms()

        self.get_re_im_objprob(ret, out_index_ub, self.pred_gm)
        self.tik_inst.data_move(self.output_proposal_gm, ret, 0, 1,
             self.ceil_div_offline(self.max_output_size * 9, 16), 0, 0, 0)

        #build
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
             inputs=(self.pred_gm, self.anchor_gm, self.merge_gm),
             outputs=(self.output_proposal_gm, ))

        return self.tik_inst


def param_check(pred_shape, anchor_shape, merge_shape, width_stride, 
                height_stride, num_class, score_threshold, iou_threshold, 
                topk, max_output_size, kernel_name):
    log.check_gt(num_class, 0, "num_class should be greater than 0")
    log.check_le(num_class, 100,
                 "num_class should be less than or equal to 100")
    log.check_eq(anchor_shape[0], 1, "anchor_shape[0] should be equal to 1")
    log.check_eq(anchor_shape[2], 1, "anchor_shape[2] should be equal to 1")
    log.check_eq(anchor_shape[1], 4, "anchor_shape[1] should be equal to 4")
    log.check_eq(pred_shape[0], 1, "pred_shape[0] should be equal to 1")
    log.check_eq(pred_shape[2], 1, "pred_shape[2] should be equal to 1")
    log.check_eq(pred_shape[3], anchor_shape[3],
                "pred_shape[3] should be equal to anchor_shape[3]")
    merge_size = (pred_shape[3] + 15) // 16 * 16 * 6
    log.check_eq(merge_shape[0], merge_size,
    "merge_shape[0] should be euqal to (pred_shape[3] + 15) // 16 * 16 * 6")
    log.check_gt(score_threshold, 0,
                "score_threshold should be greater than 0")
    log.check_lt(score_threshold, 1, "score_threshold should be less than 1")
    log.check_gt(iou_threshold, 0, "iou_threshold should be greater than 0")
    log.check_lt(iou_threshold, 1, "iou_threshold should be less than 1")
    log.check_gt(max_output_size, 0,
                "max_output_size should be greater than 0")
    log.check_le(max_output_size, 256,
                "max_output_size should be less than or equal to 256")
    log.check_gt(topk, 0, "topk should be greater than 0")
    log.check_le(topk, 512, "topk should be less than or equal to 512")
    log.check_gt(width_stride, 0, "width_stride should be greater than 0")
    log.check_gt(height_stride, 0, "height_stride should be greater than 0")

    log.check_le(len(kernel_name), 200,
        "the length of kernel_name should be less than or equal to 200")


def complex_yolo_postprocess(pred, anchors, merge_buffer, out_rois,
                             width_stride, height_stride, score_threshold,
                             iou_threshold, topk, max_output_size,
                             kernel_name="complex_yolo_postprocess"):
    """
    the compile function of complex yolo postprocess
    """
    pred_shape = pred.get('shape')
    anchor_shape = anchors.get('shape')
    merge_shape = merge_buffer.get('shape')
    num_class = pred_shape[1] - 7 # 4:bbox, 2: re + rm 1:conf

    param_check(pred_shape, anchor_shape, merge_shape, width_stride, 
                height_stride, num_class, score_threshold, iou_threshold,
                topk, max_output_size, kernel_name)

    complex_yolo_obj = ComplexYoloPostprocess(pred_shape, anchor_shape,
                       merge_shape, width_stride, height_stride, num_class,
                       score_threshold, iou_threshold, topk, max_output_size,
                       kernel_name)
    tik_inst = complex_yolo_obj.compile()

    return tik_inst