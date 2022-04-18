#!/usr/bin/env python
# coding: utf-8
from te import tik
import numpy as np 
import multi_class_nms
from util import OpLog as log

#获得总规格尺寸
def get_shape_size(shape):
    return shape[0] * shape[1] * shape[2] * shape[3]


class EfficientdetPostprocess():
    """
    EfficientdetNms class
    """
    def __init__(self, box_shape, score_shape, anchor_shape, merge_shape,    
                 width, height, num_class, score_threshold, iou_threshold,
                 topk, max_output_size, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile()) #通过传入tik.Dprofile实例，创建TIK DSL容器。
        self.dtype = "float16"
        self.num_class = num_class
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.down_filter = 0.18 
        self.width = width
        self.height = height
        self.kernel_name = kernel_name

        self.max_output_size = max_output_size
        self.slice_len = 1024 * 4       #切片长度
        self.topk = topk
        self.box_shape = box_shape
        self.score_shape = score_shape
        self.anchor_shape = anchor_shape
        self.merge_shape = merge_shape  #合并后的形状
        self.all_element = box_shape[3] 
        self.all_element_align_16 = self.ceil_div_offline(self.all_element, 16) #元素对齐
        self.cls_out_num = 32       #输出类别数量
        #alloc global buffer
        self.alloc_gm() #分配全局缓冲区
        
        #multiclassnms文件的输出
        self.multi_class_nms_obj = multi_class_nms.MultiClassNms(
             self.tik_instance, self.merge_gm, self.dtype,
             self.all_element_align_16 * 16, self.num_class, self.width,
             self.height, self.max_output_size, self.iou_threshold,
             self.topk, self.cls_out_num, self.down_filter)

    #计算大于或等于value/factor的最小整数值
    def ceil_div_offline(self, value, factor):
        result = (value + (factor - 1)) // factor   #取整除 - 返回商的整数部分（向下取整）eg
        return result

    def gen_ub(self, data_type, length, ub_name):
        return self.tik_instance.Tensor(data_type, (length, ), name=ub_name,
                                        scope=tik.scope_ubuf)
    #分配全局缓冲区                  
    def alloc_gm(self):
        #input tensor
        self.box_gm = self.tik_instance.Tensor(self.dtype, 
                      (get_shape_size(self.box_shape) + 16, ),
                      name="box_gm", scope=tik.scope_gm)
        self.score_gm = self.tik_instance.Tensor(self.dtype, 
                        (get_shape_size(self.score_shape) + 16, ),
                        name="score_gm", scope=tik.scope_gm)
        self.anchor_gm = self.tik_instance.Tensor(self.dtype,
                         (get_shape_size(self.anchor_shape) + 16, ),
                         name="anchor_gm", scope=tik.scope_gm)
        self.merge_gm = self.tik_instance.Tensor(self.dtype, 
                        (self.merge_shape[0] + 16, ), name="merge_gm",
                        scope=tik.scope_gm)
        #output tensor
        self.output_gm = self.tik_instance.Tensor(self.dtype, 
                         (self.max_output_size * 8, ), 
                         name="output_gm", scope=tik.scope_gm)
    #坐标裁切
    def coord_clip(self, coord_ub, coord_min, coord_max,
                   loop_length_align_128):
        with self.tik_instance.for_range(0, loop_length_align_128) as idx:
            cmpack = self.tik_instance.vcmp_ge(128, coord_ub[idx * 128], 
                     coord_min, 1, 1)
            self.tik_instance.vsel(128, 0, coord_ub[idx * 128], cmpack,
                                   coord_ub[idx * 128], coord_min,
                                   1, 1, 1, 1, 8, 8, 8)
            #比特位
            cmpack1 = self.tik_instance.vcmp_ge(128, coord_max,         #coord_max 大於等於coord_ub
                                                coord_ub[idx * 128], 1, 1)
            #选择出
            self.tik_instance.vsel(128, 0, coord_ub[idx * 128], cmpack1,
                                   coord_ub[idx * 128], coord_max,
                                   1, 1, 1, 1, 8, 8, 8)
    #计算坐标，将网络输出返回到原图坐标
    def cal_coord(self, input_data, output_data, anchor_data, coord_ub,
                  anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub, coord_min,
                  coord_max, coord_flag, gm_flag, offset,
                  loop_length_align_128, loop_length_align_16):
        """
        flag: if 1 process w, x; if 0 process h, y 
        """
        tmp_scalar = self.tik_instance.Scalar(self.dtype)
        tmp_scalar.set_as(0.5)
        input_offset = self.tik_instance.Scalar("int32")            #补偿值
        input_offset.set_as((coord_flag + 2) * self.all_element + offset)
        #data_move:将读取的数据经处理后写入ub缓存
        self.tik_instance.data_move(anchor0_ub, anchor_data[(coord_flag) *
                                    self.all_element + offset], 0, 1,
                                    loop_length_align_16, 1, 1)
        self.tik_instance.data_move(anchor1_ub, anchor_data[input_offset],
                                    0, 1, loop_length_align_16, 1, 1)
        self.tik_instance.data_move(coord_ub, input_data[input_offset],
                                    0, 1, loop_length_align_16, 1, 1)
        #计算过程
        self.tik_instance.vsub(128, tmp0_ub, anchor1_ub, anchor0_ub,    #anchor1ub-anchor0ub = tmp0
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vexp(128, tmp1_ub, coord_ub,              #e^coord = tmp1
                               loop_length_align_128, 1, 1, 8, 8)
        self.tik_instance.vmul(128, tmp1_ub, tmp1_ub, tmp0_ub,          #tmp1 * tmp0 = tmp1
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)

        input_offset.set_as((coord_flag) * self.all_element + offset)       #更新input_offset
        self.tik_instance.data_move(coord_ub, input_data[input_offset],     #将更新值写入ub
                                    0, 1, loop_length_align_16, 1, 1)
        self.tik_instance.vmul(128, coord_ub, coord_ub, tmp0_ub,            #coord_ub = coord_ub * tmp0_ub
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(128, anchor0_ub, anchor0_ub, anchor1_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(128, anchor0_ub, anchor0_ub, tmp_scalar,
                                loop_length_align_128, 1, 1, 8, 8)
        self.tik_instance.vadd(128, coord_ub, coord_ub, anchor0_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        # xywh --> x1y1x2y2
        self.tik_instance.vmuls(128, tmp1_ub, tmp1_ub, tmp_scalar,
                               loop_length_align_128, 1, 1, 8, 8)
        self.tik_instance.vsub(128, tmp0_ub, coord_ub, tmp1_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(128, coord_ub, coord_ub, tmp1_ub,
                               loop_length_align_128, 1, 1, 1, 8, 8, 8)
        #clip
        self.coord_clip(tmp0_ub, coord_min, coord_max, loop_length_align_128)
        self.coord_clip(coord_ub, coord_min, coord_max, loop_length_align_128)
        #move to gm  ---> x1, y1, x2, y2
        output_offset = gm_flag * self.all_element_align_16 * 16 + offset
        self.tik_instance.data_move(output_data[output_offset],
                                    tmp0_ub, 0, 1, loop_length_align_16, 1, 1)
        output_offset = output_offset + 2 * self.all_element_align_16 * 16
        self.tik_instance.data_move(output_data[output_offset], coord_ub,
                                    0, 1, loop_length_align_16, 1, 1)
    #排序，去除
    def score_filter_move_gm(self, output_data, prob_max_ub, prob_thres_ub,
                             class_ub, zero_ub, loop_length_align_128,
                             loop_idx, loop_times, loop_length_align_16,
                             loop_length, offset):
        zero = self.tik_instance.Scalar(self.dtype)
        zero.set_as(0.0)
        # 1.filter score with threshold
        with self.tik_instance.for_range(0, loop_length_align_128) as i:
            cmpack = self.tik_instance.vcmp_ge(128, prob_max_ub[128 * i],
                                               prob_thres_ub, 1, 1)
            self.tik_instance.vsel(128, 0, prob_max_ub[128 * i], cmpack,
                                   prob_max_ub[128 * i], zero_ub,
                                   1, 1, 1, 1, 8, 8, 8)
        with self.tik_instance.if_scope(loop_idx == (loop_times-1)):
            remain_element = self.all_element_align_16 * 16 - self.all_element
            with self.tik_instance.if_scope(remain_element > 0):
                with self.tik_instance.for_range(0, remain_element) as rem_idx:
                    prob_max_ub[loop_length + rem_idx].set_as(zero)

        # 2.data move to output_gm 
        output_offset = 4 * self.all_element_align_16 * 16 + offset 
        self.tik_instance.data_move(output_data[output_offset], prob_max_ub,
                                    0, 1, loop_length_align_16, 1, 1)
        output_offset = output_offset + self.all_element_align_16 * 16
        self.tik_instance.data_move(output_data[output_offset], class_ub,
                                    0, 1, loop_length_align_16, 1, 1)
    #丢掉大于阈值值
    def process_score(self, input_data, output_data, loop_times, loop_length,
                      input_offset, loop_idx):
        prob_ub = self.gen_ub(self.dtype, self.slice_len, "prob_ub")
        prob_max_ub = self.gen_ub(self.dtype, self.slice_len, "prob_max_ub")
        prob_thres_ub = self.gen_ub(self.dtype, 128, "prob_thres_ub")
        zero_ub = self.gen_ub(self.dtype, 128, "zero_ub")
        class_ub = self.gen_ub(self.dtype, self.slice_len, "class_ub")
        cur_class_ub = self.gen_ub(self.dtype, self.slice_len, "cur_class_ub")

        offset = loop_idx * self.slice_len
        loop_length.set_as(self.slice_len)
        with self.tik_instance.if_scope(loop_idx == (loop_times-1)):
            loop_length.set_as(self.all_element - offset)
        loop_align_128 = self.ceil_div_offline(loop_length, 128)
        loop_align_16 = self.ceil_div_offline(loop_length, 16)

        self.tik_instance.vector_dup(128, prob_thres_ub, 
                                     self.score_threshold, 1, 1, 8)
        self.tik_instance.vector_dup(128, zero_ub, 0.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, class_ub, 0,
                                     loop_align_128, 1, 8)
        self.tik_instance.vector_dup(128, cur_class_ub, 0,
                                     loop_align_128, 1, 8)
        self.tik_instance.vector_dup(128, prob_max_ub, 0,
                                     loop_align_128, 1, 8)

        with self.tik_instance.for_range(0, self.num_class) as cls_idx:
            input_offset.set_as(cls_idx * self.all_element + offset)
            with self.tik_instance.if_scope(cls_idx == 0):
                self.tik_instance.data_move(prob_max_ub, 
                                            input_data[input_offset], 0, 1,
                                            loop_align_16, 1, 1)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(prob_ub, input_data[input_offset],
                                            0, 1, loop_align_16, 1, 1)
                self.tik_instance.vadds(128, cur_class_ub, cur_class_ub, 1.0,
                                        loop_align_128, 1, 1, 8, 8)

                with self.tik_instance.for_range(0, loop_align_128) as idx:
                    cmpack = self.tik_instance.vcmp_ge(128, 
                             prob_max_ub[128 * idx], prob_ub[128 * idx], 1, 1)
                    self.tik_instance.vsel(128, 0, prob_max_ub[128 * idx],
                                           cmpack, prob_max_ub[128 * idx],
                                           prob_ub[128 * idx],
                                           1, 1, 1, 1, 8, 8, 8)
                    self.tik_instance.vsel(128, 0, class_ub[128 * idx], cmpack,
                                           class_ub[128 * idx],
                                           cur_class_ub[128 * idx],
                                           1, 1, 1, 1, 8, 8, 8)

        self.score_filter_move_gm(output_data, prob_max_ub, prob_thres_ub,
                                  class_ub, zero_ub, loop_align_128,
                                  loop_idx, loop_times, loop_align_16,
                                  loop_length, offset)
    #调用前面模块，坐标处理，处理坐标
    def process_coord_score(self, box_data, score_data, output_data,
                            anchor_data):
        lp_cnt = self.ceil_div_offline(self.all_element, self.slice_len)
        loop_length = self.tik_instance.Scalar("int32")
        input_offset1 = self.tik_instance.Scalar("int32")


        with self.tik_instance.for_range(0, lp_cnt, block_num=lp_cnt) as idx:
            offset = idx * self.slice_len
            loop_length.set_as(self.slice_len)

            anchor0_ub = self.gen_ub(self.dtype, self.slice_len, "anchor0_ub")
            anchor1_ub = self.gen_ub(self.dtype, self.slice_len, "anchor1_ub")
            coord_ub = self.gen_ub(self.dtype, self.slice_len, "coord_ub")
            tmp0_ub = self.gen_ub(self.dtype, self.slice_len, "tmp0_ub")
            tmp1_ub = self.gen_ub(self.dtype, self.slice_len, "tmp1_ub")
            min_ub = self.gen_ub(self.dtype, 128, "min_ub")
            max_x_ub = self.gen_ub(self.dtype, 128, "max_x_ub")
            max_y_ub = self.gen_ub(self.dtype, 128, "max_y_ub")

            self.tik_instance.vector_dup(128, min_ub, 0, 1, 1, 8, 0)
            self.tik_instance.vector_dup(128, max_x_ub, 
                                         self.width - 1, 1, 1, 8, 0)
            self.tik_instance.vector_dup(128, max_y_ub,
                                         self.height - 1, 1, 1, 8, 0)

            with self.tik_instance.if_scope(idx == (lp_cnt-1)):
                loop_length.set_as(self.all_element - offset)

            loop_length_align_128 = self.ceil_div_offline(loop_length, 128)
            loop_length_align_16 = self.ceil_div_offline(loop_length, 16)

            #x1, x2
            self.cal_coord(box_data, output_data, anchor_data, coord_ub, 
                           anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub,
                           min_ub, max_x_ub, 1, 0, offset,
                           loop_length_align_128, loop_length_align_16)

            #y1, y2
            self.cal_coord(box_data, output_data, anchor_data, coord_ub,
                           anchor0_ub, anchor1_ub, tmp0_ub, tmp1_ub, min_ub,
                           max_y_ub, 0, 1, offset, loop_length_align_128,
                           loop_length_align_16)

            self.process_score(score_data, output_data, lp_cnt,
                               loop_length, input_offset1, idx)
    #处理后的boxx调用nms
    def compile(self):
        self.process_coord_score(self.box_gm, self.score_gm, 
                                 self.merge_gm, self.anchor_gm)
        #do nms
        ret = self.multi_class_nms_obj.do_efficientdet_nms()
        self.tik_instance.data_move(self.output_gm, ret, 0, 1,
                                    self.max_output_size * 8 // 16, 0, 0, 0)        #gm存储总存储数据
        #build
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=(
                                   self.box_gm, self.score_gm, self.anchor_gm,
                                   self.merge_gm), outputs=(self.output_gm,))
        return self.tik_instance


def param_check(box_shape, score_shape, anchor_shape, merge_shape, width,
                height, num_class, score_threshold, iou_threshold, topk,
                max_output_size):
    log.check_gt(num_class, 0, "num_class should be greater than 0")
    log.check_le(num_class, 100,
                 "num_class should be less than or equal to 100")
    log.check_eq(box_shape[3], score_shape[3],
                 "box_shape[3] should be equal to score_shape[3]")
    log.check_eq(box_shape[2], anchor_shape[2],
                 "box_shape[2] should be equal to anchor_shape[2]")
    log.check_eq(box_shape[3], anchor_shape[3],
                 "box_shape[3] should be equal to anchor_shape[3]")
    log.check_eq(box_shape[0], 1, "box_shape[0] should be equal to 1")
    log.check_eq(anchor_shape[0], 1, "anchor_shape[0] should be equal to 1")
    log.check_eq(anchor_shape[2], 1, "anchor_shape[1] should be equal to 1")
    log.check_eq(score_shape[0], 1, "score_shape[0] should be equal to 1")
    log.check_eq(score_shape[2], 1, "score_shape[1] should be equal to 1")
    log.check_eq(box_shape[1], 4, "box_shape[1] should be equal to 4")
    log.check_eq(anchor_shape[1], 4, "anchor_shape[1] should be equal to 4")
    log.check_gt(width, 0, "width should be greater than 0")
    log.check_gt(height, 0, "height should be greater than 0")
    log.check_ge(score_threshold, 0,
                 "score_threshold should be greater than or equal to 0")
    log.check_le(score_threshold, 1,
                 "score_threshold should be less than or equal to 1")
    log.check_ge(iou_threshold, 0,
                 "iou_threshold should be greater than or equal to 0")
    log.check_le(iou_threshold, 1,
                 "iou_threshold should be less than or equal to 1")
    log.check_gt(topk, 0, "topk should be greater than 0")
    log.check_le(topk, 1024, "topk should be less than or equal to 1024")
    log.check_gt(max_output_size, 0,
                 "max_output_size should be greater than 0")
    log.check_le(max_output_size, 256,
                 "max_output_size should be less than or equal to 256")
    merge_len = (score_shape[3] + 15) // 16 * 16 * 6 
    log.check_ge(merge_shape[0], merge_len,
        "merge_shape[0] should be equal (score_shape[3]+15)//16*16*6")


def efficientdet_postprocess(boxes, scores, anchors, merge_buffer, out_rois,
                     width, height, score_threshold, iou_threshold, topk,
                     max_output_size, kernel_name="efficientdet_nms"):
    """
    the compile function of efficientdet nms
    """
    box_shape = boxes.get('shape')
    score_shape = scores.get('shape')
    anchor_shape = anchors.get('shape')
    merge_shape = merge_buffer.get('shape')
    num_class = score_shape[1]
    param_check(box_shape, score_shape, anchor_shape, merge_shape, width,
                height, num_class, score_threshold, iou_threshold, topk,
                max_output_size)

    efficientdet_obj = EfficientdetPostprocess(box_shape, score_shape,
                       anchor_shape, merge_shape, width, height, num_class,
                       score_threshold, iou_threshold, topk, max_output_size,
                       kernel_name)
    tik_instance = efficientdet_obj.compile()
    return tik_instance

