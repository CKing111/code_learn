


class EfficientdetPostprocess():
    """

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
        self.tik_instance.data_move(output_data[output_offset], class_ub,       #
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
        #數據填充
        self.tik_instance.vector_dup(128, prob_thres_ub,        #將得分閾值輸入到prob_thres_ub
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
                                            input_data[input_offset] , 0, 1,
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


    def compile(self):
        self._gm = self.tik_instance.Tensor(self.dtype, 
                        (get_shape_size(self.score_shape) + 16, ),
                        name="score_gm", scope=tik.scope_gm)        
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


#获得总规格尺寸
def get_shape_size(shape):
    return shape[0] * shape[1] * shape[2] * shape[3]


def _postprocess(boxes, scores,centernes, anchors, merge_buffer, out_rois,
                     width, height, score_threshold, iou_threshold, topk,
                     max_output_size, kernel_name="efficientdet_nms"):
    """
    the compile function of efficientdet nms
    """
    box_shape = boxes.get('shape')
    score_shape = scores.get('shape')
    centernes_shape = centernes.get('shape')
    anchor_shape = anchors.get('shape')
    merge_shape = merge_buffer.get('shape')
    num_class = score_shape[1]
    param_check(box_shape, score_shape, anchor_shape, merge_shape, width,
                height, num_class, score_threshold, iou_threshold, topk,
                max_output_size)
nter
        self.centernes_gm = self.tik_instance.Tensor(self.dtype, 
                        (get_shape_size(self.score_shape) + 16, ),
                        name="score_gm", scope=tik.scope_gm)      

    efficientdet_obj = EfficientdetPostprocess(box_shape, score_shape,
                       anchor_shape, merge_shape, width, height, num_class,
                       score_threshold, iou_threshold, topk, max_output_size,
                       kernel_name)
    tik_instance = efficientdet_obj.compile()
    return tik_instance


        