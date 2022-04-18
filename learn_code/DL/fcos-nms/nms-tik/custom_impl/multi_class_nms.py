from te import tik
import sort_proposal
import nms
from nms import ceil_div as ceil_div_offline


class MultiClassNms():
    def __init__(self, tik_instance, input_gm, dtype, all_element, num_class,
                 width, height, max_output_size, iou_threshold, topk,
                 cls_out_num, down_filter):
        self.tik_instance = tik_instance
        self.dtype = dtype
        self.input_gm = input_gm
        self.all_element = all_element
        self.num_class = num_class
        self.topk = topk
        self.iou_threshold = iou_threshold
        self.down_filter = down_filter
        self.cls_out_num = cls_out_num # each class max output 16 proposal

        self.burst_input_proposal_num = 64
        self.out_num = max_output_size

        self.ceil_div_offline = ceil_div_offline

        self.topk_align_16 = self.ceil_div_offline(self.topk, 16)
        self.topk_align_128 = self.ceil_div_offline(self.topk, 128)
        self.out_num_align_16 = self.ceil_div_offline(self.out_num, 16)
        self.out_num_align_128 = self.ceil_div_offline(self.out_num, 128)

        self.shape_align_16 = self.ceil_div_offline(self.all_element, 16)
        self.shape_alilgn_128 = self.ceil_div_offline(self.all_element, 128)
        self.slice_len = 1024
        self.width = width
        self.height = height

        self.sort_proposal_obj = sort_proposal.SortProposal(
                                 self.tik_instance, self.input_gm, self.dtype,
                                 self.all_element, self.topk)

    def gen_ub(self, data_type, length, ub_name):
        return self.tik_instance.Tensor(data_type, (length, ), name=ub_name,
                                        scope=tik.scope_ubuf)

    def get_class_id(self, input_data, index_ub, class_ub):
        index = self.tik_instance.Scalar(dtype="int32", name="index")
        offset = self.tik_instance.Scalar(dtype="int32", name="offset")
        tmp_ub = self.gen_ub(self.dtype, 16, "tmp_ub")
        class_int_ub = self.gen_ub("int32", self.topk_align_128 * 128,
                                "class_int_ub")

        with self.tik_instance.for_range(0, self.topk) as idx:
            index.set_as(index_ub[idx])
            offset.set_as(index)
            offset.set_as(offset + self.shape_align_16 * 16 * 5)
            self.tik_instance.data_move(tmp_ub, input_data[offset],
                                        0, 1, 1, 0, 0)
            class_ub[idx].set_as(tmp_ub[0])

        # half ---> int
        self.tik_instance.vconv(64, 'round', class_int_ub, class_ub,
                                self.topk_align_128, 1, 1, 8, 4)
        self.tik_instance.vconv(64, 'round', 
                                class_int_ub[self.topk_align_128 * 64],
                                class_ub[self.topk_align_128 * 64],
                                self.topk_align_128, 1, 1, 8, 4)
        return class_int_ub

    def get_proposal_by_index(self, input_data, index_ub, index_cnt,
                              proposal_ub, x1_ub, y1_ub, x2_ub, y2_ub,
                              prob_ub):
        index = self.tik_instance.Scalar(dtype="int32", name="index")
        offset = self.tik_instance.Scalar(dtype="int32", name="offset")
        tmp_ub = self.gen_ub(self.dtype, 16, "tmp_ub")
        with self.tik_instance.for_range(0, index_cnt) as idx:
            index.set_as(index_ub[idx])
            offset.set_as(index)

            self.tik_instance.data_move(tmp_ub, input_data[offset], 
                                        0, 1, 1, 0, 0)
            x1_ub[idx].set_as(tmp_ub[0])

            offset.set_as(offset + self.shape_align_16 * 16)
            self.tik_instance.data_move(tmp_ub, input_data[offset],
                                        0, 1, 1, 0, 0)
            y1_ub[idx].set_as(tmp_ub[0]) 

            offset.set_as(offset + self.shape_align_16 * 16)
            self.tik_instance.data_move(tmp_ub, input_data[offset],
                                        0, 1, 1, 0, 0)
            x2_ub[idx].set_as(tmp_ub[0])

            offset.set_as(offset + self.shape_align_16 * 16)
            self.tik_instance.data_move(tmp_ub, input_data[offset],
                                        0, 1, 1, 0, 0)
            y2_ub[idx].set_as(tmp_ub[0])

            offset.set_as(offset + self.shape_align_16 * 16)
            self.tik_instance.data_move(tmp_ub, input_data[offset],
                                        0, 1, 1, 0, 0)
            prob_ub[idx].set_as(tmp_ub[0])
        
        # concat proposal
        self.tik_instance.vconcat(proposal_ub, x1_ub, 
                                  self.ceil_div_offline(index_cnt, 16), 0)
        self.tik_instance.vconcat(proposal_ub, y1_ub, 
                                  self.ceil_div_offline(index_cnt, 16), 1)
        self.tik_instance.vconcat(proposal_ub, x2_ub,
                                  self.ceil_div_offline(index_cnt, 16), 2)
        self.tik_instance.vconcat(proposal_ub, y2_ub,
                                  self.ceil_div_offline(index_cnt, 16), 3)
        self.tik_instance.vconcat(proposal_ub, prob_ub,
                                  self.ceil_div_offline(index_cnt, 16), 4)

    def get_coord_by_index(self, coord_data, index_ub, index_cnt, x1_ub,
                           y1_ub, x2_ub, y2_ub):
        index = self.tik_instance.Scalar(dtype="int32", name="index")
        offset = self.tik_instance.Scalar(dtype="int32", name="offset")
        tmp_ub = self.gen_ub(self.dtype, 16, "tmp_ub")
        with self.tik_instance.for_range(0, index_cnt) as idx:
            index.set_as(index_ub[idx])
            offset.set_as(index)
            # x1
            self.tik_instance.data_move(tmp_ub, coord_data[offset],
                                        0, 1, 1, 0, 0)
            x1_ub[idx].set_as(tmp_ub[0])
            # y1
            offset.set_as(offset + self.shape_align_16 * 16)
            self.tik_instance.data_move(tmp_ub, coord_data[offset],
                                        0, 1, 1, 0, 0)
            y1_ub[idx].set_as(tmp_ub[0]) 
            # x2
            offset.set_as(offset + self.shape_align_16 * 16)
            self.tik_instance.data_move(tmp_ub, coord_data[offset],
                                        0, 1, 1, 0, 0)
            x2_ub[idx].set_as(tmp_ub[0])
            # y2
            offset.set_as(offset + self.shape_align_16 * 16)
            self.tik_instance.data_move(tmp_ub, coord_data[offset],
                                        0, 1, 1, 0, 0)
            y2_ub[idx].set_as(tmp_ub[0])

    def gen_topk_idx(self):
        tmp = self.tik_instance.Scalar(dtype="int32", name="tmp")
        tmp.set_as(0)

        idx_int_ub = self.gen_ub("int32", self.topk_align_128 * 128,
                                 "idx_int_ub")
        idx_fp_ub = self.gen_ub("float16", self.topk_align_128 * 128,
                                "idx_fp_ub")

        self.tik_instance.vector_dup(64, idx_int_ub, 0, 
                                     self.topk_align_128 * 2, 1, 8, 0)
        with self.tik_instance.for_range(0, self.topk) as idx:
            idx_int_ub[idx].set_as(tmp)
            tmp.set_as(tmp + 1)

        self.tik_instance.vconv(64, '', idx_fp_ub, idx_int_ub, 
                                self.topk_align_128, 1, 1, 4, 8, 1.0)
        self.tik_instance.vconv(64, '', idx_fp_ub[self.topk_align_128 * 64],
                                idx_int_ub[self.topk_align_128 * 64],
                                self.topk_align_128, 1, 1, 4, 8, 1.0)
        return idx_fp_ub

    def restore_coord(self, coord_ub, img_size, length_align_128):
        zero_ub = self.gen_ub(self.dtype, length_align_128 * 128,
                              "zero_ub")
        max_ub = self.gen_ub(self.dtype, length_align_128 * 128,
                              "max_ub")

        self.tik_instance.vector_dup(128, zero_ub, 0.0, 1, 1, 8)
        self.tik_instance.vmuls(128, coord_ub, coord_ub, img_size, 
                                length_align_128, 1, 1, 8, 8)
        self.tik_instance.vmax(128, coord_ub, coord_ub, zero_ub,
                               length_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vector_dup(128, max_ub, img_size, 1, 1, 8)
        self.tik_instance.vmin(128, coord_ub, coord_ub, max_ub,
                               length_align_128, 1, 1, 1, 8, 8, 8)

    def sort_cnt_great_one(self, tmp_index_ub, class_cnt, proposal_ub, x1_ub,
                           y1_ub, x2_ub, y2_ub, prob_ub, ret, ret1,
                           per_class_proposal_ub, tmp1_index_ub, proposal_cnt):
        self.get_proposal_by_index(self.input_gm, tmp_index_ub,
                                   class_cnt, proposal_ub, x1_ub,
                                   y1_ub, x2_ub, y2_ub, prob_ub)
        #nms 
        one_core_nms = nms.OneCoreNMS(self.tik_instance, (class_cnt,
                                      self.cls_out_num, 
                                      self.burst_input_proposal_num,
                                      self.down_filter))
        super_vector = one_core_nms.nms_single_core(proposal_ub, 
                           self.iou_threshold)
        proposal_cnt.set_as(0)
        with self.tik_instance.for_range(0, class_cnt) as i:
            with self.tik_instance.if_scope(
                      tik.all(super_vector[i] == 0,
                      proposal_cnt < self.cls_out_num)):
                per_class_proposal_ub[proposal_cnt * 8 + 4].set_as(
                                      prob_ub[i])
                per_class_proposal_ub[proposal_cnt * 8].set_as(
                                      tmp1_index_ub[i])
                proposal_cnt.set_as(proposal_cnt + 1)
        # sort with output
        self.tik_instance.vmrgsort4(ret, [ret1, per_class_proposal_ub,
                                    per_class_proposal_ub, 
                                    per_class_proposal_ub], 
                                    [self.out_num, proposal_cnt,
                                    0, 0], False, 3, 1, None)

        self.tik_instance.data_move(ret1, ret, 0, 1,
                                    self.out_num_align_16 * 8, 0, 0)

    def sort_cnt_equal_one(self, tmp_index_ub, class_cnt, proposal_ub, x1_ub,
                           y1_ub, x2_ub, y2_ub, prob_ub, ret, ret1,
                           per_class_proposal_ub, tmp1_index_ub):
        self.get_proposal_by_index(self.input_gm, tmp_index_ub,
                                   class_cnt, proposal_ub, x1_ub,
                                   y1_ub, x2_ub, y2_ub, prob_ub)
        #copy data
        per_class_proposal_ub[4].set_as(prob_ub[0])
        per_class_proposal_ub[0].set_as(tmp1_index_ub[0])

        # sort with output
        self.tik_instance.vmrgsort4(ret, [ret1, per_class_proposal_ub,
                                    per_class_proposal_ub, 
                                    per_class_proposal_ub],
                                    [self.out_num, 1, 0, 0],
                                    False, 3, 1, None)

        self.tik_instance.data_move(ret1, ret, 0, 1, 
                                    self.out_num_align_16 * 8, 0, 0)

    def nms_sort(self, x1_ub, y1_ub, x2_ub, y2_ub, prob_ub, 
                 per_class_proposal_ub, proposal_ub, index_ub, class_int_ub,
                 ret, idx_int_ub, idx_hf_ub, topk_idx_ub, ret1):
        class_id = self.tik_instance.Scalar(dtype="int32", name="class_id")
        class_cnt = self.tik_instance.Scalar(dtype="uint16", name="class_cnt")
        proposal_cnt = self.tik_instance.Scalar("uint16")
        with self.tik_instance.for_range(0, self.num_class) as cls_idx:
            class_cnt.set_as(0)
            self.tik_instance.vector_dup(128, per_class_proposal_ub, 0,
                 self.ceil_div_offline(self.cls_out_num * 8, 128), 1, 8)

            with self.tik_instance.for_range(0, self.topk) as idx:
                class_id.set_as(class_int_ub[idx])
                with self.tik_instance.if_scope(class_id == cls_idx):
                    idx_int_ub[class_cnt].set_as(index_ub[idx])
                    idx_hf_ub[class_cnt].set_as(topk_idx_ub[idx])
                    class_cnt.set_as(class_cnt + 1)

            with self.tik_instance.if_scope(class_cnt > 1):
                self.sort_cnt_great_one(idx_int_ub, class_cnt, proposal_ub,
                                        x1_ub, y1_ub, x2_ub, y2_ub, prob_ub,
                                        ret, ret1, per_class_proposal_ub,
                                        idx_hf_ub, proposal_cnt)

            with self.tik_instance.if_scope(class_cnt == 1):
                self.sort_cnt_equal_one(idx_int_ub, class_cnt, proposal_ub,
                                        x1_ub, y1_ub, x2_ub, y2_ub, prob_ub,
                                        ret, ret1, per_class_proposal_ub,
                                        idx_hf_ub)

    def do_each_class_nms(self, x1_ub, y1_ub, x2_ub, y2_ub, prob_ub, 
                          per_class_proposal_ub, proposal_ub, index_ub,
                          class_int_ub, ret, idx_int_ub, idx_hf_ub,
                          topk_idx_ub):

        ret1 = self.gen_ub(self.dtype, self.out_num_align_16 * 16 * 8, "ret1")

        self.tik_instance.vector_dup(128, ret1, 0.0,
                                     self.out_num_align_16, 1, 8)

        # nms for each class and sort by prob
        self.nms_sort(x1_ub, y1_ub, x2_ub, y2_ub, prob_ub, 
             per_class_proposal_ub, proposal_ub, index_ub, class_int_ub,
             ret, idx_int_ub, idx_hf_ub, topk_idx_ub, ret1)
        
        self.tik_instance.vextract(idx_hf_ub, ret, 
                                   self.ceil_div_offline(self.out_num, 16), 0)
        self.tik_instance.vconv(64, 'round', idx_int_ub, idx_hf_ub,
                                self.ceil_div_offline(self.out_num, 64),
                                1, 1, 8, 4)

    def do_nms(self):
        length = self.topk_align_128 * 128
        x1_ub = self.gen_ub(self.dtype, length, "x1_ub")
        y1_ub = self.gen_ub(self.dtype, length, "y1_ub")
        x2_ub = self.gen_ub(self.dtype, length, "x2_ub")
        y2_ub = self.gen_ub(self.dtype, length, "y2_ub")
        class_ub = self.gen_ub(self.dtype, length, "class_ub")
        prob_ub = self.gen_ub(self.dtype, length, "prob_ub")
        idx_int_ub = self.gen_ub("int32", length, "idx_int_ub")
        idx_hf_ub = self.gen_ub("float16", length, "idx_hf_ub")
        proposal_ub = self.tik_instance.Tensor(self.dtype,
                          (self.topk_align_16 * 16, 8),
                          name="proposal_ub", scope=tik.scope_ubuf)
        ret = self.gen_ub(self.dtype,
              self.ceil_div_offline(self.out_num * 8 * 2, 128) * 128, "ret")
        per_class_proposal_ub = self.gen_ub(self.dtype,
            self.ceil_div_offline(self.cls_out_num * 8, 128) * 128,
            "per_class_proposal_ub")

        self.tik_instance.vector_dup(128, x1_ub, 0.0, 
                                     self.topk_align_128, 1, 8)
        self.tik_instance.vector_dup(128, y1_ub, 0.0,
                                     self.topk_align_128, 1, 8)
        self.tik_instance.vector_dup(128, x2_ub, 0.0,
                                     self.topk_align_128, 1, 8)
        self.tik_instance.vector_dup(128, y2_ub, 0.0,
                                     self.topk_align_128, 1, 8)
        self.tik_instance.vector_dup(128, prob_ub, 0.0,
                                     self.topk_align_128, 1, 8)
        self.tik_instance.vector_dup(64, idx_int_ub, 0, 
                                     self.topk_align_128 * 2, 1, 8)
        self.tik_instance.vector_dup(128, proposal_ub, 0.0,
                                     self.topk_align_16, 1, 8)
        self.tik_instance.vector_dup(128, ret, 0.0,
             self.ceil_div_offline(self.out_num * 8 * 2, 128), 1, 8)

        # sort the proposal by score
        index_ub = self.sort_proposal_obj.sort_topk()

        # get class by index and convert to int
        class_int_ub = self.get_class_id(self.input_gm, index_ub, class_ub)
        
        # gen index, use the index to record the index after nms
        topk_idx_ub = self.gen_topk_idx()

        # do each class nms
        self.do_each_class_nms(x1_ub, y1_ub, x2_ub, y2_ub, prob_ub, 
             per_class_proposal_ub, proposal_ub, index_ub, class_int_ub,
             ret, idx_int_ub, idx_hf_ub, topk_idx_ub)

        return ret, index_ub, idx_int_ub, class_ub, x1_ub, y1_ub, x2_ub, y2_ub

    def gen_coord_mask(self, ret):
        mask_ub = self.gen_ub(self.dtype, self.out_num_align_128 * 128,
                              "mask_ub")
        one_ub = self.gen_ub(self.dtype, 128, "one_ub")
        zero_ub = self.gen_ub(self.dtype, 128, "zero_ub")

        self.tik_instance.vector_dup(128, mask_ub, 0.0,
                                     self.out_num_align_128, 1, 8)
        self.tik_instance.vector_dup(128, one_ub, 1.0, 1, 1, 8)
        self.tik_instance.vector_dup(128, zero_ub, 0.0, 1, 1, 8)

        with self.tik_instance.for_range(0, self.out_num) as idx:
            mask_ub[idx].set_as(ret[8 * idx + 4])

        with self.tik_instance.for_range(0, self.out_num_align_128) as i:
            cmp_pack = self.tik_instance.vcmp_gt(128, mask_ub[128 * i],
                                                 zero_ub, 1, 1)
            self.tik_instance.vsel(128, 0, mask_ub[128 * i], cmp_pack,
                                   one_ub, zero_ub, 1, 1, 1, 1, 8, 8, 8)
        return mask_ub

    def do_yolact_nms(self):
        out_index_ub = self.gen_ub("int32", self.out_num_align_16 * 16,
                                   "out_index_ub")
        out_coord_ub = self.gen_ub(self.dtype,
                       self.ceil_div_offline(self.out_num * 4, 16) * 16,
                       "out_coord_ub")
        
        self.tik_instance.vector_dup(64, out_index_ub, 0, 
             self.ceil_div_offline(self.out_num, 16) // 4, 1, 8)
        self.tik_instance.vector_dup(128, out_coord_ub, 0.0,
             self.ceil_div_offline(self.out_num * 4, 16) // 8, 1, 8)
        cls_idx = self.tik_instance.Scalar(dtype="int32", name="class_id")

        ret, id1_ub, id2_ub, cls_ub, x1_ub, y1_ub, x2_ub, y2_ub = self.do_nms()
        mask_ub = self.gen_coord_mask(ret)

        with self.tik_instance.for_range(0, self.out_num) as out_index:
            cls_idx.set_as(id2_ub[out_index])
            out_index_ub[out_index].set_as(id1_ub[cls_idx])
            ret[8 * out_index + 5].set_as(cls_ub[cls_idx])

        self.get_coord_by_index(self.input_gm, out_index_ub, self.out_num,
                                x1_ub, y1_ub, x2_ub, y2_ub)
        self.tik_instance.vmul(128, x1_ub, x1_ub, mask_ub,
                               self.out_num_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(128, y1_ub, y1_ub, mask_ub,
                               self.out_num_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(128, x2_ub, x2_ub, mask_ub,
                               self.out_num_align_128, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(128, y2_ub, y2_ub, mask_ub,
                               self.out_num_align_128, 1, 1, 1, 8, 8, 8)

        # set coord_ub
        with self.tik_instance.for_range(0, self.out_num) as idx:
            out_coord_ub[idx].set_as(x1_ub[idx])
            out_coord_ub[idx + self.out_num].set_as(y1_ub[idx])
            out_coord_ub[idx + 2 * self.out_num].set_as(x2_ub[idx])
            out_coord_ub[idx + 3 * self.out_num].set_as(y2_ub[idx])

        # restore the coord to the origin image
        img_size = self.tik_instance.Scalar(self.dtype)
        img_size.set_as(self.width)
        length_align_128 = self.ceil_div_offline(self.out_num, 128)
        self.restore_coord(x1_ub, img_size, length_align_128)
        self.restore_coord(x2_ub, img_size, length_align_128)
        img_size.set_as(self.height)
        self.restore_coord(y1_ub, img_size, length_align_128)
        self.restore_coord(y2_ub, img_size, length_align_128)

        with self.tik_instance.for_range(0, self.out_num) as idx:
            ret[8 * idx].set_as(x1_ub[idx])
            ret[8 * idx + 1].set_as(y1_ub[idx])
            ret[8 * idx + 2].set_as(x2_ub[idx])
            ret[8 * idx + 3].set_as(y2_ub[idx])
        return ret, out_coord_ub, out_index_ub

    def do_complex_yolo_nms(self):
        out_index_ub = self.gen_ub("int32", self.out_num_align_16 * 16,
                                   "out_index_ub")
        self.tik_instance.vector_dup(64, out_index_ub, 0, 
             self.ceil_div_offline(self.out_num, 16) // 4, 1, 8)
        cls_idx = self.tik_instance.Scalar(dtype="int32", name="class_id")

        ret, id1_ub, id2_ub, cls_ub, x1_ub, y1_ub, x2_ub, y2_ub = self.do_nms()

        with self.tik_instance.for_range(0, self.out_num) as prob_idx:
            offset = self.out_num - prob_idx - 1
            ret[offset * 9 + 7].set_as(ret[offset * 8 + 4])

        with self.tik_instance.for_range(0, self.out_num) as out_index:
            cls_idx.set_as(id2_ub[out_index])
            out_index_ub[out_index].set_as(id1_ub[cls_idx])
            ret[9 * out_index + 8].set_as(cls_ub[cls_idx])

        self.get_coord_by_index(self.input_gm, out_index_ub, self.out_num,
                                x1_ub, y1_ub, x2_ub, y2_ub)

        with self.tik_instance.for_range(0, self.out_num) as idx:
            ret[9 * idx].set_as(x1_ub[idx])
            ret[9 * idx + 1].set_as(y1_ub[idx])
            ret[9 * idx + 2].set_as(x2_ub[idx])
            ret[9 * idx + 3].set_as(y2_ub[idx])
        return ret, out_index_ub

    def do_efficientdet_nms(self):
        out_index_ub = self.gen_ub("int32", self.out_num_align_16 * 16,
                                   "out_index_ub")
        self.tik_instance.vector_dup(64, out_index_ub, 0,
             self.ceil_div_offline(self.out_num, 16) // 4, 1, 8)
        cls_idx = self.tik_instance.Scalar(dtype="int32", name="class_id")
        ret, id1_ub, id2_ub, cls_ub, x1_ub, y1_ub, x2_ub, y2_ub = self.do_nms()

        with self.tik_instance.for_range(0, self.out_num) as out_index:
            cls_idx.set_as(id2_ub[out_index])
            out_index_ub[out_index].set_as(id1_ub[cls_idx])
            ret[8 * out_index + 5].set_as(cls_ub[cls_idx])

        self.get_coord_by_index(self.input_gm, out_index_ub, self.out_num,
                                x1_ub, y1_ub, x2_ub, y2_ub)

        with self.tik_instance.for_range(0, self.out_num) as idx:
            ret[8 * idx].set_as(x1_ub[idx])
            ret[8 * idx + 1].set_as(y1_ub[idx])
            ret[8 * idx + 2].set_as(x2_ub[idx])
            ret[8 * idx + 3].set_as(y2_ub[idx])
        return ret
