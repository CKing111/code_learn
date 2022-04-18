#!/usr/bin/env python
# coding: utf-8
from te import tik
import numpy as np 


class SortProposal():
    def __init__(self, tik_instance, input_gm, dtype, all_element, topk):
        self.tik_instance = tik_instance
        self.dtype = dtype
        self.input_gm = input_gm 

        self.topk = topk
        self.all_element = all_element
        self.topk_align_16 = self.ceil_div_offline(self.topk, 16)
        self.topk_align_128 = self.ceil_div_offline(self.topk, 128)
        self.shape_align_16 = self.ceil_div_offline(self.all_element, 16)
        self.shape_alilgn_128 = self.ceil_div_offline(self.all_element, 128)
        self.slice_len = 1024
        self.compare_len = self.topk
        if self.topk > self.slice_len:
            self.compare_len = self.slice_len
        self.output_index_ub = self.gen_ub("int32", self.topk_align_128 * 128,
                                           "index_int_ub")

    def ceil_div_offline(self, value, factor):
        result = (value + (factor - 1)) // factor
        return result

    def gen_ub(self, data_type, length, ub_name):
        return self.tik_instance.Tensor(data_type, (length, ), name=ub_name,
                                        scope=tik.scope_ubuf)

    def gen_index(self, index1_ub, length):
        tmp = self.tik_instance.Scalar(dtype="int32", name="tmp")
        tmp.set_as(0)

        idx_tmp_ub = self.gen_ub("int32", self.slice_len, "idx_tmp")

        self.tik_instance.vector_dup(64, idx_tmp_ub, tmp,
                                     self.slice_len // 64, 1, 8, 0)
        with self.tik_instance.for_range(0, length) as slice_idx:
            idx_tmp_ub[slice_idx].set_as(tmp)
            tmp.set_as(tmp + 1)

        self.tik_instance.vconv(64, '', index1_ub, idx_tmp_ub,
                                self.slice_len // 128, 1, 1, 4, 8, 1.0)
        self.tik_instance.vconv(64, '', index1_ub[self.slice_len // 2],
                                idx_tmp_ub[self.slice_len // 2], 
                                self.slice_len // 128, 1, 1, 4, 8, 1.0)

    def gen_proposal(self, input_data, proposal_ub, prob_max_ub, index0_ub,
                     index1_ub, loop_length, offset, loop_idx):
        loop_length_align_16 = self.ceil_div_offline(loop_length, 16)
        #move prob 
        input_offset = 4 * self.shape_align_16 * 16 + offset
        self.tik_instance.data_move(prob_max_ub, input_data[input_offset],
                                    0, 1, loop_length_align_16, 0, 0)
        
        self.tik_instance.vconcat(proposal_ub, index0_ub,
                                  self.slice_len // 16, 0)
        self.tik_instance.vconcat(proposal_ub, index1_ub,
                                  self.slice_len // 16, 1)
        self.tik_instance.vconcat(proposal_ub, index0_ub,
                                  self.slice_len // 16, 2)
        self.tik_instance.vconcat(proposal_ub, index0_ub,
                                  self.slice_len // 16, 3)
        self.tik_instance.vconcat(proposal_ub, prob_max_ub,
                                  self.slice_len // 16, 4)

    def single_label_score_sort_all(self, per_class_ub, out_per_class_ub,
                                    per_class_ub_num):
        """
        Function: sort proposal by score
        """
        tmp_per_class_ub = self.gen_ub("float16", self.slice_len * 8,
                                       "tmp_per_class_ub")
        tmp_sort_ub = self.gen_ub("float16", self.slice_len * 8, "tmp_sort_ub")
        tmp_out_per_class_ub = self.gen_ub("float16", per_class_ub_num * 2,
                                           "tmp_out_per_class_ub")

        self.tik_instance.vrpsort16(tmp_per_class_ub, per_class_ub,
                                    self.slice_len // 16)
        self.tik_instance.vector_dup(128, tmp_sort_ub, 0.0,
                                     self.slice_len * 8 // 128, 1, 8, 0)
        self.tik_instance.vector_dup(128, tmp_out_per_class_ub, 0.0,
                                     per_class_ub_num * 2 // 128, 1, 8, 0)

        num_sort4 = (self.slice_len // 16) // 4  # num_sort4:16
        with self.tik_instance.for_range(0, num_sort4) as sort_idx:
            offset = sort_idx * 16 * 4 * 8
            self.tik_instance.vmrgsort4(tmp_sort_ub[offset], 
                 [tmp_per_class_ub[offset], tmp_per_class_ub[offset + 128],
                 tmp_per_class_ub[offset + 256],
                 tmp_per_class_ub[offset + 384]],
                 [16, 16, 16, 16], False, 15, 1, None)
        num_sort4 = num_sort4 // 4  # num_sort4:4
        with self.tik_instance.for_range(0, num_sort4) as sort_idx:
            offset = sort_idx * 16 * 8 * 4 * 4
            self.tik_instance.vmrgsort4(tmp_per_class_ub[offset],
                 [tmp_sort_ub[offset], tmp_sort_ub[offset + 512],
                 tmp_sort_ub[offset + 1024], tmp_sort_ub[offset + 1536]],
                 [64, 64, 64, 64], False, 15, 1, None)
        num_sort4 = num_sort4 // 4 # num_sort4:1
        with self.tik_instance.for_range(0, num_sort4) as sort_idx:
            offset = sort_idx * 16 * 8 * 4 * 4
            self.tik_instance.vmrgsort4(tmp_sort_ub[offset], 
                 [tmp_per_class_ub[offset], tmp_per_class_ub[offset + 2048],
                 tmp_per_class_ub[offset + 4096], 
                 tmp_per_class_ub[offset + 6144]],
                 [256, 256, 256, 256], False, 15, 1, None)

        self.tik_instance.vmrgsort4(tmp_out_per_class_ub, [tmp_sort_ub[0],
                        out_per_class_ub, out_per_class_ub, out_per_class_ub],
                        [self.compare_len, per_class_ub_num // 8, 0, 0],
                        False, 3, 1, None)

        self.tik_instance.data_move(out_per_class_ub, tmp_out_per_class_ub, 
                                    0, 1, per_class_ub_num // 16, 0, 0)

    def  get_index(self, output_index_ub, index_proposal):
        offset = self.tik_instance.Scalar("int32")
        length = self.topk_align_128 * 128
        index0_int_ub = self.gen_ub("int32", length, "index0_int_ub")
        index1_int_ub = self.gen_ub("int32", length, "index1_int_ub")
        index0_hf_ub = self.gen_ub(self.dtype, length, "index1_hf_ub")
        index1_hf_ub = self.gen_ub(self.dtype, length, "index1_hf_ub")
        index_slicelen_int_ub = self.gen_ub("int32", length,
                                "index_slicelen_int_ub")

        self.tik_instance.vextract(index0_hf_ub, index_proposal[0],
                                   self.topk_align_128 * 8, 0)
        self.tik_instance.vextract(index1_hf_ub, index_proposal[0],
                                   self.topk_align_128 * 8, 1)

        self.tik_instance.vconv(64, 'round', index0_int_ub, index0_hf_ub,
                                self.topk_align_128, 1, 1, 8, 4)
        self.tik_instance.vconv(64, 'round', 
                                index0_int_ub[self.topk_align_128 * 64], 
                                index0_hf_ub[self.topk_align_128 * 64], 
                                self.topk_align_128, 1, 1, 8, 4)
        self.tik_instance.vconv(64, 'round', index1_int_ub, index1_hf_ub,
                                self.topk_align_128, 1, 1, 8, 4)
        self.tik_instance.vconv(64, 'round', 
                                index1_int_ub[self.topk_align_128 * 64],
                                index1_hf_ub[self.topk_align_128 * 64],
                                self.topk_align_128, 1, 1, 8, 4)

        self.tik_instance.vector_dup(64, index_slicelen_int_ub, 
             self.slice_len, self.topk_align_128 * 2, 1, 8)

        self.tik_instance.vmul(64, index0_int_ub, index0_int_ub, 
             index_slicelen_int_ub, self.topk_align_128 * 2, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(64, output_index_ub, index0_int_ub,
             index1_int_ub, self.topk_align_128 * 2, 1, 1, 1, 8, 8, 8)

    def sort_topk(self):
        loop_times = self.ceil_div_offline(self.shape_align_16 * 16,
                                           self.slice_len)
        loop_length = self.tik_instance.Scalar("int32")
        one_scalar = self.tik_instance.Scalar(self.dtype)
        one_scalar.set_as(1.0)
        out_proposal_num = self.topk_align_16 * 16 * 8

        one_ub = self.gen_ub(self.dtype, self.slice_len, "one_ub")
        index0_ub = self.gen_ub(self.dtype, self.slice_len, "index0_ub")
        index1_ub = self.gen_ub(self.dtype, self.slice_len, "index1_ub")
        index2_ub = self.gen_ub(self.dtype, self.slice_len, "index2_ub")
        prob_max_ub = self.gen_ub(self.dtype, self.slice_len, "prob_max_ub")
        proposal_ub1 = self.gen_ub(self.dtype, self.slice_len * 8,
                                   "proposal_ub1")
        out_proposal_ub = self.gen_ub(self.dtype, out_proposal_num,
                                      "out_proposal_ub")

        self.tik_instance.vector_dup(128, one_ub, one_scalar,
                                     self.slice_len // 128, 1, 8)
        self.tik_instance.vector_dup(128, out_proposal_ub, 0.0,
                                     out_proposal_num // 128, 1, 8)
        self.tik_instance.vector_dup(128, index0_ub, 0, 
                                     self.slice_len // 128, 1, 8)
        self.gen_index(index1_ub, self.slice_len)
        self.tik_instance.vadd(128, index0_ub, index0_ub, index0_ub, 
                               self.slice_len // 128, 1, 1, 1, 8, 8, 8)

        with self.tik_instance.for_range(0, loop_times) as loop_idx:
            offset = loop_idx * self.slice_len
            loop_length.set_as(self.slice_len)

            with self.tik_instance.if_scope(loop_idx == loop_times-1):
                loop_length.set_as(self.shape_align_16 * 16 - offset)
                self.tik_instance.vector_dup(128, prob_max_ub, 0.0, 
                                             self.slice_len // 128, 1, 8)
                self.gen_index(index2_ub, loop_length)
                self.gen_proposal(self.input_gm, proposal_ub1, prob_max_ub,
                                  index0_ub, index2_ub, loop_length, offset,
                                  loop_idx)
            with self.tik_instance.else_scope():
                self.gen_proposal(self.input_gm, proposal_ub1, prob_max_ub,
                                  index0_ub, index1_ub, loop_length, offset,
                                  loop_idx)

            #index0 + 1 
            self.tik_instance.vadd(128, index0_ub, index0_ub, one_ub,
                                   self.slice_len // 128, 1, 1, 1, 8, 8, 8)
            self.single_label_score_sort_all(proposal_ub1, out_proposal_ub,
                                             out_proposal_num)
        
        self.get_index(self.output_index_ub, out_proposal_ub)
        return self.output_index_ub

