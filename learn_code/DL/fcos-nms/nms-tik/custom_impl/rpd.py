"""
    operator: rpnproposalssd
"""
import copy
from te import tik
import numpy as np
from blob import Blob
from rpnproposalssd_base import RpnProposalSsdBatchOne as RBO
import mdc_tik_util as mtu


class RpnProposalSsdBatchMore(object):
    """
    NOTE: baidu traffic light operators
    -------------------------------------------------
    Parameters:
        input list should be as numpy array type
    -------------------------------------------------
    """

    # @mtu.glog_print_value
    def __init__(self,
                 cls_shape,
                 bbox_shape,
                 im_info_shape,
                 rois,
                 anchor_height,
                 anchor_width,
                 bbox_mean,
                 bbox_std,
                 intype="float16",
                 top_n=300,
                 min_size_mode="HEIGHT_OR_WIDTH",
                 min_size_w=6.160560,
                 min_size_h=6.160560,
                 heat_map_a=8,
                 overlap_ratio=0.7,
                 threshold_objectness=0.2000,
                 max_candidate_n=3000,
                 refine_out_of_map_bbox=False,
                 use_soft_nms=False,
                 voting=False,
                 vote_iou=0.700000, kernel_name="rpnproposalssd"):
        self.kernel_name = kernel_name
        # prototxt arguments
        self.bbox_mean_ = bbox_mean
        self.bbox_std_ = bbox_std
        self.heat_map_a_ = heat_map_a
        self.min_size_h_ = min_size_h
        self.min_size_w_ = min_size_w
        self.min_size_mode_ = False \
            if min_size_mode == "HEIGHT_OR_WIDTH" else True
        self.threshold_objectness_ = threshold_objectness
        self.anchor_width_ = anchor_width
        self.anchor_height_ = anchor_height
        self.refine_out_of_map_bbox_ = refine_out_of_map_bbox
        self.overlap_threshold_ = overlap_ratio
        self.top_n_ = top_n
        self.max_condidate_n_ = max_candidate_n
        self.use_soft_nms_ = use_soft_nms
        self.voting_ = voting
        self.vote_iou_ = vote_iou

        # shape
        self.cls_shape_ = cls_shape
        self.bbox_shape_ = bbox_shape
        self.im_info_shape_ = list(im_info_shape)
        self.real_im_info_shape_ = copy.deepcopy(im_info_shape)
        self.im_info_shape_[1] = ((im_info_shape[1] - 1) // 16 + 1) * 16

        # self define
        self.num_anchors_ = len(self.anchor_height_)
        self.batch_size_ = Blob.batch_size(self.bbox_shape_)
        self.intype_ = intype
        self.intype_bytes_ = 2 if self.intype_ == "float16" else 4
        self.out_rois_shape_ = [Blob.batch_size(self.bbox_shape_),
                                self.top_n_,
                                8, 1]

    def ask_for_tik_res(self):
        """
            ask for global tik resources
            self.tik_instance_, self.input1_cls_prob_gm_,
            self.input2_bbox_pred_gm_,
            self.input3_im_info_gm_, self.output_rois_gm_
        """
        self.tik_instance_ = tik.Tik(tik.Dprofile())
        self.input1_cls_prob_gm_ = self.tik_instance_.Tensor(self.intype_, (
                                            Blob.count(self.cls_shape_, 0),),
                                            name="input1_cls_prob_gm",
                                            scope=tik.scope_gm)
        self.input2_bbox_pred_gm_ = self.tik_instance_.Tensor(self.intype_, (
                                            Blob.count(self.bbox_shape_, 0),),
                                            name="input2_bbox_pred_gm",
                                            scope=tik.scope_gm)
        self.input3_im_info_gm_ = self.tik_instance_.Tensor(self.intype_, (
                                        Blob.count(self.im_info_shape_, 0),),
                                        name="input3_im_info_gm",
                                        scope=tik.scope_gm)
        self.output_rois_gm_ = self.tik_instance_.Tensor(self.intype_, (
                                        Blob.count(self.out_rois_shape_, 0),),
                                        name="output_rois_gm",
                                        scope=tik.scope_gm)

    def compute(self):
        """
        compute rpnproposalssd
        """
        batch_size = Blob.batch_size(self.bbox_shape_)
        with self.tik_instance_.for_range(0, batch_size,
                                          block_num=batch_size) as core_id:
            cls_prob_gm_off = Blob.count(self.cls_shape_, 1) * core_id
            bbox_pred_gm_off = Blob.count(self.bbox_shape_, 1) * core_id
            im_info_gm_off = Blob.count(self.real_im_info_shape_, 1) * core_id
            print(self.real_im_info_shape_)
            output_rois_gm_off = Blob.count(self.out_rois_shape_, 1) * core_id
            obj_rbo = RBO(self.anchor_height_, self.anchor_width_,
                          self.bbox_mean_, self.bbox_std_, self.cls_shape_,
                          self.bbox_shape_, self.im_info_shape_, self.top_n_,
                          self.min_size_mode_, self.min_size_w_,
                          self.min_size_h_, self.heat_map_a_,
                          self.overlap_threshold_, self.threshold_objectness_,
                          self.max_condidate_n_, self.refine_out_of_map_bbox_,
                          self.use_soft_nms_, self.voting_,
                          self.vote_iou_, self.intype_, self.tik_instance_,
                          self.input1_cls_prob_gm_, cls_prob_gm_off,
                          self.input2_bbox_pred_gm_, bbox_pred_gm_off,
                          self.input3_im_info_gm_, im_info_gm_off,
                          self.output_rois_gm_, output_rois_gm_off)
            obj_rbo()

    def compile(self):
        """
            compile to kernel_meta *.json, *.o
        """
        self.tik_instance_.BuildCCE(kernel_name=self.kernel_name,
                                    inputs=(
                                        self.input1_cls_prob_gm_,
                                        self.input2_bbox_pred_gm_,
                                        self.input3_im_info_gm_),
                                    outputs=(self.output_rois_gm_,),
                                    enable_l2=False)
