from te import tik
import numpy as np
from blob import Blob
import mdc_tik_util as mtu
import nms
from util import OpLog as log


class RpnProposalSsdBatchOne(object):
    """
        NOTE: rpn proposalssd base support one core
    """

    @mtu.glog_print_value
    def __init__(self, anchor_height, anchor_width, bbox_mean, bbox_std,
                 cls_shape, bbox_shape, im_info_shape, top_n,
                 min_size_mode,
                 min_size_w, min_size_h, heat_map_a, overlap_threshold,
                 threshold_objectness, max_candidate_n,
                 refine_out_of_map_bbox, use_soft_nms, voting, vote_iou,
                 in_data_type, tik_instance=None,
                 input1_cls_prob_gm=None, input1_cls_prob_gm_off=0,
                 input2_bbox_pred_gm=None, input2_bbox_pred_gm_off=0,
                 input3_im_info_gm=None, input3_im_info_gm_off=0,
                 output_rois_gm=None, output_rois_gm_off=0):
        self.bbox_mean_ = np.array(bbox_mean)
        self.bbox_std_ = np.array(bbox_std)
        self.heat_map_a_ = heat_map_a
        self.min_size_h_ = min_size_h
        self.min_size_w_ = min_size_w
        self.min_size_mode_ = min_size_mode
        self.threshold_objectness_ = threshold_objectness
        self.anchor_width_ = np.array(anchor_width)
        self.anchor_height_ = np.array(anchor_height)
        self.refine_out_of_map_bbox_ = refine_out_of_map_bbox
        self.overlap_threshold_ = overlap_threshold
        self.top_n_ = top_n
        self.use_soft_nms_ = use_soft_nms
        self.voting_ = voting
        self.vote_iou_ = vote_iou
        self.max_candidate_n_ = max_candidate_n
        self.cls_shape_ = cls_shape
        self.bbox_shape_ = bbox_shape
        self.im_info_shape_ = im_info_shape
        self.heat_map_b_ = 0.0
        self.in_data_type_ = in_data_type
        self.num_anchors_ = self.anchor_height_.size
        self.batch_size_ = Blob.batch_size(self.bbox_shape_)
        self.intype_ = "float16"
        self.intype_bytes_ = 2 if self.intype_ == "float16" else 4
        self.confidence_threshold_ = 0.0
        self.proposals_l1_ = None
        self.feature_map_size_ = Blob.count(self.bbox_shape_, 2)
        self.out_rois_shape_ = [self.batch_size_, top_n, 8, 1]
        self._bsz01_ = 1.0
        self.anc_data_ = None
        self.negative_ratio_ = float(-2.0)
        self.positive_ratio_ = float(2.0)
        self.anchor_ratio_ = 0.5
        self.tik_instance_ = tik_instance
        self.input1_cls_prob_gm_ = input1_cls_prob_gm
        self.input1_cls_prob_gm_off_ = input1_cls_prob_gm_off
        self.input2_bbox_pred_gm_ = input2_bbox_pred_gm
        self.input2_bbox_pred_gm_off_ = input2_bbox_pred_gm_off
        self.input3_im_info_gm_ = input3_im_info_gm
        self.input3_im_info_gm_off_ = input3_im_info_gm_off
        self.output_rois_gm_ = output_rois_gm
        self.output_rois_gm_off_ = output_rois_gm_off
        self.non_zero_score_cnt_ = None
        self.proposals_cnt_scalar_ = None
        self.coord_id_l1_ = None

    def __call__(self):
        self.compute()

    def ask_for_tik_res(self):
        self.proposals_l1_ = self.tik_instance_.Tensor(self.intype_, (
                            self.feature_map_size_ * 8 * self.num_anchors_,),
                            name="input1_cls_prob_l1",
                            scope=tik.scope_cbuf)
        self.non_zero_score_cnt_ = self.tik_instance_.Scalar(dtype="float16",
                                                        name="non_zero_score",
                                                        init_value=0.0)
        self.proposals_cnt_scalar_ = self.tik_instance_.Scalar(dtype="int32",
                                                name="proposals_cnt_scalar",
                                                init_value=0)
        self.coord_id_l1_ = self.tik_instance_.Tensor(self.intype_, (
                                self.feature_map_size_ * self.num_anchors_,),
                                name="coord_id_l1",
                                scope=tik.scope_cbuf)
        self.proposals_max_candidate_l1_ = self.tik_instance_.Tensor(
                                            self.intype_,
                                            (self.max_candidate_n_, 8),
                                            name="proposals_max_candidate_l1",
                                            scope=tik.scope_cbuf)

    def check_param(self):
        """
        NOTE: check params right or now
        ---------------------------------------------------------
        anchor_height.size == anchor_width_.size

        """
        log.check_eq(self.anchor_height_.size, self.anchor_width_.size,
                     "anchor_height_size(%d) != anchor_width_size(%d)" % (
                         self.anchor_height_.size, self.anchor_width_.size))
        log.check_gt(self.anchor_height_.size, 0,
                     "anchor_height_size(%d) not fit" % (
                         self.anchor_height_.size))
        log.check_eq(self.bbox_mean_.size,
                     self.bbox_std_.size,
                     "bbox_mean_size(%d) != bbox_std_size(%d)" % (
                     self.bbox_mean_.size,
                     self.bbox_std_.size))
        log.check_eq(self.bbox_mean_.size, 4,
            "bbox_mean size is %d and not equal to 4" % self.bbox_mean_.size)
        log.check_eq(Blob.count(self.im_info_shape_, 1), 16,
                     "count(im_info_shape[1]) should be equal to 16")
        log.check_eq(Blob.channels(self.cls_shape_),
                     self.num_anchors_ * 2,
                "cls_shape channels should be eq to self.num_anchors * 2")
        log.check_eq(Blob.batch_size(self.cls_shape_),
                     Blob.batch_size(self.bbox_shape_),
                "cls_shape batch size should be eq to bbox_shape batch size")
        log.check_eq(Blob.channels(self.bbox_shape_),
                     self.num_anchors_ * 4,
                "cls_shape channels should be eq to self.num_anchors * 4")
        log.check_eq(Blob.height(self.cls_shape_),
                     Blob.height(self.bbox_shape_),
                     "input cls and bbox feature map size not equal")
        log.check_eq(Blob.width(self.cls_shape_),
                     Blob.width(self.bbox_shape_),
                     "input cls and bbox feature map size not equal")

    def calc_anc_data(self):
        anchor_x1_vec_ = np.subtract(self.anchor_width_,
                                     self._bsz01_) / self.negative_ratio_
        anchor_y1_vec_ = np.subtract(self.anchor_height_,
                                     self._bsz01_) / self.negative_ratio_
        anchor_x2_vec_ = np.subtract(self.anchor_width_,
                                     self._bsz01_) / self.positive_ratio_
        anchor_y2_vec_ = np.subtract(self.anchor_height_,
                                     self._bsz01_) / self.positive_ratio_

        anchor_width = np.add((anchor_x2_vec_ - anchor_x1_vec_), self._bsz01_)
        anchor_height = np.add((anchor_y2_vec_ - anchor_y1_vec_), self._bsz01_)
        anchor_ctr_x = anchor_x1_vec_ + self.anchor_ratio_ * (
                    anchor_width - self._bsz01_)
        anchor_ctr_y = anchor_y1_vec_ + self.anchor_ratio_ * (
                    anchor_height - self._bsz01_)

        self.anc_data_ = np.transpose(
            [anchor_ctr_x, anchor_ctr_y, anchor_width, anchor_height])
        self.anc_data_ = self.anc_data_.reshape(self.anchor_width_.size, 4, 1,
                                                1)

    def feature_score_reset0_lt_threshold_ojectness(self, score_ub, zero_ub,
                                                    axis):
        # zero[0, :] dup 0
        mtu.vector_dup_align256(self.tik_instance_, zero_ub, 0,
                                self.feature_map_size_, self.intype_bytes_, 0)
        # initialize zero[1, :] euqals to threshold_objectness
        mtu.vector_dup_align256(self.tik_instance_, zero_ub,
                                self.feature_map_size_, self.feature_map_size_,
                                self.intype_bytes_, self.threshold_objectness_)
        # set score 0 when score is less than threshold_objectness 
        mtu.vector_cmp_assign(self.tik_instance_, score_ub, 0, zero_ub,
                              1 * self.feature_map_size_,
                              self.feature_map_size_,
                              self.intype_bytes_, zero_ub, 0, score_ub, 0,
                              score_ub, 0)

    def calc_ctx_cty_tw_th(self, idx, tgt_ub, width_ub, height_ub, zero_ub,
                            anchor_ctr_x, anchor_ctr_y,
                            anchor_width, anchor_height):
        # calc ctx: tgt_ub[0, :]
        if (idx == 0):
            # calc input_ctr_x
            # w: zero_ub[0, ], input_ctr_x: zero_ub[0, ]
            self.tik_instance_.data_move(zero_ub, width_ub, 0, 1,
                                            self.feature_map_size_ *
                                            self.intype_bytes_ // 32,
                                            0, 0)
            mtu.vmuls_vadds_align256(self.tik_instance_, zero_ub, 0,
                                        float(self.heat_map_a_),
                                        float(
                                            self.heat_map_b_ + anchor_ctr_x),
                                        self.feature_map_size_,
                                        self.intype_bytes_)
            # calc ctx
            mtu.vmuls_vadd_align256(self.tik_instance_, tgt_ub,
                                    idx * self.feature_map_size_,
                                    float(anchor_width), zero_ub,
                                    self.feature_map_size_,
                                    self.intype_bytes_)
        # calc cty: tgt_ub[1, :]
        elif (idx == 1):
            # calc input_ctr_y: equals to zero_ub[0, :]
            self.tik_instance_.data_move(zero_ub, height_ub, 0, 1,
                                            self.feature_map_size_ *
                                            self.intype_bytes_ // 32,
                                            0, 0)
            mtu.vmuls_vadds_align256(self.tik_instance_, zero_ub, 0,
                                        float(self.heat_map_a_),
                                        float(
                                            self.heat_map_b_ + anchor_ctr_y),
                                        self.feature_map_size_,
                                        self.intype_bytes_)
            # calc cty
            mtu.vmuls_vadd_align256(self.tik_instance_, tgt_ub,
                                    idx * self.feature_map_size_,
                                    float(anchor_height), zero_ub,
                                    self.feature_map_size_,
                                    self.intype_bytes_)
        # calc tw: tgt_ub[2, :]
        elif (idx == 2):
            mtu.vexp_mul_scalar(self.tik_instance_, tgt_ub,
                                idx * self.feature_map_size_,
                                float(anchor_width),
                                self.feature_map_size_,
                                self.intype_bytes_)
        # calc th: euqals to tgt_ub[3, :]
        elif (idx == 3):
            mtu.vexp_mul_scalar(self.tik_instance_, tgt_ub,
                                idx * self.feature_map_size_,
                                float(anchor_height),
                                self.feature_map_size_,
                                self.intype_bytes_)

    def feature_calc_ctx_cty_tw_th(self, tgt_ub, axis, width_ub, height_ub,
                                   zero_ub):
        anchor_ctr_x = self.anc_data_[axis, 0, 0, 0]
        anchor_ctr_y = self.anc_data_[axis, 1, 0, 0]
        anchor_width = self.anc_data_[axis, 2, 0, 0]
        anchor_height = self.anc_data_[axis, 3, 0, 0]

        for i in range(4):
            # calc tgt_ub[0]: tg0, tgt_ub[1]: tg1,
            # tgt_ub[2]: tg2, tgt_ub[3]: tg3
            mtu.vmuls_vadds_align256(self.tik_instance_, tgt_ub,
                                     i * self.feature_map_size_,
                                     float(self.bbox_std_[i]),
                                     float(self.bbox_mean_[i]),
                                     self.feature_map_size_,
                                     self.intype_bytes_)
            self.calc_ctx_cty_tw_th(i, tgt_ub, width_ub, height_ub, zero_ub,
                                    anchor_ctr_x, anchor_ctr_y,
                                    anchor_width, anchor_height)
    
    def calc_min_size_mode(self, zero_ub, feature_zero_ub, score_ub):
        with self.tik_instance_.new_stmt_scope():
            # cmp and set value, if min_size_mode_ = or mode
            with self.tik_instance_.if_scope(self.min_size_mode_):
                mtu.vector_cmp_assign(self.tik_instance_, zero_ub,
                                        2 * self.feature_map_size_, zero_ub,
                                        0, self.feature_map_size_,
                                        self.intype_bytes_, feature_zero_ub,
                                        0, score_ub, 0, score_ub, 0)
                mtu.vector_cmp_assign(self.tik_instance_, zero_ub,
                                        3 * self.feature_map_size_, zero_ub,
                                        1 * self.feature_map_size_,
                                        self.feature_map_size_,
                                        self.intype_bytes_, feature_zero_ub,
                                        0, score_ub, 0, score_ub, 0)
            with self.tik_instance_.else_scope():
                # if and mode
                # cmp and
                mtu.vector_cmp_assign(self.tik_instance_, zero_ub,
                                        2 * self.feature_map_size_, zero_ub,
                                        0, self.feature_map_size_,
                                        self.intype_bytes_, feature_zero_ub,
                                        0, score_ub, 0, zero_ub, 0)
                mtu.vector_cmp_assign(self.tik_instance_, zero_ub,
                                        3 * self.feature_map_size_, zero_ub,
                                        1 * self.feature_map_size_,
                                        self.feature_map_size_,
                                        self.intype_bytes_,
                                        feature_zero_ub, 0,
                                        score_ub, 0, zero_ub,
                                        1 * self.feature_map_size_)
                mtu.vector_cmp_assign(self.tik_instance_,
                                        zero_ub, 0, zero_ub,
                                        1 * self.feature_map_size_,
                                        self.feature_map_size_,
                                        self.intype_bytes_, zero_ub,
                                        1 * self.feature_map_size_, zero_ub,
                                        0, score_ub, 0)

    def feature_calc_min_size_mode(self, tgt_data_ub,
                                   score_ub, zero_ub, axis):
        # dump self.mini_size_w in zero_ub[0, :]
        mtu.vector_dup_align256(self.tik_instance_, zero_ub, 0,
                                self.feature_map_size_,
                                self.intype_bytes_, self.min_size_w_)
        # dump self.mini_size_h in zero_ub[1, :]
        mtu.vector_dup_align256(self.tik_instance_, zero_ub,
                                self.feature_map_size_,
                                self.feature_map_size_,
                                self.intype_bytes_, self.min_size_h_)
        with self.tik_instance_.new_stmt_scope():
            # feature_zero_ub
            feature_zero_ub = self.tik_instance_.Tensor(
                                                    self.intype_,
                                                    (self.feature_map_size_,),
                                                    name="feature_zero_ub",
                                                    scope=tik.scope_ubuf)
            mtu.vector_dup_align256(self.tik_instance_, feature_zero_ub, 0,
                                    self.feature_map_size_,
                                    self.intype_bytes_, 0)
            # calc rbx - ltx + bsz01, equals to target0 zero_ub[2, :]
            # calc rby - lty + bsz01, equals to target1 zero_ub[3, :]
            mtu.vsub_align256(self.tik_instance_, zero_ub,
                              2 * self.feature_map_size_, tgt_data_ub,
                              2 * self.feature_map_size_,
                              tgt_data_ub, 0, 2 * self.feature_map_size_,
                              self.intype_bytes_)
            mtu.vadds_align256(self.tik_instance_, zero_ub,
                               2 * self.feature_map_size_, self._bsz01_,
                               2 * self.feature_map_size_,
                               self.intype_bytes_)
            self.calc_min_size_mode(zero_ub, feature_zero_ub, score_ub)

    def feature_calc_ltx_lty(self, tgt_data_ub, zero_ub):
        """
            save ltx, lty in tgt_data_ub[0, :], tgt_data_ub[1, :]
        """
        # copy tw, th to zero_ub[0:1, :]
        mtu.data_move_align256(self.tik_instance_, tgt_data_ub,
                               2 * self.feature_map_size_, zero_ub, 0,
                               self.feature_map_size_ * 2,
                               self.intype_bytes_)
        # tw, th - self._bsz01_
        mtu.vadds_align256(self.tik_instance_, zero_ub, 0,
                           -1 * self._bsz01_,
                           2 * self.feature_map_size_,
                           self.intype_bytes_)
        # tw, th * 0.5
        mtu.vmuls_align256(self.tik_instance_, zero_ub, 0, 0.5,
                           2 * self.feature_map_size_, self.intype_bytes_)
        # calc array of (ctx, cty) sub array of (zero)
        mtu.vsub_align256(self.tik_instance_, tgt_data_ub, 0, tgt_data_ub, 0,
                          zero_ub, 0,
                          2 * self.feature_map_size_, self.intype_bytes_)

    def feature_calc_rbx_rby(self, tgt_data_ub):
        """
            save rbx, rby in tgt_data[2, :], tgt_data[3, :]
        """
        # calc array of (ltx, lty) add array of (tw, th)
        mtu.vadd_align256(self.tik_instance_, tgt_data_ub,
                          2 * self.feature_map_size_, tgt_data_ub, 0,
                          tgt_data_ub,
                          2 * self.feature_map_size_,
                          2 * self.feature_map_size_, self.intype_bytes_)
        # calc rbx, rby
        mtu.vadds_align256(self.tik_instance_, tgt_data_ub,
                           2 * self.feature_map_size_, -1 * self._bsz01_,
                           2 * self.feature_map_size_, self.intype_bytes_)

    def feature_calc_refine_out_of_map_bbox(self, tgt_data_ub, im_info_ub,
                                            zero_ub):
        # cmp with 0 and get max: relu, tgt_data_ub[0:3, :]
        mtu.vrelu_align256(self.tik_instance_, tgt_data_ub, 0,
                           tgt_data_ub, 0,
                           self.feature_map_size_ * 4,
                           self.intype_bytes_)
        # cmp with width or height
        # get cmp scalar
        for i in range(4):
            cmp_scalar = im_info_ub[0] if i % 2 == 0 else im_info_ub[1]
            input_scalar = self.tik_instance_.Scalar(dtype='float16')
            input_scalar.set_as(cmp_scalar)
            # calc im_info - 1
            mtu.vector_dup_align256(self.tik_instance_, zero_ub, 0,
                                    self.feature_map_size_,
                                    self.intype_bytes_,
                                    input_scalar)
            mtu.vadds_align256(self.tik_instance_, zero_ub, 0, -1,
                               self.feature_map_size_, self.intype_bytes_)
            off = i * self.feature_map_size_
            mtu.vector_cmp_assign(self.tik_instance_, tgt_data_ub, off,
                                  zero_ub, 0, self.feature_map_size_,
                                  self.intype_bytes_, tgt_data_ub, off,
                                  zero_ub, 0, tgt_data_ub, off)

    def record_non_zero_score(self, score_ub, zero_ub):
        # init self.confidence_threshold_ vector, zero_ub[0, :]
        mtu.vector_dup_align256(self.tik_instance_, zero_ub,
                                0 * self.feature_map_size_,
                                self.feature_map_size_,
                                self.intype_bytes_,
                                self.confidence_threshold_)
        # dup 1 vector
        mtu.vector_dup_align256(self.tik_instance_, zero_ub,
                                1 * self.feature_map_size_,
                                self.feature_map_size_,
                                self.intype_bytes_,
                                1.0)
        # dup 0 vector
        mtu.vector_dup_align256(self.tik_instance_, zero_ub,
                                2 * self.feature_map_size_,
                                self.feature_map_size_,
                                self.intype_bytes_, 0.0)
        # cmp with threshold, save to zero_ub[0, :]
        mtu.vector_cmp_assign_gt(self.tik_instance_, score_ub, 0,
                                 zero_ub, 0,
                                 self.feature_map_size_,
                                 self.intype_bytes_, zero_ub,
                                 1 * self.feature_map_size_, zero_ub,
                                 2 * self.feature_map_size_, zero_ub, 0)
        # sum
        self.tik_instance_.vcadd(128, zero_ub, zero_ub, 8, 1, 1, 8)
        zero_ub[8].set_as(self.non_zero_score_cnt_)
        self.tik_instance_.vcadd(9, zero_ub, zero_ub, 8, 8, 1, 8)
        self.non_zero_score_cnt_.set_as(zero_ub[0])

    def sort_feature_map_proposals(self, proposals_ub, axis):
        with self.tik_instance_.new_stmt_scope():
            tmp_ub = self.tik_instance_.Tensor(self.intype_,
                                               (self.feature_map_size_ * 8,),
                                               name="tmp_ub",
                                               scope=tik.scope_ubuf)
            # sort proposals_ub: sort 16 proposals
            self.tik_instance_.vrpsort16(tmp_ub, proposals_ub,
                                         self.feature_map_size_ // 16)
            # sort 64 proposals
            loop_times = self.feature_map_size_ // 16 // 4
            with self.tik_instance_.for_range(0, loop_times) as cnt:
                p0_off = 16 * 8 * 4 * cnt + 0
                p1_off = 16 * 8 + p0_off
                p2_off = 16 * 8 + p1_off
                p3_off = 16 * 8 + p2_off
                self.tik_instance_.vmrgsort4(proposals_ub[p0_off],
                                             (tmp_ub[p0_off], tmp_ub[p1_off],
                                              tmp_ub[p2_off], tmp_ub[p3_off]),
                                             (16, 16, 16, 16),
                                             False, 15, 1)
            # sort 64 x 4 = 256 proposals
            loop_times = self.feature_map_size_ // 64 // 4
            with self.tik_instance_.for_range(0, loop_times) as cnt:
                p0_off = 64 * 8 * 4 * cnt + 0
                p1_off = 64 * 8 + p0_off
                p2_off = 64 * 8 + p1_off
                p3_off = 64 * 8 + p2_off
                self.tik_instance_.vmrgsort4(tmp_ub[p0_off], (
                proposals_ub[p0_off], proposals_ub[p1_off],
                proposals_ub[p2_off], proposals_ub[p3_off]),
                                             (64, 64, 64, 64), False, 15, 1)
            # sort feature proposals, 256 * 4
            loop_times = self.feature_map_size_ // 256 // 4
            with self.tik_instance_.for_range(0, loop_times) as cnt:
                p0_off = 256 * 8 * 4 * cnt + 0
                p1_off = 256 * 8 + p0_off
                p2_off = 256 * 8 + p1_off
                p3_off = 256 * 8 + p2_off
                self.tik_instance_.vmrgsort4(proposals_ub[p0_off],
                                             (tmp_ub[p0_off], tmp_ub[p1_off],
                                              tmp_ub[p2_off], tmp_ub[p3_off]),
                                             (256, 256, 256, 256), False, 15,
                                             1)

    def concat_sorted_feature_proposals(self, tgt_data_ub, score_ub, axis):
        with self.tik_instance_.new_stmt_scope():
            proposals_ub = self.tik_instance_.Tensor(
                                                self.intype_,
                                                (self.feature_map_size_ * 8,),
                                                name="zero_ub",
                                                scope=tik.scope_ubuf)
            # concat score_data to proposals
            self.tik_instance_.vconcat(proposals_ub, score_ub,
                                       self.feature_map_size_ // 16, 4)
            # concat tgt_data to proposals
            with self.tik_instance_.for_range(0, 4) as idx:
                self.tik_instance_.vconcat(proposals_ub, tgt_data_ub[
                    idx * self.feature_map_size_],
                                           self.feature_map_size_ // 16, idx)
            # sort proposals feature map
            self.sort_feature_map_proposals(proposals_ub, axis)
            mtu.data_move_align256(self.tik_instance_, proposals_ub, 0,
                                   self.proposals_l1_,
                                   self.feature_map_size_ * 8 * axis,
                                   self.feature_map_size_ * 8,
                                   self.intype_bytes_)

    def compute_batch_one_and_sort_proposals(self, im_info_ub, score_ub, 
                                            tgt_data_ub, zero_ub, width_ub, 
                                            height_ub):
        for axis in range(self.anchor_height_.size):
            # copy cls prob data from gm to score ub
            cls_prob_off = (self.anchor_width_.size + axis) * \
                            self.feature_map_size_
            tgt_data_gm_off = axis * 4 * self.feature_map_size_
            self.tik_instance_.data_move(score_ub,
                                            self.input1_cls_prob_gm_[
                                                self.input1_cls_prob_gm_off_
                                                + cls_prob_off],
                                            0,
                                            1,
                                            self.feature_map_size_ *
                                            self.intype_bytes_ // 32,
                                            0, 0)
            # move tgt data to ub
            self.tik_instance_.data_move(tgt_data_ub,
                                            self.input2_bbox_pred_gm_[
                                                self.input2_bbox_pred_gm_off_
                                                + tgt_data_gm_off],
                                            0, 1,
                                            self.feature_map_size_ * 4 *
                                            self.intype_bytes_ // 32,
                                            0, 0)
            # reset score if under threshold objectness
            self.feature_score_reset0_lt_threshold_ojectness(score_ub,
                                                            zero_ub,
                                                            axis)
            # calc tw/th/ctx/cty/ltx/lty/rbx/rby
            self.feature_calc_ctx_cty_tw_th(tgt_data_ub, axis, width_ub,
                                            height_ub, zero_ub)
            # calc ltx, lty
            self.feature_calc_ltx_lty(tgt_data_ub, zero_ub)
            # # calc feature map rbx, rby
            self.feature_calc_rbx_rby(tgt_data_ub)  # ok
            # calc feature refine_out_of_map_bbox
            self.feature_calc_refine_out_of_map_bbox(tgt_data_ub,
                                                        im_info_ub,
                                                        zero_ub)
            # calc feature map size min_size_mode
            self.feature_calc_min_size_mode(tgt_data_ub, score_ub,
                                            zero_ub,
                                            axis)
            # calc non-zero value
            self.record_non_zero_score(score_ub, zero_ub)
            # concat feature proposals, sort and copy to l1
            self.concat_sorted_feature_proposals(tgt_data_ub, score_ub,
                                                    axis)
        # copy non zero score count to proposals_cnt_scalar
        mtu.vconv_fp16_scalar_to_int32(self.tik_instance_,
                                        self.non_zero_score_cnt_,
                                        self.proposals_cnt_scalar_)

    def compute_batch_one_and_sort_proposals_l1(self):
        with self.tik_instance_.new_stmt_scope():
            # ub_buffer, depth is 4 # equal 4k
            zero_ub = self.tik_instance_.Tensor(self.intype_,
                                                (self.feature_map_size_ * 4,),
                                                name="zero_ub",
                                                scope=tik.scope_ubuf)
            # im_info_ub
            im_info_element_number = mtu.data_addr_align32(
                Blob.count(self.im_info_shape_, 1), self.intype_bytes_)

            im_info_ub = self.tik_instance_.Tensor(self.intype_,
                                                   (im_info_element_number,),
                                                   name="im_info_ub",
                                                   scope=tik.scope_ubuf)
            mtu.data_move_align256(self.tik_instance_,
                                   self.input3_im_info_gm_,
                                   self.input3_im_info_gm_off_, im_info_ub,
                                   0, im_info_element_number,
                                   self.intype_bytes_)
            # init height ub
            width_ub = self.tik_instance_.Tensor(self.intype_,
                                                 (self.feature_map_size_,),
                                                 name="width_ub",
                                                 scope=tik.scope_ubuf)
            mtu.vdup_width_plane(self.tik_instance_, width_ub,
                                 Blob.height(self.bbox_shape_),
                                 Blob.width(self.bbox_shape_),
                                 self.intype_bytes_)
            # init width ub
            height_ub = self.tik_instance_.Tensor(self.intype_,
                                                  (self.feature_map_size_,),
                                                  name="height_ub",
                                                  scope=tik.scope_ubuf)
            mtu.vdup_height_plane(self.tik_instance_, height_ub,
                                  Blob.height(self.bbox_shape_),
                                  Blob.width(self.bbox_shape_),
                                  self.intype_bytes_)
            # malloc score_ub
            score_ub = self.tik_instance_.Tensor(self.intype_,
                                                 (self.feature_map_size_,),
                                                 name="score_ub",
                                                 scope=tik.scope_ubuf)
            # malloc tgt_data_ub
            tgt_data_ub = self.tik_instance_.Tensor(self.intype_, (
            self.feature_map_size_ * 4,), name="tgt_data_ub",
                                                    scope=tik.scope_ubuf)
            self.compute_batch_one_and_sort_proposals(im_info_ub, score_ub, 
                                            tgt_data_ub, zero_ub, width_ub, 
                                            height_ub)

    def sort_proposals_and_get_ahead_subprocess(self, sorting_proposals_ub,
                                        loop_times, tmp_proposals_ub):
        mtu.data_move_align256(self.tik_instance_,
                                self.proposals_l1_, 0,
                                sorting_proposals_ub,
                                0, self.feature_map_size_ * 8 * 3,
                                self.intype_bytes_)
        with self.tik_instance_.for_range(1, loop_times) as cnt:
            p0_off = 4096 * 8 * cnt + 0
            # copy cnt proposals 3 to ub
            mtu.data_move_align256(self.tik_instance_,
                                    self.proposals_l1_,
                                    p0_off, sorting_proposals_ub,
                                    self.feature_map_size_ * 8 * 3,
                                    self.feature_map_size_ * 8 * 3,
                                    self.intype_bytes_)
            # cmp 3072 proposals
            off = self.feature_map_size_ * 8 * 3
            self.tik_instance_.vmrgsort4(tmp_proposals_ub[0], (
            sorting_proposals_ub[0], sorting_proposals_ub[off],
            sorting_proposals_ub[0], sorting_proposals_ub[off]),
                                            (3072, 3072, 0, 0), False, 3, 1)
            mtu.data_move_align256(self.tik_instance_,
                                    tmp_proposals_ub, 0,
                                    sorting_proposals_ub,
                                    0, self.feature_map_size_ * 8 * 3,
                                    self.intype_bytes_)
        # copy sorted proposals to self.proposals_max_candidate_l1_
        self.tik_instance_.data_move(
            self.proposals_max_candidate_l1_[0, 0],
            tmp_proposals_ub, 0, 1,
            self.max_candidate_n_ * 8 // 16, 0, 0)

    def sort_proposals_and_get_ahead(self):
        with self.tik_instance_.new_stmt_scope():
            sorting_proposals_ub = self.tik_instance_.Tensor(
                                        self.intype_,
                                        (3 * 2 * self.feature_map_size_ * 8,),
                                        name="sorting_proposals_ub",
                                        scope=tik.scope_ubuf)
            tmp_proposals_ub = self.tik_instance_.Tensor(
                                        self.intype_,
                                        (3 * 2 * self.feature_map_size_ * 8,),
                                        name="sorting_proposals_ub",
                                        scope=tik.scope_ubuf)
            # sorted 4096 proposals: 4096 x 3 + 3072 proposals
            loop_times = (self.num_anchors_ - 1) // 4 + 1
            for cnt in range(loop_times):
                p0_off = 1024 * 8 * 4 * cnt + 0
                p1_off = 1024 * 8 + 0
                p2_off = 1024 * 8 + p1_off
                p3_off = 1024 * 8 + p2_off
                valid = 15
                bete = 4
                if cnt == (loop_times - 1):
                    valid = 7
                    bete = 3
                mtu.data_move_align256(self.tik_instance_,
                                       self.proposals_l1_,
                                       p0_off, sorting_proposals_ub, 0,
                                       self.feature_map_size_ * 8 * bete,
                                       self.intype_bytes_)
                self.tik_instance_.vmrgsort4(tmp_proposals_ub[0],
                                             (sorting_proposals_ub[0],
                                              sorting_proposals_ub[p1_off],
                                              sorting_proposals_ub[p2_off],
                                              sorting_proposals_ub[p3_off]),
                                             (1024, 1024, 1024, 1024), False,
                                             valid, 1)
                mtu.data_move_align256(self.tik_instance_,
                                       tmp_proposals_ub, 0,
                                       self.proposals_l1_, p0_off,
                                       self.feature_map_size_ * 8 * bete,
                                       self.intype_bytes_)
            # get ahead 4096 proposals
            # buf less and cmp 2 group
            loop_times = 4
            self.sort_proposals_and_get_ahead_subprocess(sorting_proposals_ub,
                                        loop_times, tmp_proposals_ub)

    def proposals_nms(self, num):
        with self.tik_instance_.new_stmt_scope():
            with self.tik_instance_.if_scope(
                    self.proposals_cnt_scalar_ > self.max_candidate_n_):
                self.proposals_cnt_scalar_.set_as(self.max_candidate_n_)
            nms_param = [self.proposals_cnt_scalar_, self.top_n_, 128, 1]
            nms_instance = nms.OneCoreNMS(self.tik_instance_, nms_param)
            proposals_ok_ub = self.tik_instance_.Tensor(self.intype_,
                                                        (self.top_n_ * 8,),
                                                        name="proposals_ok_ub",
                                                        scope=tik.scope_ubuf)
            self.tik_instance_.data_move(proposals_ok_ub,
                                         self.proposals_max_candidate_l1_[
                                             0, 0], 0, 1,
                                         self.top_n_ * 8 // 16, 0, 0)
            sup_vector = nms_instance.nms_single_core(
                self.proposals_max_candidate_l1_, self.overlap_threshold_)

            with self.tik_instance_.for_range(0, self.max_candidate_n_) as cnt:
                with self.tik_instance_.if_scope(
                                        tik.all(
                                            sup_vector[cnt] == 0,
                                            cnt < self.proposals_cnt_scalar_,
                                            num < self.top_n_)):
                    with self.tik_instance_.for_range(0, 8) as i:
                        proposals_ok_ub[num * 8 + i].set_as(
                            proposals_ok_ub[cnt * 8 + i])
                    num.set_as(num + 1)
            self.tik_instance_.data_move(
                self.proposals_max_candidate_l1_[0, 0], proposals_ok_ub, 0, 1,
                self.top_n_ * 8 // 16, 0, 0)
            num.set_as(num)

    def copy_proposals_to_gm(self, num):
        with self.tik_instance_.new_stmt_scope():
            proposals_out_ub = self.tik_instance_.Tensor(
                                                    self.intype_,
                                                    (self.top_n_, 8),
                                                    name="proposals_out_ub",
                                                    scope=tik.scope_ubuf)
            mtu.vector_dup_align256(self.tik_instance_, proposals_out_ub, 0,
                                    self.top_n_ * 8, self.intype_bytes_, 0)
            self.tik_instance_.data_move(proposals_out_ub[0, 0],
                                         self.proposals_max_candidate_l1_[
                                             0, 0], 0, 1,
                                         self.top_n_ * 8 // 16, 0, 0)
            # insert proposals_num, should after proposals
            mtu.vconv_int32_scalar_to_fp16(self.tik_instance_, num,
                                           self.non_zero_score_cnt_)
            proposals_out_ub[0, 5].set_as(self.non_zero_score_cnt_)

            mtu.data_move_align256(self.tik_instance_, proposals_out_ub, 0,
                                   self.output_rois_gm_,
                                   self.output_rois_gm_off_,
                                   self.top_n_ * 8, self.intype_bytes_)

    def compute(self):
        """
            NOTE
            -------------------
            support more batches
        """
        self.check_param()
        self.calc_anc_data()
        self.ask_for_tik_res()
        self.compute_batch_one_and_sort_proposals_l1()
        with self.tik_instance_.if_scope(self.proposals_cnt_scalar_ == 0):
            self.copy_proposals_to_gm(0)
        with self.tik_instance_.else_scope():
            self.sort_proposals_and_get_ahead()
            with self.tik_instance_.if_scope(self.proposals_cnt_scalar_ == 1):
                self.copy_proposals_to_gm(1)
            with self.tik_instance_.else_scope():
                num = self.tik_instance_.Scalar("int32", name="num",
                                                init_value=0)
                self.proposals_nms(num)
                self.copy_proposals_to_gm(num)
