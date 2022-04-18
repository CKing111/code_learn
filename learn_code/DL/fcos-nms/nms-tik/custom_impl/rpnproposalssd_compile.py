from te import tik
import numpy
import mdc_tik_util as mtu
from rpd import RpnProposalSsdBatchMore
from util import OpLog as oplog


@mtu.glog_print_value
def rpnproposalssd(rpn_cls_prob_reshape,
                   rpn_bbox_pred,
                   im_info,
                   rois,
                   anchor_height,
                   anchor_width,
                   bbox_mean,
                   bbox_std,
                   intype,
                   top_n,
                   min_size_mode,
                   min_size_w,
                   min_size_h,
                   heat_map_a,
                   overlap_ratio,
                   threshold_objectness,
                   max_candidate_n,
                   refine_out_of_map_bbox,
                   use_soft_nms,
                   voting,
                   vote_iou,
                   kernel_name="rpnproposalssd"):
    # parse params
    cls_shape = rpn_cls_prob_reshape.get("shape")
    bbox_shape = rpn_bbox_pred.get("shape")
    im_info_shape = im_info.get("shape")
    out_rois_shape = rois.get("shape")
    top_n = top_n[0]
    heat_map_a = heat_map_a[0]
    overlap_ratio = overlap_ratio[0]
    max_candidate_n = max_candidate_n[0]
    use_soft_nms = use_soft_nms[0]
    voting = voting[0]
    vote_iou = vote_iou[0]
    check_params(intype, min_size_w, min_size_h, heat_map_a,
                 overlap_ratio, threshold_objectness, im_info_shape,
                 top_n, max_candidate_n, vote_iou)

    obj = RpnProposalSsdBatchMore(cls_shape, bbox_shape,
                                    im_info_shape, out_rois_shape,
                                    anchor_height, anchor_width,
                                    bbox_mean, bbox_std,
                                    intype, top_n, min_size_mode,
                                    min_size_w, min_size_h,
                                    heat_map_a, overlap_ratio,
                                    threshold_objectness, max_candidate_n,
                                    refine_out_of_map_bbox, use_soft_nms,
                                    voting, vote_iou, kernel_name)
    obj.ask_for_tik_res()
    obj.compute()
    obj.compile()


def check_params(intype, min_size_w, min_size_h, heat_map_a,
                 overlap_ratio, threshold_objectness, im_info_shape,
                 top_n, max_candidate_n, vote_iou):
    oplog.check_gt(min_size_w, 0, "min_size_w must greater than 0")
    oplog.check_gt(min_size_h, 0, "min_size_h must greater than 0")
    oplog.check_eq(intype, "float16", "data type must be float16")
    oplog.check_gt(heat_map_a, 0, "heat_map_a must greater than 0")
    oplog.check_ge(overlap_ratio, 0,
                   "overlap_ratio must greater or equal to 0")
    oplog.check_le(overlap_ratio, 1, "overlap_ratio must lower or equal to 1")
    oplog.check_ge(threshold_objectness, 0,
                   "threshold_objectness must greater or equal to 0")
    oplog.check_le(threshold_objectness, 1,
                   "threshold_objectness must lower or equal to 1")
    oplog.check_le(im_info_shape[0], 4,
                   "im_info_shape[0] must lower than or equal to 4")
    oplog.check_eq(im_info_shape[1], 6, "im_info_shape[1] must equal to 6")
    oplog.check_eq(im_info_shape[2], 1, "im_info_shape[2] must equal to 1")
    oplog.check_eq(im_info_shape[3], 1, "im_info_shape[3] must equal to 1")
    oplog.check_eq(top_n, 300, "top_n must equal to 300")
    oplog.check_eq(max_candidate_n, 3000, 
                "max_candidate_n must equal to 3000")
    oplog.check_gt(vote_iou, 0, "vote_iou must greater than 0")
    oplog.check_lt(vote_iou, 1, "vote_iou must lower than 1")

