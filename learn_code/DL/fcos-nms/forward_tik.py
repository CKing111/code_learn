from te import tik

import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes



    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        #建立boxs盒子数据打包容器
        # sampled_boxes = []
        self.sampled_boxs_ub = self.tik_instance.Tensor(self.dtype,                         
                      (len(zip(locations, box_cls, box_regression, centerness)) + 16, ),        #（，5）
                      name="sampled_boxs_ub", scope=tik.scope_gm)
        #for循环索引指数        
        idx = self.tik_instance.Scalar(dtype="int32", name="idx")

        # locations_zip_ub = self.tik_instance.Scalar(dtype="int32", name="locations_zip_ub")
        # box_cls_zip_ub = self.tik_instance.Scalar(dtype="int32", name="box_cls_zip_ub")
        # box_regression_zip_ub = self.tik_instance.Scalar(dtype="int32", name="box_regression_zip_ub")
        # centerness_zip_ub = self.tik_instance.Scalar(dtype="int32", name="centerness_zip_ub")
        
        #循环打包输入数据返回索引，并将数据带入单层金字塔特征图后处理
        #写入后处理结果（将正采样边界框的回归结果存入类BoxList）
        # for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
        #     sampled_boxes.append(
        #         self.forward_for_single_feature_map(
        #             l, o, b, c, image_sizes
        #         )
        #     )
        with self.tik_instance.for_range(0,len(zip(locations, box_cls, box_regression, centerness)) + 16) as idx:
            # locations_zip_ub.set_as(locations[idx])
            # box_cls_zip_ub.set_as(box_cls[idx])
            # box_regression_zip_ub.set_as(box_regression[idx])
            # centerness_zip_ub.set_as(centerness[idx])
            self.tik_instance.data_move(self.sampled_boxs_ub[idx], self.forward_for_single_feature_map
                                            (locations[idx], box_cls[idx], box_regression[idx], centerness[idx], image_sizes[idx]), 0, 1, 32, 0, 0)

        #建立boxs盒子数据打包容器
        # boxlists = list(zip(*sampled_boxes))
        self.boxlists_ub = self.tik_instance.Tensor(self.dtype,                         
                      (len(self.sampled_boxs_ub) + 16, ),
                      name="boxlists_ub", scope=tik.scope_gm)
        self.tik_instance.data_move(self.boxlists_ub, self.sampled_boxs_ub, 0, 1, 32, 0, 0)

        #cat_boxlist:将BoxList列表(具有相同图像大小)连接成单个BoxList
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]   #cat_boxlist:将BoxList列表(具有相同图像大小)连接成单个BoxList

        #布尔值默认bbox_aug_enabled=False
        # if not self.bbox_aug_enabled:       #        bbox_aug_enabled=True
        #     boxlists = self.select_over_all_levels(boxlists)
        with self.tik_instance.if_scope(not self.bbox_aug_enabled):
            self.tik_instance.data_move(self.boxlists_ub, self.select_over_all_levels(boxlists_ub),
                                                                 0, 1, 32, 0, 0)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        # num_images = len(boxlists)
        num_images = self.tik_instance.Scalar(dtype="int32", name="index")
        num_images.set_as(len(boxlists))
        # results = []
        self.results_ub = self.tik_instance.Tensor(self.dtype,                         #
                      (num_images + 16, ),
                      name="results_ub", scope=tik.scope_gm)   

        # for i in range(num_images):
        with self.tik_instance.for_range(0,num_images + 16) as i:
        
            # multiclass nms
            # result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(boxlist_ml_nms(boxlists[i], self.nms_thresh))
            self.result_ub = self.tik_instance.Tensor(self.dtype,                         #
                      (number_of_detections + 16, ),
                      name="result_ub", scope=tik.scope_gm)   

            self.tik_instance.data_move(result_ub, boxlist_ml_nms(boxlists[i], self.nms_thresh), 0, 1, 32, 0, 0)


            # Limit to max_per_image detections **over all classes**
            # if number_of_detections > self.fpn_post_nms_top_n > 0:
            with self.tik_instance.if_scope(tik.all(number_of_detections > self.fpn_post_nms_top_n,
                                                                                        self.fpn_post_nms_top_n > 0)):

                self.cls_scores = self.tik_instance.Tensor(self.dtype,                         #
                      (number_of_detections + 16, ),
                      name="cls_scores", scope=tik.scope_gm)   
                cls_scores.set_as(result_ub.index("scores"))            ###
                #返回cls_score最后一维度第k个元素最小值

                # image_thresh, _ = torch.kthvalue(
                #     cls_scores.cpu(),
                #     number_of_detections - self.fpn_post_nms_top_n + 1
                # )
                kthvalue_nums = self.tik_instance.Scalar(dtype="int32", name="index")
                kthvalue_nums.set_as(number_of_detections - self.fpn_post_nms_top_n + 1)
                
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

def boxlist_ml_nms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    # if nms_thresh <= 0:
    with self.tik_instance.if_scope(nms_thresh <= 0)):
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)
    keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)




def make_fcos_postprocessor(config):                    #引入fcos_core.config.defaults中的预设默认值
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH     #0.05
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N     #1000
    nms_thresh = config.MODEL.FCOS.NMS_TH               #0.6
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG #100

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh, #0.05
        pre_nms_top_n=pre_nms_top_n, #1000
        nms_thresh=nms_thresh, #0.6
        fpn_post_nms_top_n=fpn_post_nms_top_n, #100
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES #81
    )

    return box_selector





















    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        # num_images = len(boxlists)
        num_images = self.tik_instance.Scalar(dtype="int32", name="index")
        num_images.set_as(len(boxlists))

        # results = []
        self.results_ub = self.tik_instance.Tensor(self.dtype,                         #建立boxs盒子数据打包容器
                      (num_images + 16, ),
                      name="results_ub", scope=tik.scope_gm)     
        # for i in range(num_images):
        with self.tik_instance.for_range(0,num_images + 16) as i:
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            
            scores = self.tik_instance.Scalar(dtype="int32", name="scores")
            scores.set_as(boxlists[i].index("scores"))
            labels = self.tik_instance.Scalar(dtype="int32", name="labels")
            labels.set_as(boxlists[i].index("labels"))        
            boxes = self.tik_instance.Scalar(dtype="int32", name="index")
            boxes.set_as(len(boxlists))        
            boxlist = self.tik_instance.Scalar(dtype="int32", name="index")
            boxlist.set_as(len(boxlists))
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

