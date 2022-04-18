import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(           #传参
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()       #继承
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

    def forward_for_single_feature_map(                     #对每个fpn level的结果进行后处理
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:                                          #N代表数量， C代表channel，H代表高度，W代表宽度.
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W          #对应分别是图片张数
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)      
        box_cls = box_cls.reshape(N, -1, C).sigmoid()  #预测的分类信息
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)  #预测的回归信息
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()  #预测的中心度数据

        candidate_inds = box_cls > self.pre_nms_thresh #0.05
            #candudate_inds(0/1)代表各频道像素点（NxH*WxC）是否为正采样
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)   
            #展开为n
            #pre_nms_top_n代表神经网络算的各频道正采样个数
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n) #限定在1000以内

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None] 

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]        #赋值类别得分
            per_candidate_inds = candidate_inds[i]     #赋值正采样像素点个数
            per_box_cls = per_box_cls[per_candidate_inds]   #满足条件的分类得分

            per_candidate_nonzeros = per_candidate_inds.nonzero()   #.nonzero()返回不为o的元素引导
            per_box_loc = per_candidate_nonzeros[:, 0]  #位置
            per_class = per_candidate_nonzeros[:, 1] + 1    #类别，背景为0故加一

            per_box_regression = box_regression[i]  #赋值回归距离值，正采样数据
            per_box_regression = per_box_regression[per_box_loc]  #取出满足条件正采样数据
            per_locations = locations[per_box_loc]  #取出位置信息

            per_pre_nms_top_n = pre_nms_top_n[i]
                
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                #判断是否满足1000阈值上限
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                        #获得top n 个类别得分指引，不按顺序导出
                per_class = per_class[top_k_indices]    #导出的topk类别
                per_box_regression = per_box_regression[top_k_indices] #回归
                per_locations = per_locations[top_k_indices]    #位置
                
                #计算topk位置信息位置信息
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
                        #得到删减空白边框的列表
                        #BoxList调用自fcos_core.structures.bounding_box
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
                        #裁剪边框
                        #remove_small_boxes调用自fcos_core.structures.boxlist_ops
            results.append(boxlist)
        #将正采样边界框的回归结果存入类BoxList，
        # 包括边界框左上、右下坐标，图的大小，边界框对应的类别，
        # 最终分数(分类分数×中心度，开根)，经过裁剪过滤后将所有边界框作为结果返回。
        return results

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
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]   #cat_boxlist:将BoxList列表(具有相同图像大小)连接成单个BoxList
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
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
