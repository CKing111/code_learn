from te import tik
import numpy as np
# import torch

from .bounding_box import BoxList

from fcos_core.layers import nms as _box_nms
from fcos_core.layers import ml_nms as _box_ml_nms


# TODO redundant, remove
#Tensors:可以是任意相同Tensor 类型的python 序列
# dim:沿着此维连接张量序列。
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # assert isinstance(tensors, (list, tuple))
    # if len(tensors) == 1:
    #     return tensors[0]
    # return torch.cat(tensors, dim)

    with self.tik_instance.if_scope(isinstance(tensors, (list, tuple)):
        with self.tik_instance.if_scope(len(tensors) == 1):
            return tensors[0]
        with tik_instance.elif_scope(len(tensors) > 1):
            return np.concatenate(tensors, dim)

def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
