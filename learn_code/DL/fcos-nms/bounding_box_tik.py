from te import tik
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class BoxList(object):
    def __init__(self, bbox, image_size, mode="xyxy"):
        # device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        self.tik_instance = tik.Tik(tik.Dprofile()) #通过传入tik.Dprofile实例，创建TIK DSL容器。
        # bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        # bbox=np.array(bbox)
        self.bbox = self.tik_instance.Tensor(self.float32,                         #
                      (len(bbox) + 16, ),
                      name="bbox", scope=tik.scope_gm)   
        # if bbox.ndimension() != 2:
        #     raise ValueError(
        #         "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
        #     )
        # if bbox.size(-1) != 4:
        #     raise ValueError(
        #         "last dimension of bbox should have a "
        #         "size of 4, got {}".format(bbox.size(-1))
        #     )
        # if mode not in ("xyxy", "xywh"):
        #     raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}


    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        k = self.tik_instance.Scalar(dtype="int32", name="k")
        v = self.tik_instance.Scalar(dtype="int32", name="v")
        idx = self.tik_instance.Scalar(dtype="int32", name="idx")

        # for k, v in bbox.extra_fields.items():
        with self.tik_instance.for_range(0,len(zip(locations, box_cls, box_regression, centerness)) + 16) as idx:
            # self.extra_fields[k] = v
            v.set_as(self.extra_fields[k] )


    #转换
    def convert(self, mode):

        # if mode not in ("xyxy", "xywh"):
        #     raise ValueError("mode should be 'xyxy' or 'xywh'")
        with self.tik_instance.if_scope(mode  in ("xyxy", "xywh")and mode == self.mode):
            # if mode == self.mode:
            # with self.tik_instance.if_scope(mode == self.mode):  
            #     return self         #链式调用

            #提取坐标范围
            # xmin, ymin, xmax, ymax = self._split_into_xyxy()
            self.xmin = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="xmin", scope=tik.scope_gm)   
            self.ymin = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="ymin", scope=tik.scope_gm)
            self.xmax = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="xmax", scope=tik.scope_gm)   
            self.ymax = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="ymax", scope=tik.scope_gm)   
            self.tik_instance.data_move((xmin, ymin, xmax, ymax), self._split_into_xyxy(), 0, 1, 32, 0, 0)

            # if mode == "xyxy":
            with self.tik_instance.if_scope(mode == "xyxy"):
                # self.tik_instance.data_move(bbox, np.concatenate((xmin, ymin, xmax, ymax), dim=-1),
                #                                                                  0, 1, 32, 0, 0)
                bbox = np.concatenate((xmin, ymin, xmax, ymax), dim=-1)
                # bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
                bbox = BoxList(bbox, self.size, mode=mode)
            # else:
            with tik_instance.else_scope():
                TO_REMOVE = 1
                # bbox = torch.cat(
                #     (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
                # )
                bbox = np.concatenate( (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE),
                                                                 dim=-1)
                # bbox = BoxList(bbox, self.size, mode=mode)
                self.tik_instance.data_move(bbox, BoxList(bbox, self.size, mode=mode), 0, 1, 32, 0, 0)
            bbox._copy_extra_fields(self)
        return bbox


    def _split_into_xyxy(self):
        # if self.mode == "xyxy":
        with self.tik_instance.if_scope(self.mode == "xyxy"):
            self.xmin = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="xmin", scope=tik.scope_gm)   
            self.ymin = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="ymin", scope=tik.scope_gm)
            self.xmax = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="xmax", scope=tik.scope_gm)   
            self.ymax = self.tik_instance.Tensor(self.float32,                         #
                        (len(bbox) + 16, ),
                      name="ymax", scope=tik.scope_gm)   
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        # elif self.mode == "xywh":
        with tik_instance.else_scope(self.mode == "xywh"):

            TO_REMOVE = 1   
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")