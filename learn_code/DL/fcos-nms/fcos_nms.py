from te import tik
import numpy as np

def get_shape_size(shape):
    return shape[0] * shape[1] * shape[2] * shape[3]

class Fcos_nms ():
    def _init_(self, cls_tower, logits, centerness, bbox_reg) :
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.cls_tower = cls_tower
        self.logits = logits
        self.centerness = centerness
        self.bbox_reg = bbox_reg
        self.fpn_strides = [8,16,32,64,128]
        #分配内存
        self.alloc_gm() 

        self.multi_class_nms_obj = multi_class_nms.MultiClassNms(
             self.tik_instance, self.merge_gm, self.dtype,
             self.all_element_align_16 * 16, self.num_class, self.width,
             self.height, self.max_output_size, self.iou_threshold,
             self.topk, self.cls_out_num, self.down_filter)

    #定义缓存区tensor         
    def gen_ub(self, data_type, length, ub_name):
        return self.tik_instance.Tensor(data_type, (length, ), name=ub_name,
                                        scope=tik.scope_ubuf)
    
    #定义内存中数据，输入输出
    def alloc_gm(self):
        #input tensor
        self.box_gm = self.tik_instance.Tensor(self.dtype, 
                      (get_shape_size(self.box_shape) + 16, ),
                      name="box_gm", scope=tik.scope_gm)
        self.logits_gm = self.tik_instance.Tensor(self.dtype, 
                        (get_shape_size(self.logits_shape) + 16, ),
                        name="score_gm", scope=tik.scope_gm)
        self.centerness_gm = self.tik_instance.Tensor(self.dtype,
                         (get_shape_size(self.centerness_shape) + 16, ),
                         name="centerness_gm", scope=tik.scope_gm)

        #output tensor
        self.output_gm = self.tik_instance.Tensor(self.dtype, 
                         (self.max_output_size * 8, ), 
                         name="output_gm", scope=tik.scope_gm)

    #坐标筛选，移除不在图片内的坐标
    def coord_clip(self, coord_ub, coord_min, coord_max,
                   loop_length_align_128):
        with self.tik_instance.for_range(0, loop_length_align_128) as idx:
            cmpack = self.tik_instance.vcmp_ge(128, coord_ub[idx * 128], 
                     coord_min, 1, 1)  #比较，大于或等于
            self.tik_instance.vsel(128, 0, coord_ub[idx * 128], cmpack,
                                   coord_ub[idx * 128], coord_min,
                                   1, 1, 1, 1, 8, 8, 8)  #根据sel值，比较，迭代
            cmpack1 = self.tik_instance.vcmp_ge(128, coord_max,
                                                coord_ub[idx * 128], 1, 1)
            self.tik_instance.vsel(128, 0, coord_ub[idx * 128], cmpack1,
                                   coord_ub[idx * 128], coord_max,
                                   1, 1, 1, 1, 8, 8, 8)


    def compute_locations(self, features):
            tik_instance = tik.Tik()
                locations = self.gen_ub(self.dtype, self.locations, "locations")
                for level, feature in enumerate(features):
                    h, w = feature.size()[-2:]
                    locations_per_level = self.compute_locations_per_level(
                        h, w, self.fpn_strides[level]
                    )
                    locations.append(locations_per_level)
                return locations
    def compute_locations_per_level(self, h, w, stride):
        shifts_x = np.arange(
            0, w * stride, step=stride,
            dtype=float16
        )
        shifts_y = np.arange(
            0, h * stride, step=stride,
            dtype=float16
        )
        shift_y, shift_x = np.meshgrid(shifts_x,shifts_y)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = np.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

                               
    def compile(self):
        self.process_coord_score(self.box_gm, self.logits_gm, 
                                 self.centerness_gm)
        #do nms
        ret = self.multi_class_nms_obj.do_efficientdet_nms()
        #数据转移到内存中
        self.tik_instance.data_move(self.output_gm, ret, 0, 1,
                                    self.max_output_size * 8 // 16, 0, 0, 0)
        #build编译  名称、输入、输出
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=(
                                   self.box_gm, self.logits_gm, self.centerness_gm,
                                   ), outputs=(self.output_gm,))
        return self.tik_instance

def fcos_nms(bbox_reg, logits, centerness, score_threshold, iou_threshold, topk,kernel_name="fcos_nms"):
    """
    the compile function of efficientdet nms
    """
    box_shape = bbox_reg.get('shape')
    logits_shape = logits.get('shape')
    centerness_shape = centerness.get('shape')
    num_class = logits_shape[1]
    param_check(box_shape, logits_shape, anchor_shape, merge_shape, width,
                height, num_class, score_threshold, iou_threshold, topk,
                max_output_size)

    fcos_obj = Fcos_nms(box_shape, logits_shape,centerness_shape,
                       num_class,
                       score_threshold, iou_threshold, topk, 
                       kernel_name)
    tik_instance = fcos_obj.compile()
    return tik_instance