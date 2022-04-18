import numpy as np
from te import tik


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(           #传参
        self,
        pre_nms_thresh,
        max_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        features,
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
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
        """

        self.tik_instance = tik.Tik(tik.Dprofile())# 初始化tik容器

        super(FCOSPostProcessor, self).__init__()       #继承
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.features = features

        
        # 定义标量tik接口 
        self.pre_nms_thresh = self.tik _instance.Scalar(" float16")
        self.pre_nms_top_n = self.tik_instance.Scalar(" int16 ")
        self.nms_thresh = self.tik_isntance.Scalar(" float16")
        self.fpn_post_nms_top_n=self.tik_instance.Scalar("int16")
        self.min_size = self.tik_instance.Scalar("int16")
        self.num_classes = self.tik_instance.Scalar("int16")

    def gen_ub(self, data_type, length, ub_name):
        return self.tik_instance.Tensor(data_type, (length, ), name=ub_name,
                                        scope=tik.scope_ubuf)
#计算每个feature上每个像素点对应原图坐标
    # def compute_locations(self, features):
    #         tik_instance = tik.Tik()
    #             # locations = self.gen_ub(self.dtype, self.locations, "locations")

    #             # for level, feature in enumerate(features):
    #             features_enum = self.gen_ub(self.dtype, self.features, "features_enum")
    #             self.tik_instance.data_move(features_enum,list(enumerate(features)), 0, 1, 32, 0, 0)
    #             level = self.tik_instance.Scalar(dtype="int32", name="level")                
    #             with self.tik_instance.for_range(0,features_enum[:,0]) as level:        #((0,*),(1,*),(2,*)....)

    #                 # h, w = feature.size()[-2:]
    #                 h,w = features_enum[:,1].size()[-2:]

    #                 # locations_per_level = self.compute_locations_per_level(
    #                 #     h, w, self.fpn_strides[level]
    #                 # )
    #             self.locations_per_level = self.tik_instance.Tensor(self.dtype,locations_shape, 
    #                     name="locations_per_level", scope=tik.scope_ub)                    
    #             self.tik_instance.data_move(locations_per_level,self.compute_locations_per_level(
    #                     h, w, self.fpn_strides[level]), 0, 1, 32, 0, 0)   
    #                 # locations.append(locations_per_level)                     
    #             self.tik_instance.data_move(locations,locations_per_level, 0, 1, 32, 0, 0)             
    #             return locations
    def topk(matrix, K, axis=1):
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis])
            topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
            topk_data = matrix[topk_index, row_index]
            topk_index_sort = np.argsort(-topk_data,axis=axis)
            topk_data_sort = topk_data[topk_index_sort,row_index]
            topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
        else:
            column_index = np.arange(matrix.shape[1 - axis])[:, None]
            topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
            topk_data = matrix[column_index, topk_index]
            topk_index_sort = np.argsort(-topk_data, axis=axis)
            topk_data_sort = topk_data[column_index, topk_index_sort]
            topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
        return topk_data_sort, topk_index_sort
#计算特征图对应原图坐标            
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
  
    def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))

    def forward_for_single_feature_map(                     #对FPN金字塔的每一层（P3，P4,P5,P6,P7）的输出结果进行后处理
            self, stride, box_cls,
            box_regression, centerness,
            image_sizes):

        locations_shape = locations.get("shape")
        box_cls_shape = box_cls.get("shape")
        box_regression_shape = box_regression.get("shape")
        centerness_shape = centerness.get("shape")
        image_sizes_shape = image_sizes.get("shape")
#占位
        self.locations = self.tik_instance.Tensor("float16",locations_shape,scope = scope_ubuf,
            name="locations",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
        self.box_cls =self.tik_instance.Tensor("float16",box_cls_shape, scope=scope_ubuf, 
            name = "box_cls",enable_buffer_reuse= False, no_reuse_list  = None)
        self.box_regression =self.tik_instance.Tensor("float16",box_regression_shape, scope=scope_ubuf, 
            name = "box_regression",enable_buffer_reuse= False, no_reuse_list  = None)
        self.centerness=self.tik_instance.Tensor("float16",centerness_shape, scope=scope_ubuf, 
            name = "centerness",enable_buffer_reuse= False, no_reuse_list  = None)
        self.image_sizes = self.tik_instance.Tensor("float16",image_sizes_shape, scope=scope_ubuf, 
            name = "image_sizes",enable_buffer_reuse= False, no_reuse_list  = None)

        self.candidate_inds = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
            name="candidate_inds",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
         
        self.pre_nms_top_n = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="pre_nms_top_n",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
    

        self.per_box_cls = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_box_cls",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
        self.per_candidate_inds = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_candidate_inds",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
    

        self.per_candidate_nonzeros = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_candidate_nonzeros",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
        self.per_box_loc = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_box_loc",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
        self.per_class = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_class",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)


        self.per_box_regression = self.tik_instance.Tensor("float16",box_regression_shape,scope = scope_ubuf,
                name="per_box_regression",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
        self.per_locations = self.tik_instance.Tensor("float16",locations_shape,scope = scope_ubuf,
                name="per_locations",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
        self.per_pre_nms_top_n = self.tik_instance.Tensor("float16",pre_nms_top_n_shape ,scope = scope_ubuf,
                name="per_pre_nms_top_n",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
            
        self.per_box_cls_topk = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_box_cls_topk",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
                
        self.top_k_indices = self.tik_instance.Tensor(self.dtype,(len(per_box_cls_topk) , ), 
                name="top_k_indices", scope=tik.scope_gm)
  
        self.detections = self.tik_instance.Tensor(self.dtype,(len(per_locations) ,4 ), 
                name="detections", scope=tik.scope_gm)
# shape
        N, C, H, W = box_cls.get("shape")
        
        locations = self.compute_locations_per_level(H, W, stride)

        '''
        将预测结果的shape进行处理以方便与locations进行对齐
        locations的shape为(H*W,2) 其中每个是(x,y)形式
        代表该特征层每个点在输入图像的位置
        '''
        box_cls = box_cls.reshape(N,C,H,W).transpose(0,2,3,1)  #先将张量展开成一维再将tensor的维度换位 NCHW---> 0123 

        box_cls = box_cls.reshape(N, -1, C).sigmoid()        # sigmod预测分类  (N,H*W,C) 将logits经过sigmoid函数做多个二分类
    
        box_regression = box_regression.reshape(N,4,H,W).transpose(0,2,3,1)      #将其展成一维，并对其维度进行换位
        
        box_regression = box_regression.reshape(N, -1, 4)   # 预测回归边框 重新调整其shape

        centerness = centerness.reshape(N,1,H,W).transpose(0,2,3,1)        # 将其展成一维，并将其维度进行换位

        centerness = centerness.reshape(N, -1).sigmoid()        #预测的中心度数据

        candidate_inds = box_cls > self.pre_nms_thresh        #  经过筛选后每张图片保留下来的候选目标数量

        pre_nms_top_n = candidate_inds.reshape(N,-1)
        pre_nms_top_n = np.sum( pre_nms_top_n , axis  = 0,keepdims = False)
        
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)  #torch.clamp()将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。

        box_cls = box_cls * centerness[:, :, None] 

        detections = []
        with self.tik_instance.for_range(0,N) as i:         # for i in range(N):
            per_box_cls = box_cls[i]        #赋值类别得分
            per_candidate_inds = candidate_inds[i]     #赋值正采样像素点个数
            per_box_cls = per_box_cls[per_candidate_inds]   #满足条件的分类得分

            per_candidate_nonzeros = per_candidate_inds.nonzero()   #.nonzero()返回数组中元素不为0 的位置
            per_box_loc = per_candidate_nonzeros[:, 0]  #位置
            per_class = per_candidate_nonzeros[:, 1] + 1    #类别，背景为0故加一
            
            per_box_regression = box_regression[i]  #赋值回归距离值，正采样数据
            per_box_regression = per_box_regression[per_box_loc]  #取出满足条件正采样数据
            per_locations = locations[per_box_loc]  #取出位置信息
            per_pre_nms_top_n = pre_nms_top_n[i]
              

            # if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
            with self.tik_instance.if_scope(np.sum(per_candidate_inds) > np.sum(per_pre_nms_top_n)):
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)                
 
                per_class = per_class[top_k_indices]    #导出的topk类别
                per_box_regression = per_box_regression[top_k_indices] #回归
                per_locations = per_locations[top_k_indices]    #位置
            
            #计算topk位置信息
            detections = np.stack([
                # x1 = x -l
                per_locations[:, 0] - per_box_regression[:, 0],
                # y1= y-t
                per_locations[:, 1] - per_box_regression[:, 1],
                # x2  = x+r
                per_locations[:, 0] + per_box_regression[:, 2],
                # y2 = y+b
                per_locations[:, 1] + per_box_regression[:, 3],
            ],axis = 0)         #axis = 0表示增加第一维度         

        return detections, per_class, per_box_cls
   
    def forward( ):
        
