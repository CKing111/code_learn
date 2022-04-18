import numpy as np
from te import tik


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(self,box_shape,score_shape, centerness_shape,
        pre_nms_thresh, max_top_n, nms_thresh, num_classes,
        features,):

        self.tik_instance = tik.Tik(tik.Dprofile())# 初始化tik容器

        super(FCOSPostProcessor, self).__init__()       #继承
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.features = features
        self.slice_len = ??????

        self.sort_proposal_obj = sort_proposal.SortProposal(
                                 self.tik_instance, self.input_gm, self.dtype,
                                 self.all_element, self.topk) 
        self.alloc_gm
        
    #分配全局缓冲区                  
    def alloc_gm(self):
        #input tensor
        self.box_gm = self.tik_instance.Tensor(self.dtype, 
                      (get_shape_size(self.box_shape) + 16, ),
                      name="box_gm", scope=tik.scope_gm)
        self.score_gm = self.tik_instance.Tensor(self.dtype, 
                        (get_shape_size(self.score_shape) + 16, ),
                        name="score_gm", scope=tik.scope_gm)
        self.centerness_gm = self.tik_instance.Tensor(self.dtype,
                         (get_shape_size(self.anchor_shape) + 16, ),
                         name="centerness_gm", scope=tik.scope_gm)
        #output tensor
        self.output_gm = self.tik_instance.Tensor(self.dtype, 
                         (self.max_output_size * 8, ), 
                         name="output_gm", scope=tik.scope_gm)

    def gen_ub(self, data_type, length, ub_name):
        return self.tik_instance.Tensor(data_type, (length, ), name=ub_name,
                                        scope=tik.scope_ubuf)
        #计算大于或等于value/factor的最小整数值
    def ceil_div_offline(self, value, factor):
        result = (value + (factor - 1)) // factor   #取整除 - 返回商的整数部分（向下取整）eg
        return result

    def topk(matrix, K, axis=1):
        matrix = self.gen_ub(self.dtype, self.slice_len, "matrix")
        matrix = self.gen_ub(self.dtype, self.slice_len, "matrix")
        matrix = self.gen_ub(self.dtype, self.slice_len, "matrix")
                
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
#计算单层特征图对应原图坐标            
    def compute_locations_per_level(self, h, w, stride):
    
        h = self.tik_instance.Scalar(dtype="int32", name="h")
        w = self.tik_instance.Scalar(dtype="int32", name="w")
        stride = self.tik_instance.Scalar(dtype="int32", name="stride")
        locations_ub = self.gen_ub(self.dtype, self.slice_len, "locations_ub")
        shifts_x = self.gen_ub(self.dtype, self.slice_len, "shifts_x")
        shifts_y = self.gen_ub(self.dtype, self.slice_len, "shifts_y")

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
        locations_ub = np.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations_ub
  
    def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))

    def forward_for_single_feature_map(                     #对FPN金字塔的每一层（P3，P4,P5,P6,P7）的输出结果进行后处理
            self, stride, box_cls,
            box_regression, centerness):

        box_cls_ub = self.gen_ub(self.dtype, self.slice_len, "box_cls_ub")
        box_regression_ub = self.gen_ub(self.dtype, self.slice_len, "box_regression_ub")
        centerness_ub = self.gen_ub(self.dtype, self.slice_len, "centerness_ub")
        loop_length_align_16 = self.ceil_div_offline(loop_length, 16)      
        
        self.tik_instance.data_move(box_cls_ub, box_cls, 0, 1, loop_length_align_16, 1, 1)
        self.tik_instance.data_move(box_regression_ub, box_regression, 0, 1, loop_length_align_16, 1, 1)
        self.tik_instance.data_move(centerness_ub, centerness, 0, 1, loop_length_align_16, 1, 1)

        N, C, H, W = box_cls.get("shape")
        
        locations_ub = self.compute_locations_per_level(H, W, stride)

        '''
        将预测结果的shape进行处理以方便与locations进行对齐
        locations的shape为(H*W,2) 其中每个是(x,y)形式
        代表该特征层每个点在输入图像的位置
        '''
        box_cls_ub = box_cls_ub.reshape(N,C,H,W).transpose(0,2,3,1)  #先将张量展开成一维再将tensor的维度换位 NCHW---> 0123 

        box_cls_ub = box_cls_ub.reshape(N, -1, C).sigmoid()        # sigmod预测分类  (N,H*W,C) 将logits经过sigmoid函数做多个二分类
    
        box_regression = box_regression.reshape(N,4,H,W).transpose(0,2,3,1)      #将其展成一维，并对其维度进行换位
        
        box_regression = box_regression.reshape(N, -1, 4)   # 预测回归边框 重新调整其shape

        centerness = centerness.reshape(N,1,H,W).transpose(0,2,3,1)        # 将其展成一维，并将其维度进行换位

        centerness = centerness.reshape(N, -1).sigmoid()        #预测的中心度数据

        candidate_inds = box_cls > self.pre_nms_thresh        #  经过筛选后每张图片保留下来的候选目标数量

        pre_nms_top_n = candidate_inds.reshape(N,-1)
        pre_nms_top_n = np.sum( pre_nms_top_n , axis  = 0,keepdims = False)
        
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.max_top_n)  #torch.clamp()将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。

        box_cls = box_cls * centerness[:, :, None] 

        detections = []
        with self.tik_instance.for_range(0,N) as i:   # for i in range(N):
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
        
