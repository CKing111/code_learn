import numpy as np
from .bounding_box import BoxList

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
######################################################################################################################

        self.tik_instance = tik.Tik(tik.Dprofile())# 初始化tik容器

        super(FCOSPostProcessor, self).__init__()       #继承
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

        
        # 定义标量tik接口 
        self.pre_nms_thresh = self.tik _instance.Scalar(" float16")
        self.pre_nms_top_n = self.tik_instance.Scalar(" int16 ")
        self.nms_thresh = self.tik_isntance.Scalar(" float16")
        self.fpn_post_nms_top_n=self.tik_instance.Scalar("int16")
        self.min_size = self.tik_instance.Scalar("int16")
        self.num_classes = self.tik_instance.Scalar("int16")

    def sigmoid(self,x):
            return 1 / (1 + np.exp(-x))
#####################################################################################################################
    def forward_for_single_feature_map(                     #对FPN金字塔的每一层（P3，P4,P5,P6,P7）的输出结果进行后处理
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):

        """
        Arguments:                                          #N代表数量， C代表channel，H代表高度，W代表宽度.
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W          #对应分别是图片张数
            box_regression: tensor of size N, A * 4, H, W


        在进行NMS前处理一个batch中单个特征层所有点对应的预测结果，
        得到整个batch在该层中的预测结果，其中bbox坐标对应到输入图像空间
        """
        

        locations_shape = locations.get("shape")
        box_cls_shape = box_cls.get("shape")
        box_regression_shape = box_regression.get("shape")
        centerness_shape = centerness.get("shape")
        image_sizes_shape = image_sizes.get("shape")


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



        # shape
        N, C, H, W = box_cls.get("shape")
        
        '''
        将预测结果的shape进行处理以方便与locations进行对齐
        locations的shape为(H*W,2) 其中每个是(x,y)形式
        代表该特征层每个点在输入图像的位置
        '''

    
        # 这里需要知道tik中的几个接口：
        # view()将张量展开为一维，,permute()将张量的维度换位
        # put in the same format as locations
        tik_instance = tik.Tik()  # tik容器
        #box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1) #先将张量展开成一维再将tensor的维度换位

        # 展开成一维
        # box_cls = box_cls.reshape(N,C,H,W)
        self.tik_instance.data_move(box_cls, box_cls.reshape(N,C,H,W), 0, 1, 32, 0, 0)

        #维度换位  NCHW---> 0123
        # box_cls = box_cls.transpose(0,2,3,1)
        self.tik_instance.data_move(box_cls, box_cls.transpose(0,2,3,1), 0, 1, 32, 0, 0)

        
        # sigmod预测分类  (N,H*W,C) 将logits经过sigmoid函数做多个二分类
        # box_cls = box_cls.reshape(N, -1, C).sigmoid()  
        self.tik_instance.data_move(box_cls,box_cls.reshape(N, -1, C).sigmoid(), 0, 1, 32, 0, 0)

        #box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1) #将其展成一维，并对其维度进行换位
        box_regression = box_regression.reshape(N,4,H,W)
        box_regression = box_regression.transpose(0,2,3,1)


        box_regression = box_regression.reshape(N, -1, 4)  # 预测回归边框 重新调整其shape

        #centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1) # 将其展成一维，并将其维度进行换位
        centerness = centerness.reshape(N,1,H,W)
        centerness = centerness.transpose(0,2,3,1)

        # sigmod 预测中心度
        centerness = centerness.reshape(N, -1).sigmoid()  #预测的中心度数据
    
        self.candidate_inds = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="candidate_inds",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
    
        # candidate_inds = box_cls > self.pre_nms_thresh  # 0.05    经过筛选后每张图片保留下来的候选目标数量
        #candudate_inds(0/1)代表各频道像素点（NxH*WxC）是否为正采样
        self.tik_instance.data_move(candidate_inds, box_cls > self.pre_nms_thresh, 0, 1, 32, 0, 0)




        #pre_nms_top_n = candidate_inds.view(N, -1).sum(1)   
        
        # (N,H*W,C)->(N,H*W*C)->(N,) bool->int
        # pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # 亲测以上那句会由于内存不连续而报错 于是我这里改成了以下
        # 这里reshape()等价于contigous().view()
       # pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)

        self.pre_nms_top_n = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="pre_nms_top_n",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
    
        # pre_nms_top_n = candidate_inds.reshape(N,-1) 
        # pre_nms_top_n = np.sum( pre_nms_top_n , axis  = 0,keepdims = False)

        self.tik_instance.data_move(pre_nms_top_n, candidate_inds.reshape(N,-1), 0, 1, 32, 0, 0)
        self.tik_instance.data_move(pre_nms_top_n, np.sum( pre_nms_top_n , axis  = 0,keepdims = False), 0, 1, 32, 0, 0)

        # pre_nms_top_n_shape = pre_nms_top_n.get("shape")
        #展开为n
        #pre_nms_top_n代表神经网络算的各频道正采样个数

        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n) #限定在1000以内
        #torch.clamp()将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。



        
        # 分类得分 用中心得分
        # box_cls = box_cls * centerness[:, :, None] 
        self.tik_instance.data_move(box_cls,box_cls * centerness[:, :, None], 0, 1, 32, 0, 0)


        i = self.tik_instance.Scalar(dtype="int32", name="idx")

        results = []
       with self.tik_instance.for_range(0,N) as i:
        # for i in range(N):
            # per_box_cls = box_cls[i]        #赋值类别得分
            # per_candidate_inds = candidate_inds[i]     #赋值正采样像素点个数
            # per_box_cls = per_box_cls[per_candidate_inds]   #满足条件的分类得分
            self.per_box_cls = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_box_cls",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
            self.per_candidate_inds = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_candidate_inds",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
    
            self.tik_instance.data_move(per_box_cls, box_cls[i], 0, 1, 32, 0, 0)
            self.tik_instance.data_move(per_candidate_inds, candidate_inds[i], 0, 1, 32, 0, 0)
            self.tik_instance.data_move(per_box_cls, per_box_cls[per_candidate_inds], 0, 1, 32, 0, 0)                        
 
            '''
            numpy.nonzeros(a)返回数组a中值不为零的元素的下标，
            它的返回值是一个长度为a.ndim(数组a的轴数)的元组，
            元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值
            '''
            self.per_candidate_nonzeros = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_candidate_nonzeros",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
            self.per_box_loc = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_box_loc",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
            self.per_class = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                name="per_class",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
                                                                              
            # per_candidate_nonzeros = per_candidate_inds.nonzero()   #.nonzero()返回数组中元素不为0 的位置
            # per_box_loc = per_candidate_nonzeros[:, 0]  #位置
            # per_class = per_candidate_nonzeros[:, 1] + 1    #类别，背景为0故加一

            self.tik_instance.data_move(per_candidate_nonzeros, per_candidate_inds.nonzero(), 0, 1, 32, 0, 0)
            self.tik_instance.data_move(per_box_loc, per_candidate_nonzeros[:, 0], 0, 1, 32, 0, 0)
            self.tik_instance.data_move(per_class, per_candidate_nonzeros[:, 1] + 1, 0, 1, 32, 0, 0)                        

            self.per_box_regression = self.tik_instance.Tensor("float16",box_regression_shape,scope = scope_ubuf,
                name="per_box_regression",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
            self.per_locations = self.tik_instance.Tensor("float16",locations_shape,scope = scope_ubuf,
                name="per_locations",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
            self.per_pre_nms_top_n = self.tik_instance.Tensor("float16",pre_nms_top_n_shape ,scope = scope_ubuf,
                name="per_pre_nms_top_n",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
            
            # per_box_regression = box_regression[i]  #赋值回归距离值，正采样数据
            # per_box_regression = per_box_regression[per_box_loc]  #取出满足条件正采样数据
            # per_locations = locations[per_box_loc]  #取出位置信息
            # per_pre_nms_top_n = pre_nms_top_n[i]
                
            self.tik_instance.data_move(per_box_regression, box_regression[i], 0, 1, 32, 0, 0)
            self.tik_instance.data_move(per_box_regression, per_box_regression[per_box_loc], 0, 1, 32, 0, 0)
            self.tik_instance.data_move(per_locations, locations[per_box_loc], 0, 1, 32, 0, 0)                        
            self.tik_instance.data_move(per_pre_nms_top_n, pre_nms_top_n[i], 0, 1, 32, 0, 0)                        

                '''
                torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) 
                求tensor中某个dim的前k大或者前k小的值以及对应的index。

                torch.item()是得到一个张量里面的元素值
                '''

                '''
                vec_reduce_add(mask, dst, src, work_tensor, repeat_times, src_rep_stride)
                dst输出， src输入
                '''

           
            # # if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
            # if_scope ( np.sum(per_candidate_inds) > np.sum(per_pre_nms_top_n)):
            #     #判断是否满足1000阈值上限
            #     per_box_cls, top_k_indices = \
            #         per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            #             #获得top n 个类别得分指引，不按顺序导出


            with self.tik_instance.if_scope(np.sum(per_candidate_inds) > np.sum(per_pre_nms_top_n)):

                self.per_box_cls_topk = self.tik_instance.Tensor("float16",box_cls_shape,scope = scope_ubuf,
                    name="per_box_cls_topk",enable_buffer_reuse= False,no_reuse_list=None, is_atomic_add=False)
                # self.tik_instance.data_move(per_box_cls_topk, per_box_cls.topk(per_pre_nms_top_n, sorted=False), 0, 1, 32, 0, 0)                        
            
                self.top_k_indices = self.tik_instance.Tensor(self.dtype,                         
                      (len(per_box_cls_topk) , ), name="top_k_indices", scope=tik.scope_gm)
                self.tik_instance.data_move((per_box_cls_topk,top_k_indices), per_box_cls.topk(per_pre_nms_top_n, sorted=False), 0, 1, 32, 0, 0)                        

                # per_box_cls, top_k_indices = \
                #     per_box_cls.topk(per_pre_nms_top_n, sorted=False)

                self.tik_instance.data_move(per_class, per_class[top_k_indices], 0, 1, 32, 0, 0)                        
                self.tik_instance.data_move(per_box_regression, per_box_regression[top_k_indices], 0, 1, 32, 0, 0)                        
                self.tik_instance.data_move(per_locations, per_locations[top_k_indices], 0, 1, 32, 0, 0)                        

                # per_class = per_class[top_k_indices]    #导出的topk类别
                # per_box_regression = per_box_regression[top_k_indices] #回归
                # per_locations = per_locations[top_k_indices]    #位置




                
                '''torch.stack() 将序列连接，形成一个新的tensor结构，此结构中会增加一个维度。连接中的 每个tensor都要保持相同的大小。
                '''
            #计算topk位置信息
            # detections = torch.stack([
            #     per_locations[:, 0] - per_box_regression[:, 0],
            #     per_locations[:, 1] - per_box_regression[:, 1],
            #     per_locations[:, 0] + per_box_regression[:, 2],
            #     per_locations[:, 1] + per_box_regression[:, 3],
            # ], dim=1)

            detections =np.stack([
                # x1 = x -l
                per_locations[:, 0] - per_box_regression[:, 0],
                # y1= y-t
                per_locations[:, 1] - per_box_regression[:, 1],
                # x2  = x+r
                per_locations[:, 0] + per_box_regression[:, 2],
                # y2 = y+b
                per_locations[:, 1] + per_box_regression[:, 3],
            ],axis = 0)  #axis = 0表示增加第一维度


            h, w = image_sizes[i]  # 输入图像的高、宽
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy") # 实例化BoxList()对
                        #得到删减空白边框的列表
                        #BoxList调用自fcos_core.structures.bounding_box
            boxlist.add_field("labels", per_class) # 预测类别
            # boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist.add_field("score", per_box_cls *per_box_cls) # 预测分数
            boxlist = boxlist.clip_to_image(remove_empty=False)    # 将bbox坐标限制在输入图像尺寸范围内 remove_empty=False代表不对bbox的坐标作检查(即不要求x2>x1&y2>y1)
           
    #   in bounding_box.py
    # def clip_to_image(self, remove_empty=True):
    #     TO_REMOVE = 1
    #     self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
    #     self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
    #     self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
    #     self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
    #     if remove_empty:
    #         box = self.bbox
    #         keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
    #         return self[keep]
    #     return self



            # 过滤掉尺寸较小的bbox 实际的实现是保留边长不小于self.min_size的那批
            # self.min_size默认为0 于是在这个条件下 就相当于对以上那句设置了remove_empty=True
            boxlist = remove_small_boxes(boxlist, self.min_size)

#in boxlist_ops.py
# def remove_small_boxes(boxlist, min_size):
#     """
#     Only keep boxes with both sides >= min_size

#     Arguments:
#         boxlist (Boxlist)
#         min_size (int)
#     """
#     # TODO maybe add an API for querying the ws / hs
#     xywh_boxes = boxlist.convert("xywh").bbox
#     _, _, ws, hs = xywh_boxes.unbind(dim=1)
#     keep = (
#         (ws >= min_size) & (hs >= min_size)
#     ).nonzero().squeeze(1)
#     return boxlist[keep]


            #裁剪边框
            #remove_small_boxes调用自fcos_core.structures.boxlist_ops
            results.append(boxlist)
        #将正采样边界框的回归结果存入类BoxList，
        # 包括边界框左上、右下坐标，图的大小，边界框对应的类别，
        # 最终分数(分类分数×中心度，开根)，经过裁剪过滤后将所有边界框作为结果返回。
        return results
