# -*- coding:utf-8 -*-
from te import tik
from util import OpLog as log

BLOCK_NUMBER = 8
UB_FP16_SIZE = 120 * 1024


def ceil_div_offline(value, factor):
    if value % factor == 0:
        return value // factor
    else:
        return value // factor + 1


def cls_prob(cls_input_dic, dir_input_dic, output1_dic, output2_dic,
    use_sigmoid=True, kernel_name="ClassCal"):
    '''
    ----------
    cls_input_dic: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, the data type, src_dtype equals dst_dtype,
        only support fp16
        cls_shape: NCHW   [N, Anchors * cls_num, H, W] 
    dir_input_dic: dict
        input dic: dir_shape: NCHW   [N, Anchors * 2, H, W]
    output1_dic: dict
        output1 dic: cls_output: NCHW  [N, Anchors, cls_num, Aligned16(H*W)]
    output2_dic:dict
        output2 dic: dir_output: NCHW  [N, Anchors, 1, Aligned16(H*W)]
    use_sigmoid: bool
        use_sigmoid attr
    kernel_name: str
        cce kernel name, default value is "ClassCal"

    Returns
    ------
    '''
    dtype = cls_input_dic.get("dtype")
    cls_shape = cls_input_dic.get("shape")
    dir_shape = dir_input_dic.get("shape")
    inst = ClsProbBase(dtype, cls_shape, dir_shape, use_sigmoid, kernel_name)
    tik_instance = inst.build()
    return tik_instance


def sigmoid(tik_instance, ubinput_addr, uboutput_addr, length, c):
    scale_negative_1 = tik_instance.Scalar(dtype="float16")
    scale_negative_1.set_as(-1.0)
    scale_1 = tik_instance.Scalar(dtype="float16")
    scale_1.set_as(1.0)
    
    scalar_repeat_num = length // 128
    scalar_remain = length % 128
    tik_instance.vmuls(128, uboutput_addr[c, 0], ubinput_addr[c, 0],
                       scale_negative_1, scalar_repeat_num, 1, 1, 8, 8)
    with tik_instance.if_scope(scalar_remain > 0):
        tik_instance.vmuls(scalar_remain,
                           uboutput_addr[c, 128*scalar_repeat_num],
                           ubinput_addr[c, 128*scalar_repeat_num],
                           scale_negative_1, 1, 1, 1, 8, 8)
    tik_instance.vexp(128, ubinput_addr[c, 0],
                      uboutput_addr[c, 0], scalar_repeat_num, 1, 1, 8, 8)
    with tik_instance.if_scope(scalar_remain > 0):
        tik_instance.vexp(scalar_remain, 
                          ubinput_addr[c, 128*scalar_repeat_num],
                          uboutput_addr[c, 128*scalar_repeat_num],
                          1, 1, 1, 8, 8)
    tik_instance.vadds(128, uboutput_addr[c, 0], ubinput_addr[c, 0], scale_1,
                       scalar_repeat_num, 1, 1, 8, 8)
    with tik_instance.if_scope(scalar_remain > 0):
        tik_instance.vadds(scalar_remain,
                           uboutput_addr[c, 128*scalar_repeat_num],
                           ubinput_addr[c, 128*scalar_repeat_num],
                           scale_1, 1, 1, 1, 8, 8)
    tik_instance.vrec(128, ubinput_addr[c, 0], uboutput_addr[c, 0],
                      scalar_repeat_num, 1, 1, 8, 8)
    with tik_instance.if_scope(scalar_remain > 0):
        tik_instance.vrec(scalar_remain,
                          ubinput_addr[c, 128*scalar_repeat_num],
                          uboutput_addr[c, 128*scalar_repeat_num],
                          1, 1, 1, 8, 8)


def softmax_second(tik_instance, cls_num, repeat_time, data_exp_ub,
                   data_one_ub, data_zero_ub, vmask,
                   data_max_ub, last_index):
    with tik_instance.for_range(0, cls_num) as c:
        with tik_instance.for_range(0, repeat_time) as t:
            cmpmask = tik_instance.vcmp_eq(128, data_exp_ub[c, t*128],
                                           data_one_ub, 1, 1)
            tik_instance.vsel(128, 0, data_exp_ub[c, t*128], cmpmask,
                              data_exp_ub[c, t*128],
                              data_zero_ub, 1, 1, 1, 1, 8, 8, 8)
        with tik_instance.if_scope(vmask > 0):
            cmpmask = tik_instance.vcmp_eq(vmask,
                                           data_exp_ub[c, repeat_time*128],
                                           data_one_ub, 1, 1)
            tik_instance.vsel(vmask, 0, data_exp_ub[c, repeat_time*128],
                              cmpmask, data_exp_ub[c, repeat_time*128],
                              data_zero_ub, 1, 1, 1, 1, 8, 8, 8)
    tik_instance.vrec(128, data_max_ub, data_max_ub, repeat_time, 1, 1, 8, 8)
    with tik_instance.if_scope(vmask > 0):
        tik_instance.vrec(vmask, data_max_ub[last_index],
                          data_max_ub[last_index], 1, 1, 1, 8, 8)
    with tik_instance.for_range(0, cls_num) as c:
        tik_instance.vmul(128, data_exp_ub[c, 0], data_exp_ub[c, 0],
                          data_max_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        with tik_instance.if_scope(vmask > 0):
            tik_instance.vmul(vmask, data_exp_ub[c, last_index],
                              data_exp_ub[c, last_index],
                              data_max_ub[last_index], 1, 1, 1, 1, 8, 8, 8)


def softmax(tik_instance, data_max_ub, data_exp_ub, burst_length, repeat_time,
             vmask, last_index, cls_num, data_one_ub, data_zero_ub):
    tik_instance.tensor_mov(data_max_ub, data_exp_ub, "", 1,
                            burst_length, 0, 0)
    with tik_instance.for_range(1, cls_num) as c:
        tik_instance.vmax(128, data_max_ub, data_max_ub, data_exp_ub[c, 0], \
                          repeat_time, 1, 1, 1, 8, 8, 8)
        with tik_instance.if_scope(vmask > 0):
            tik_instance.vmax(vmask, data_max_ub[last_index],
                              data_max_ub[last_index],
                              data_exp_ub[c, last_index], 1, 1, 1, 1, 8, 8, 8)
    with tik_instance.for_range(0, cls_num) as c:
        tik_instance.vsub(128, data_exp_ub[c, 0], data_exp_ub[c, 0],
                          data_max_ub, repeat_time, 1, 1, 1, 8, 8, 8)
        tik_instance.vexp(128, data_exp_ub[c, 0], data_exp_ub[c, 0],
                          repeat_time, 1, 1, 8, 8)
        with tik_instance.if_scope(vmask > 0):
            tik_instance.vsub(vmask, data_exp_ub[c, last_index],
                              data_exp_ub[c, last_index],
                              data_max_ub[last_index], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vexp(vmask, data_exp_ub[c, last_index],
                              data_exp_ub[c, last_index], 1, 1, 1, 8, 8)
    # sum
    tik_instance.tensor_mov(data_max_ub, data_exp_ub, "", 1,
                            burst_length, 0, 0)
    with tik_instance.for_range(1, cls_num) as c:
        tik_instance.vadd(128, data_max_ub, data_max_ub,
                          data_exp_ub[c, 0], repeat_time, 1, 1, 1, 8, 8, 8)
        with tik_instance.if_scope(vmask > 0):
            tik_instance.vadd(vmask, data_max_ub[last_index],
                              data_max_ub[last_index],
                              data_exp_ub[c, last_index], 1, 1, 1, 1, 8, 8, 8)
    # each anchor reserve max probablity others set 0
    softmax_second(tik_instance, cls_num, repeat_time, data_exp_ub,
                   data_one_ub, data_zero_ub, vmask,
                   data_max_ub, last_index)


class ClsProbBase:
    def __init__(self, dtype, cls_shape, dir_shape, use_sigmoid, kernel_name):
        '''
        cls_shape:  N, Anchors * cls_num, H, W
        dir_shape:  N, Anchors * 2, H, W
        '''
        self.INT32 = "int32"
        self.FLOAT16 = "float16"
        self.kernel_name_ = kernel_name
        self.cls_shape = cls_shape
        self.dir_shape = dir_shape
        self.batch_size_ = dir_shape[0]
        self.anchor_num_ = dir_shape[1] // 2
        self.cls_num_ = cls_shape[1] // self.anchor_num_
        self.height_ = dir_shape[2]
        self.width_ = dir_shape[3]
        self.dtype_ = dtype
        self.use_sigmoid_ = use_sigmoid
        self.hw_size_ = cls_shape[2] * cls_shape[3]
        self.cls_batch_size_ = cls_shape[1] * cls_shape[2] * cls_shape[3] 
        self.dir_batch_size_ = dir_shape[1] * dir_shape[2] * dir_shape[3] 
        self.mask_batch_size_ = self.anchor_num_ * self.hw_size_
        self.hw_align16_size_ = ceil_div_offline(self.width_ * self.height_,
                                                 16) * 16 
        self.cls_batch_align_size_ = cls_shape[1] * self.hw_align16_size_
        self.dir_batch_align_size_ = dir_shape[1] * self.hw_align16_size_
        self.last_block_vld_ = (self.width_ * self.height_) % 16
        padding = 16 if self.hw_size_ % 16 else 0 
        self.norm_param()
        self.dir_param()

        # gm input output
        cls_gm_in_size  = cls_shape[0] * self.cls_batch_size_ + padding 
        cls_gm_out_size = cls_shape[0] * cls_shape[1] * self.hw_align16_size_ 
        dir_gm_in_size  = dir_shape[0] * self.dir_batch_size_ + padding 
        dir_gm_out_size = dir_shape[0] * self.anchor_num_ * \
                          self.hw_align16_size_ 
        self.tik_instance_ = tik.Tik(tik.Dprofile())
        self.cls_gm_in  = self.tik_instance_.Tensor(self.FLOAT16,
                                                    (cls_gm_in_size,),
                                                    name="cls_gm_in",
                                                    scope=tik.scope_gm)
        self.cls_gm_out = self.tik_instance_.Tensor(self.FLOAT16,
                                                    (cls_gm_out_size,),
                                                    name="cls_gm_out",
                                                    scope=tik.scope_gm)
        self.dir_gm_in  = self.tik_instance_.Tensor(self.FLOAT16,
                                                    (dir_gm_in_size,),
                                                    name="dir_gm_in",
                                                    scope=tik.scope_gm)
        self.dir_gm_out = self.tik_instance_.Tensor(self.FLOAT16,
                                                    (dir_gm_out_size,),
                                                    name="dir_gm_out",
                                                    scope=tik.scope_gm)

    def norm_param(self):
        ub_buf = (UB_FP16_SIZE / 2) // (self.cls_num_ + 1) 
        self.burst_len_ = ub_buf // 16 
        self.ub_buf_ = self.burst_len_ * 16 
        self.loop_num_  = (self.hw_size_ - 1) // self.ub_buf_ + 1
        self.repeat_time_ = self.burst_len_ // BLOCK_NUMBER
        self.index_ = self.repeat_time_ * 128
        self.vmask_ = self.ub_buf_ - self.repeat_time_ * 128
        self.last_burst_len_ = ceil_div_offline(
            self.hw_size_ - self.ub_buf_ * (self.loop_num_ - 1), 16)
        self.last_repeat_time_ = self.last_burst_len_ // BLOCK_NUMBER
        self.last_index_ = self.last_repeat_time_ * 128
        self.last_vmask_ = self.last_burst_len_ * 16 - \
                           self.last_repeat_time_ * 128 

    def dir_param(self):
        ub_buf = (UB_FP16_SIZE / 2) // 2 
        self.dir_burst_len_ = ub_buf // 16 
        self.dir_ub_buf_ = self.dir_burst_len_ * 16 
        self.dir_loop_num_  = (self.hw_size_ - 1) // self.dir_ub_buf_ + 1
        self.dir_last_burst_len_ = ceil_div_offline(
            self.hw_size_ - self.dir_ub_buf_ * (self.dir_loop_num_ - 1), 16)


    def class_sigmoid_second(self, data_sigmoid_ub, burst_len, data_sigmax_ub,
                             repeat_time, vmask, last_index,
                             batch_id, anchor_id, loop_id):
        with self.tik_instance_.for_range(0, self.cls_num_) as c:
            sigmoid(self.tik_instance_, data_sigmoid_ub, data_sigmoid_ub,
                    burst_len*16, c)
        if self.cls_num_ > 1: 
            self.tik_instance_.tensor_mov(data_sigmax_ub, data_sigmoid_ub, "",
                                            1, burst_len, 0, 0)
            with self.tik_instance_.for_range(1, self.cls_num_) as c:
                self.tik_instance_.vmax(128, data_sigmax_ub, data_sigmax_ub,
                                        data_sigmoid_ub[c, 0],
                                        repeat_time, 1, 1, 1, 8, 8, 8)
                with self.tik_instance_.if_scope(vmask > 0):
                    self.tik_instance_.vmax(vmask, data_sigmax_ub[last_index],
                                            data_sigmax_ub[last_index],
                                            data_sigmoid_ub[c, last_index],
                                            1, 1, 1, 1, 8, 8, 8)
            with self.tik_instance_.for_range(0, self.cls_num_) as c:
                with self.tik_instance_.for_range(0, repeat_time) as t:
                    cmpmask = self.tik_instance_.vcmp_eq(
                        128, data_sigmoid_ub[c, t*128],
                        data_sigmax_ub[t*128], 1, 1)
                    self.tik_instance_.vsel(128, 0, data_sigmoid_ub[c, t*128],
                        cmpmask, data_sigmoid_ub[c, t*128],
                        self.data_zero_ub, 1, 1, 1, 1, 8, 8, 8)
                with self.tik_instance_.if_scope(vmask > 0):
                    cmpmask = self.tik_instance_.vcmp_eq(vmask,
                        data_sigmoid_ub[c, last_index],
                        data_sigmax_ub[last_index], 1, 1)
                    self.tik_instance_.vsel(vmask, 0,
                        data_sigmoid_ub[c, last_index],
                        cmpmask, data_sigmoid_ub[c, last_index],
                        self.data_zero_ub, 1, 1, 1, 1, 8, 8, 8)
        
        self.mask_padding(batch_id, anchor_id, loop_id, data_sigmoid_ub,
                          burst_len, repeat_time, vmask, last_index)
        with self.tik_instance_.for_range(0, self.cls_num_) as c:
            gm_out_addr = self.cls_batch_align_size_ * batch_id + anchor_id * \
                          self.cls_num_ * self.hw_align16_size_ + c * \
                          self.hw_align16_size_ + loop_id * self.ub_buf_
            self.tik_instance_.tensor_mov(self.cls_gm_out[gm_out_addr],
                                          data_sigmoid_ub[c, 0], "",
                                          1, burst_len, 0, 0)


    def class_sigmoid(self, batch_id, anchor_id, loop_num):
        with self.tik_instance_.new_stmt_scope():
            thread_nums = 2
            if (loop_num <= 1):
                thread_nums = loop_num    
            with self.tik_instance_.for_range(
                0, loop_num, thread_num=thread_nums) as loop_id:
                data_sigmoid_ub = self.tik_instance_.Tensor(
                    self.FLOAT16, (self.cls_num_, self.ub_buf_),
                    name="data_sigmoid_ub", scope=tik.scope_ubuf)
                data_sigmax_ub = self.tik_instance_.Tensor(
                    self.FLOAT16, (self.ub_buf_,),
                    name="data_sigmax_ub", scope=tik.scope_ubuf)    
                burst_lens = self.tik_instance_.Scalar(
                    self.INT32, name="sigmoid_burst_len")
                repeat_time = self.tik_instance_.Scalar(
                    self.INT32, name="sigmoid_repeat_time")
                last_index = self.tik_instance_.Scalar(
                    self.INT32, name="sigmoid_last_index")
                vmask = self.tik_instance_.Scalar(self.INT32,
                                                  name="sigmoid_vmask")
                with self.tik_instance_.if_scope(loop_id == loop_num - 1):
                    burst_lens.set_as(self.last_burst_len_)
                    repeat_time.set_as(self.last_repeat_time_)
                    last_index.set_as(self.last_index_)
                    vmask.set_as(self.last_vmask_)
                with self.tik_instance_.else_scope():
                    burst_lens.set_as(self.burst_len_)
                    repeat_time.set_as(self.repeat_time_)
                    last_index.set_as(self.index_)
                    vmask.set_as(self.vmask_)
                with self.tik_instance_.for_range(0, self.cls_num_) as c:
                    gm_addr = self.cls_batch_size_ * batch_id + \
                              anchor_id * self.cls_num_ * self.hw_size_ + \
                              c * self.hw_size_ + loop_id * self.ub_buf_
                    self.tik_instance_.tensor_mov(data_sigmoid_ub[c, 0],
                                                  self.cls_gm_in[gm_addr], "",
                                                  1, burst_lens, 0, 0)
                self.class_sigmoid_second(data_sigmoid_ub, burst_lens,
                    data_sigmax_ub, repeat_time, vmask, last_index,
                    batch_id, anchor_id, loop_id)

    
    def mask_padding_seconde(self, loop_id, data_ub, burst_len, scalar_zero):
        with self.tik_instance_.if_scope(loop_id == (self.loop_num_ - 1)):
            with self.tik_instance_.for_range(0, self.cls_num_) as c:
                with self.tik_instance_.for_range(
                    0, 16-self.last_block_vld_) as vld_idx:
                    data_ub[c, burst_len*16-1-vld_idx].set_as(scalar_zero)


    def mask_padding(self, batch_id, anchor_id, loop_id, data_ub,
                     burst_len, repeat_time, vmask, last_index):
        if(self.last_block_vld_ > 0):
            scalar_zero = self.tik_instance_.Scalar(self.FLOAT16,
                                                    name="scalar_zero")
            scalar_zero.set_as(0.0)
            self.mask_padding_seconde(loop_id, data_ub, burst_len, scalar_zero)


    def multiclass_softmax(self, batch_id, anchor_id, loop_num):
        with self.tik_instance_.new_stmt_scope():
            thread_nums = 2
            if (loop_num <= 1):
                thread_nums = loop_num
            with self.tik_instance_.for_range(
                0, loop_num, thread_num=thread_nums) as loop_id:
                burst_len = self.tik_instance_.Scalar(
                    self.INT32, name="softmax_burst_length")
                repeat_time = self.tik_instance_.Scalar(
                    self.INT32, name="softmax_repeat_time")
                last_index = self.tik_instance_.Scalar(
                    self.INT32, name="softmax_last_index")
                vmask = self.tik_instance_.Scalar(self.INT32,
                                                  name="softmax_vmask")
                data_exp_ub = self.tik_instance_.Tensor(
                    self.FLOAT16, (self.cls_num_, self.ub_buf_),
                    name="data_ub_exp", scope=tik.scope_ubuf)
                data_max_ub = self.tik_instance_.Tensor(
                    self.FLOAT16, (self.ub_buf_,),
                    name="data_max_ub", scope=tik.scope_ubuf)
                with self.tik_instance_.if_scope(loop_id == (loop_num - 1)):
                    burst_len.set_as(self.last_burst_len_)
                    repeat_time.set_as(self.last_repeat_time_)
                    last_index.set_as(self.last_index_)
                    vmask.set_as(self.last_vmask_)
                with self.tik_instance_.else_scope():
                    burst_len.set_as(self.burst_len_)
                    repeat_time.set_as(self.repeat_time_)
                    last_index.set_as(self.index_)
                    vmask.set_as(self.vmask_)
                # load data from gm
                with self.tik_instance_.for_range(0, self.cls_num_) as c:
                    gm_addr = self.cls_batch_size_ * batch_id + anchor_id * \
                              self.cls_num_ * self.hw_size_ + \
                              c * self.hw_size_ + loop_id * self.ub_buf_
                    self.tik_instance_.tensor_mov(data_exp_ub[c, 0],
                                                  self.cls_gm_in[gm_addr], "",
                                                  1, burst_len, 0, 0)
                softmax(self.tik_instance_, data_max_ub, data_exp_ub,
                        burst_len, repeat_time, vmask, last_index,
                        self.cls_num_, self.data_one_ub, self.data_zero_ub)
                self.mask_padding(batch_id, anchor_id, loop_id, data_exp_ub,
                                  burst_len, repeat_time, vmask, last_index)
                with self.tik_instance_.for_range(0, self.cls_num_) as c:
                    gm_out_addr = self.cls_batch_align_size_ * batch_id + \
                                  anchor_id * self.cls_num_ * \
                                  self.hw_align16_size_ + c * \
                                  self.hw_align16_size_ + loop_id * \
                                  self.ub_buf_
                    self.tik_instance_.tensor_mov(self.cls_gm_out[gm_out_addr],
                        data_exp_ub[c, 0], "", 1, burst_len, 0, 0)


    def dir_cmp(self, dir_0, dir_1, length):
        repeat_num = length // 128
        remain = length % 128
        with self.tik_instance_.for_range(0, repeat_num) as t:
            cmpmask = self.tik_instance_.vcmp_ge(128, dir_0[t*128],
                                                 dir_1[t*128], 1, 1)
            self.tik_instance_.vsel(128, 0, dir_0[t*128], cmpmask,
                                    self.data_one_ub, self.data_zero_ub,
                                    1, 1, 1, 1, 8, 8, 8)
        with self.tik_instance_.if_scope(remain > 0):
            cmpmask = self.tik_instance_.vcmp_ge(remain, dir_0[repeat_num*128],
                                                 dir_1[128*repeat_num], 1, 1)
            self.tik_instance_.vsel(remain, 0, dir_0[repeat_num*128], cmpmask,
                                    self.data_one_ub, \
                                    self.data_zero_ub, 1, 1, 1, 1, 8, 8, 8)

    def direction_max_cal(self, batch_id, anchor_id, loop_num):
        with self.tik_instance_.new_stmt_scope():
            with self.tik_instance_.for_range(0, loop_num,
                                              thread_num=2) as loop_id:
                ub_buf = self.dir_ub_buf_
                data_dir_ub_0 = self.tik_instance_.Tensor(
                    self.FLOAT16, (ub_buf,),
                    name="dir_ub_0", scope=tik.scope_ubuf)
                data_dir_ub_1 = self.tik_instance_.Tensor(
                    self.FLOAT16, (ub_buf,),
                    name="dir_ub_1", scope=tik.scope_ubuf)
                burst_len = self.tik_instance_.Scalar(self.INT32,
                                                      name="dir_burst_length")
                with self.tik_instance_.if_scope(loop_id == loop_num - 1):
                    burst_len.set_as(self.dir_last_burst_len_)
                with self.tik_instance_.else_scope():
                    burst_len.set_as(self.dir_burst_len_)
                gm_addr0 = self.width_ * self.height_ * 2 * \
                           self.anchor_num_ * batch_id + self.width_ * \
                           self.height_ * 2 * anchor_id + loop_id * ub_buf
                gm_addr1 = self.width_ * self.height_ * 2 * \
                           self.anchor_num_ * batch_id + self.width_ * \
                           self.height_ * 2 * anchor_id + self.width_ * \
                           self.height_ + loop_id * ub_buf
                self.tik_instance_.tensor_mov(data_dir_ub_0,
                                              self.dir_gm_in[gm_addr0],
                                              "", 1, burst_len, 0, 0)
                self.tik_instance_.tensor_mov(data_dir_ub_1,
                                              self.dir_gm_in[gm_addr1],
                                              "", 1, burst_len, 0, 0)
                self.dir_cmp(data_dir_ub_0, data_dir_ub_1, burst_len*16)
                gm_out_addr = self.hw_align16_size_ *  self.anchor_num_ * \
                              batch_id + self.hw_align16_size_ * \
                              anchor_id + loop_id * ub_buf
                self.tik_instance_.tensor_mov(self.dir_gm_out[gm_out_addr],
                                              data_dir_ub_0,  "", 1,
                                              burst_len, 0, 0)

    def param_check(self):
        log.check_eq(self.dtype_, self.FLOAT16,
                     "data type only support float16")
        log.check_eq(self.cls_shape[2], self.dir_shape[2],
                     "the height of cls_shape and dir_shape must be the same")
        log.check_eq(self.cls_shape[3], self.dir_shape[3],
                     "the width of cls_shape and dir_shape must be the same")
        log.check_eq(self.dir_shape[1] % 2, 0,
                     "dir_shape[1] should equal to 2 * anchors")
        log.check_eq(self.cls_shape[1] % self.anchor_num_, 0,
                     "cls_shape[1] should equal to N * anchors")
        log.check_ge(self.cls_shape[1], self.anchor_num_,
                     "cls_shape[1] should larger than anchor_num")
        log.check_kernelname(self.kernel_name_)
        
    
    def process(self):
        self.data_one_ub = self.tik_instance_.Tensor(self.FLOAT16, (128,),
                                                     name="data_one_ub",
                                                     scope=tik.scope_ubuf)
        self.data_zero_ub = self.tik_instance_.Tensor(self.FLOAT16, (128,),
                                                      name="data_zero_ub",
                                                      scope=tik.scope_ubuf)
        self.tik_instance_.vector_dup(128, self.data_one_ub, 1, 1, 1, 8)
        self.tik_instance_.vector_dup(128, self.data_zero_ub, 0, 1, 1, 8)
        with self.tik_instance_.for_range(0, self.batch_size_) as batch_id:
            with self.tik_instance_.for_range(0, self.anchor_num_,
                block_num=self.anchor_num_) as anchor_id:
                if self.use_sigmoid_:
                    self.class_sigmoid(batch_id, anchor_id, self.loop_num_)
                else:
                    self.multiclass_softmax(batch_id, anchor_id,
                                            self.loop_num_)
                self.direction_max_cal(batch_id, anchor_id, self.dir_loop_num_)
        
    def build(self):
        self.param_check()
        self.process()
        self.tik_instance_.BuildCCE(self.kernel_name_, inputs=[self.cls_gm_in,
                                    self.dir_gm_in], 
                                    outputs=[self.cls_gm_out, self.dir_gm_out],
                                    enable_l2=True)
        return self.tik_instance_

