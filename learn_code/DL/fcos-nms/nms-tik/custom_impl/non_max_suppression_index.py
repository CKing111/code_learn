# encoding=utf-8
from ascend import AVecBuf
from ascend import VecGCmd
from ascend import VecGExecutor
from uti import interface_check
from version import get_version
from non_max_suppression.common_method import CommonMethod
from non_max_suppression.merge_sort import MergeSort
from non_max_suppression.non_max_suppression import NonMaximumSuppression
from non_max_suppression.get_boxes_num import GetBoxesNom


class NonMaxSuppressionIndex(object):
    """
    tensorflow NonMaxSuppressionV3
    """

    def __init__(self, boxes, scores, selected_indices, max_output_size,
                 iou_threshold, score_threshold, kernel_name):
        """
        init attributes
        :param boxes:
            {"shape": (4, boxes_num), "dtype": "float32", "format": "ND"}
        :param scores:
            {"shape": (boxes_num, ), "dtype": "float32", "format": "ND"}
        :param selected_indices:
            {"shape": (max_output_size, ), "dtype": "int32", "format": "ND"}
        :param max_output_size: max num of select boxes; int
        :param iou_threshold:
            threshold of intersection over Union between two boxes; <=
            float
        :param score_threshold:
            scores threshold; >
            float
        :param kernel_name: str
        """
        interface_check.check_kernelname(kernel_name)
        self._check_attrs(max_output_size, iou_threshold, score_threshold)
        self._check_params(boxes, scores, selected_indices)
        self._check_params_shape(boxes, scores, selected_indices,
                                 max_output_size)

        self.cont = get_version.get_aicore_container(("Ascend610",))
        self.tik, self.tik_inst = self.cont.tik, self.cont.tinst
        self.method = CommonMethod(self.cont)
        self.pro_data_num = self.cont.const_proposal_data_num
        self.pro_repeat_num = self.cont.const_proposal_repeat_num
        self.ub_size = self.cont.const_ub_max_byte
        self.l1_size = self.cont.const_l1_max_byte
        self.repeat_time_max = self.cont.const_vector_proc_max_rpt
        self.ai_core_use = self.cont.const_aicore_num

        self.data_type = boxes.get("dtype")
        self.data_size, self.block_data_num, self.repeat_data_num = \
            self.method.get_type_const(self.data_type)
        self.block_pro_num = self.method.get_block_pro_num(self.data_type)
        self.int32_type = "int32"
        (self.int32_size, self.int32_block_data_num,
         self.int32_repeat_data_num) = \
            self.method.get_type_const(self.int32_type)

        self.neg_inf, self.inf = -65504.0, 65504.0
        self.get_boxes_num = GetBoxesNom(
            self.cont, self.data_type, self.ub_size)
        self.merge_sort = MergeSort(
            self.cont, self.data_type, self.ub_size)
        self.nms = NonMaximumSuppression(
            self.cont, self.data_type, self.ub_size)

        self.kernel_name = kernel_name
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.boxes_num = boxes.get("shape")[1]
        self.proposal_num = self.method.get_align_num(
            self.boxes_num, self.pro_repeat_num)
        # count merge sort num
        self.ub_pro_num_max, self.ub_sort_num = \
            self.merge_sort.get_pro_num_info()
        self.sort_max_num = min(self.ub_sort_num, self.proposal_num)
        self.boxes = self.tik_inst.Tensor(boxes.get("dtype"),
                                          boxes.get("shape"),
                                          self.tik.scope_gm, "boxes")
        self.scores = self.tik_inst.Tensor(scores.get("dtype"),
                                           scores.get("shape"),
                                           self.tik.scope_gm, "scores")
        self.selected_indices = self.tik_inst.Tensor(
            selected_indices.get("dtype"), selected_indices.get("shape"),
            self.tik.scope_gm, "selected_indices")

    @staticmethod
    def _check_attrs(max_output_size, iou_threshold, score_threshold):
        """
        check max_out_size, iou_threshold
        max_out_size: int; 0<max_out_size<=512
        iou_threshold: float; 0<=iou_threshold<=1
        score_threshold: float
        """
        if not isinstance(max_output_size, int):
            raise RuntimeError(
                "max_output_size type is not supported, must be int")
        if max_output_size <= 0 or max_output_size > 512:
            raise RuntimeError("max_output_size is illegal")
        if not isinstance(iou_threshold, float):
            raise RuntimeError(
                "iou_threshold type is not supported, must be float")
        if iou_threshold < 0 or iou_threshold > 1:
            raise RuntimeError("iou_threshold is illegal")
        if not isinstance(score_threshold, float):
            raise RuntimeError(
                "score_threshold type is not supported, must be float")

    @staticmethod
    def _check_params(boxes, scores, selected_indices):
        interface_check.check_param(boxes, [2], ["float32"],
                                    ["NCHW", "ND", "NHWC"])
        interface_check.check_param(scores, [1], ["float32"],
                                    ["NCHW", "ND", "NHWC"])
        interface_check.check_param(selected_indices, [1], ["int32"],
                                    ["NCHW", "ND", "NHWC"])

    @staticmethod
    def _check_params_shape(boxes, scores, selected_indices, max_output_size):
        """
        :param boxes: boxes_shape: (4, boxes_num);
        :param scores: scores_shape: (boxes_num, );
        :param selected_indices: selected_indices_shape: (max_output_size, )
        :param max_output_size: int
        :return:
        """
        boxes_shape = boxes.get("shape")
        scores_shape = scores.get("shape")
        selected_indices_shape = selected_indices.get("shape")
        boxes_num = scores_shape[0]
        if boxes_num > 1000000:
            raise RuntimeError("boxes_num dose not support")
        if boxes_shape != (4, boxes_num):
            raise RuntimeError("boxes shape dose not support")
        if selected_indices_shape != (max_output_size,):
            raise RuntimeError("selected_indices shape dose not support")

    def _score_move_in_fun(self, scores_ub, start_index, data_num):
        block_num = self.method.ceil_div(data_num, self.block_data_num)
        self.tik_inst.data_move(scores_ub, self.scores[start_index], 0, 1,
                                block_num, 0, 0)

    def mode1_compute(self):
        # get scores num bigger than score_threshold
        num_max = self.get_boxes_num.get_boxes_num(
            self.boxes_num, self.score_threshold, self.neg_inf,
            self.inf, self._score_move_in_fun)
        with self.tik_inst.if_scope(num_max > 0):
            self._mode1_compute_each_core(num_max)
        with self.tik_inst.else_scope():
            self._get_no_result()
        self.tik_inst.BuildCCE(inputs=[self.boxes, self.scores],
                               outputs=[self.selected_indices],
                               kernel_name=self.kernel_name)

    def _get_no_result(self):
        max_output_size_format = self.method.get_align_num(
            self.max_output_size, self.repeat_data_num)
        select_index_int32 = self.tik_inst.Tensor(
            self.int32_type, (max_output_size_format,),
            self.tik.scope_ubuf, "select_index_int32")
        self.method.vector_dup(select_index_int32, -1)
        self.tik_inst.data_move(
            self.selected_indices, select_index_int32, 0, 1,
            self.method.ceil_div(self.max_output_size,
                                 self.int32_block_data_num),
            0, 0)

    def _mode1_compute_each_core(self, num_max):
        # get boxes proposal and sort proposal
        top_proposal_l1 = self.tik_inst.Tensor(
            self.data_type, (1, self.ub_pro_num_max, self.pro_data_num),
            self.tik.scope_cbuf, "top_proposal")
        self.merge_sort.get_top_proposal(
            top_proposal_l1, 0, self.boxes_num, self.sort_max_num,
            get_proposal_ub=self._get_proposal_and_sort)
        # get nms result
        with self.tik_inst.if_scope(num_max > self.sort_max_num):
            num_max.set_as(self.sort_max_num)
        select_pro_ub, select_num_scalar = \
            self.nms.non_maximum_suppression(
                top_proposal_l1, 0, num_max, self.max_output_size,
                self.iou_threshold)
        max_output_size_format = self.method.get_align_num(
            self.max_output_size, self.repeat_data_num)
        select_index_fp32 = self.tik_inst.Tensor("float32",
                                                 (max_output_size_format,),
                                                 self.tik.scope_ubuf,
                                                 "select_index_fp32")
        select_index_int32 = self.tik_inst.Tensor("int32",
                                                  (max_output_size_format,),
                                                  self.tik.scope_ubuf,
                                                  "select_index_int32")
        self.tik_inst.vextract(select_index_fp32, select_pro_ub,
                               self.method.ceil_div(self.max_output_size,
                                                    self.pro_repeat_num), 5)
        self.tik_inst.vconv(64, "round", select_index_int32, select_index_fp32,
                            max_output_size_format // 64, 1, 1, 8, 8)
        with self.tik_inst.if_scope(select_num_scalar < self.max_output_size):
            last_index_scalar = self.tik_inst.Scalar("int32")
            last_index_scalar.set_as(select_index_int32[select_num_scalar - 1])
            with self.tik_inst.for_range(select_num_scalar,
                                         self.max_output_size) as result_index:
                select_index_int32[result_index].set_as(last_index_scalar)
        self.tik_inst.data_move(self.selected_indices, select_index_int32, 0,
                                1,
                                self.method.ceil_div(
                                    self.max_output_size,
                                    self.int32_block_data_num),
                                0, 0)

    def _get_proposal_and_sort(self, boxes_num, start_index):
        # get_proposal
        proposal_num = self.method.get_align_num(
            boxes_num, self.pro_repeat_num)
        ub_proposal_1 = \
            self.tik_inst.Tensor(self.data_type,
                                 (proposal_num, self.pro_data_num),
                                 self.tik.scope_ubuf, "ub_proposal_1")
        self._get_proposal(ub_proposal_1, start_index, boxes_num, proposal_num)
        ub_proposal_2 = \
            self.tik_inst.Tensor(self.data_type,
                                 (proposal_num, self.pro_data_num),
                                 self.tik.scope_ubuf,
                                 "ub_proposal_2")
        return ub_proposal_1, ub_proposal_2

    def _get_proposal(self, proposal_ub, start_index, boxes_num, proposal_num):
        with self.tik_inst.new_stmt_scope():
            boxes_def_num = 0.0
            scores_def_num = self.neg_inf
            block_move_in = self.method.ceil_div(boxes_num,
                                                 self.block_data_num)
            mask_h, mask_l, index_last = self.method.get_mask(
                boxes_num, self.repeat_data_num, self.block_data_num)
            proposal_data_ub = self.tik_inst.Tensor(self.data_type,
                                                    (proposal_num,),
                                                    self.tik.scope_ubuf,
                                                    "proposal_data_ub")
            # boxes move in
            self.method.vector_dup(proposal_data_ub,
                                   boxes_def_num)
            with self.tik_inst.for_range(0, 4) as index_channel:
                self.tik_inst.data_move(proposal_data_ub,
                                        self.boxes[index_channel, start_index],
                                        0, 1, block_move_in, 0, 0)
                if mask_h != 0 or mask_l != 0:
                    self.tik_inst.vector_dup([mask_h, mask_l],
                                             proposal_data_ub[index_last],
                                             boxes_def_num, 1, 1, 8)
                self.method.vector_concat(proposal_ub, proposal_data_ub,
                                          index_channel, proposal_num)
            # scores move in
            self.method.vector_dup(proposal_data_ub, scores_def_num)
            self.tik_inst.data_move(proposal_data_ub, self.scores[start_index],
                                    0, 1, block_move_in, 0, 0)
            if mask_h != 0 or mask_l != 0:
                self.tik_inst.vector_dup([mask_h, mask_l],
                                         proposal_data_ub[index_last],
                                         scores_def_num, 1, 1, 8)
            self.method.vector_concat(proposal_ub, proposal_data_ub, 4,
                                      proposal_num)
            # get index
            self._get_index(proposal_data_ub, start_index, proposal_num)
            self.method.vector_concat(proposal_ub, proposal_data_ub, 5,
                                      proposal_num)

    def _get_index(self, index_ub, start_index, proposal_num):
        padded_num = self.block_data_num
        start_index_fp32 = self.tik_inst.Scalar(self.data_type)
        with self.tik_inst.new_stmt_scope():
            shape = (padded_num,)
            ub_fp32 = self.tik_inst.Tensor(self.data_type, shape,
                                           self.tik.scope_ubuf, "ub_fp32")
            ub_int32 = self.tik_inst.Tensor("int32", shape,
                                            self.tik.scope_ubuf, "ub_int32")
            self.tik_inst.vector_dup(padded_num, ub_int32, start_index, 1, 1,
                                     8)
            self.tik_inst.vconv(padded_num, '', ub_fp32, ub_int32, 1, 1, 1, 8,
                                8)
            start_index_fp32.set_as(ub_fp32[0])
        self.tik_inst.vector_dup(padded_num, index_ub, start_index_fp32, 1, 1,
                                 8)
        stride = 1.0
        for pad_index in range(1, padded_num):
            mask_pad = 2 ** pad_index
            self.tik_inst.vadds([0, mask_pad], index_ub, index_ub,
                                stride * pad_index, 1, 1, 1, 8, 8)
        self._pad_index(index_ub, padded_num, stride, proposal_num)

    def _pad_index(self, index_ub, padded_num, stride, data_num):
        if padded_num < data_num:
            last_num = data_num - padded_num
            if last_num > padded_num:
                padding_num = padded_num
            else:
                padding_num = last_num
            buf_pad_all = {
                "tensor_src": AVecBuf(index_ub, padding_num, 0, self.cont,
                                      False, self.repeat_data_num),
                "tensor_dst": AVecBuf(index_ub, padding_num, padded_num,
                                      self.cont, False,
                                      self.repeat_data_num), }
            cmd_pad_tensor = [VecGCmd("vadds", "tensor_dst", "tensor_src",
                                      scalar=stride * padded_num)]
            VecGExecutor.exec_vec_g_cmd(self.cont, buf_pad_all, cmd_pad_tensor,
                                        "tensor_src")
            self._pad_index(index_ub, padded_num + padding_num, stride,
                            data_num)

    def tik_output_debug(self):
        return self.tik_inst


def non_max_suppression_index(boxes,
                              scores,
                              selected_indices,
                              max_output_size,
                              iou_threshold,
                              score_threshold,
                              kernel_name="NonMaxSuppressionIndex",
                              test=False):
    obj = NonMaxSuppressionIndex(boxes, scores, selected_indices,
                                 max_output_size, iou_threshold,
                                 score_threshold, kernel_name)
    obj.mode1_compute()
    if not test:
        return 0

    obj.tik_output_debug()
    return 0
