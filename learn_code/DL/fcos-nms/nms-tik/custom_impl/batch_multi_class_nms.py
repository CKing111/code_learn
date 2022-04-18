from uti import interface_check
from version import get_version
from non_max_suppression.common_method import CommonMethod
from non_max_suppression.merge_sort import MergeSort
from non_max_suppression.non_max_suppression import NonMaximumSuppression
from non_max_suppression.get_boxes_num import GetBoxesNom


class BatchMultiClassNonMaxSuppression(object):
    """
    batch multi class non max suppression
    """

    def __init__(self, boxes, scores, proposal_selected, max_output_size,
                 iou_threshold, score_threshold, kind_start_index, sort_num,
                 weights, biases, kernel_name):
        """
        init attributes
        Args:
            boxes: {"shape": (batch_num, 4, kind_num, boxes_num),
                    "dtype": "float16", "format": "NCHW"}
            scores: {"shape": (batch_num, 1, kind_num, boxes_num),
                     "dtype": "float16", "format": "NCHW"}
            proposal_selected:
                {"shape": (batch_num, kind_num, max_output_size, 8),
                "dtype": "float16", "format": "NCHW"}
            max_output_size: max num of select boxes; int
            iou_threshold:
                threshold of intersection over Union between two boxes,
                iou <= iou_threshold; float
            score_threshold:
                threshold of scores,
                score > score_threshold; float16
            sort_num: int
            weights: ListFloat, len == 4
            biases: ListFloat, len == 4
            kernel_name: str
        """
        interface_check.check_kernelname(kernel_name)
        self._check_attr(max_output_size, iou_threshold, score_threshold,
                         kind_start_index, sort_num, weights, biases)
        self._check_params(boxes, scores, proposal_selected)
        self._check_params_shape(boxes, scores, proposal_selected,
                                 max_output_size, kind_start_index)

        self.cont = get_version.get_aicore_container(("Ascend610",))
        self.tik, self.tik_inst = self.cont.tik, self.cont.tinst
        self.method = CommonMethod(self.cont)
        # 248 * 1024
        self.ub_size = self.cont.const_ub_max_byte
        self.l1_size = self.cont.const_l1_max_byte
        self.repeat_time_max = self.cont.const_vector_proc_max_rpt
        self.ai_core_use = self.cont.const_aicore_num
        self.pro_data_num = self.cont.const_proposal_data_num
        self.pro_repeat_num = self.cont.const_proposal_repeat_num

        self.data_type = boxes.get("dtype")
        self.data_size, self.block_data_num, self.repeat_data_num = \
            self.method.get_type_const(self.data_type)
        self.block_pro_num = self.method.get_block_pro_num(self.data_type)

        self.neg_fp16_inf, self.fp16_inf = -65504.0, 65504.0
        self.boxes_def_num, self.scores_def_num = 0.0, self.neg_fp16_inf
        self.label_def_num = -1.0

        self.get_boxes_num, self.merge_sort, self.nms = self._init_class()

        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.kind_start_index = kind_start_index
        self.sort_num, self.weights, self.biases = sort_num, weights, biases
        self.kernel_name = kernel_name

        self.batch_num, _, self.kind_num, self.boxes_num = boxes.get("shape")

        self.proposal_num = self.method.get_align_num(
            self.boxes_num, self.pro_repeat_num)
        self.max_output_size_format = self.method.get_align_num(
            self.max_output_size, self.pro_repeat_num)
        # merge sort need 1 extra space
        (self.ub_pro_num_max, self.ub_sort_num) = \
            self.merge_sort.get_pro_num_info()
        self.sort_num = min(self.sort_num, self.ub_sort_num, self.boxes_num)
        self.boxes = self.tik_inst.Tensor(
            boxes.get("dtype"), boxes.get("shape"), self.tik.scope_gm, "boxes")
        self.scores = self.tik_inst.Tensor(
            scores.get("dtype"), scores.get("shape"),
            self.tik.scope_gm, "scores")
        self.proposal_selected = self.tik_inst.Tensor(
            proposal_selected.get("dtype"), proposal_selected.get("shape"),
            self.tik.scope_gm, "proposal_selected")

    def _check_attr(self, max_output_size, iou_threshold, score_threshold,
                    kind_start_index, sort_num, weights, biases):
        """
        check max_out_size, iou_threshold
        Args:
            max_output_size: int; 0<max_out_size<=512
            iou_threshold: float; 0<=iou_threshold<=1
        """
        self._check_attr_1(max_output_size, iou_threshold, score_threshold,
                           kind_start_index, sort_num)
        self._check_attr_2(weights, biases)

    def _check_attr_1(self, max_output_size, iou_threshold, score_threshold,
                      kind_start_index, sort_num):
        """
        check max_out_size, iou_threshold
        Args:
            max_output_size: int; 0<max_out_size<=512
            iou_threshold: float; 0<=iou_threshold<=1
            score_threshold: float;
            kind_start_index: int; >= 0
            sort_num: int; > 0
        """
        if not isinstance(max_output_size, int):
            raise RuntimeError("max_output_size type is not supported, "
                               "must be int")
        if max_output_size <= 0 or max_output_size > 512:
            raise RuntimeError("max_output_size is illegal")
        if not isinstance(iou_threshold, float):
            raise RuntimeError("iou_threshold type is not supported, "
                               "must be float")
        if iou_threshold < 0 or iou_threshold > 1:
            raise RuntimeError("iou_threshold is illegal")
        if not isinstance(score_threshold, float):
            raise RuntimeError("score_threshold type is not supported, "
                               "must be float")
        if not isinstance(kind_start_index, int) or kind_start_index < 0:
            raise RuntimeError("kind_start_index type is not supported")
        if not isinstance(sort_num, int) or sort_num <= 0:
            raise RuntimeError("sort_num is not supported")

    def _check_attr_2(self, weights, biases):
        """
        Args:
            weights: List[float]
            biases: List[float]
        """
        if not isinstance(weights, (list, tuple)) or len(weights) != 4:
            raise RuntimeError("weights is not supported")
        if not isinstance(biases, (list, tuple)) or len(biases) != 4:
            raise RuntimeError("biases is not supported")
        weights_float = [isinstance(weight, float) for weight in weights]
        biases_float = [isinstance(bias, float) for bias in biases]
        if not all(weights_float):
            raise RuntimeError("weights is not supported")
        if not all(biases_float):
            raise RuntimeError("biases is not supported")

    def _check_params(self, boxes, scores, proposal_selected):
        interface_check.check_param(boxes, [4], ["float16"],
                                    ["NCHW", "ND"])
        interface_check.check_param(scores, [4], ["float16"],
                                    ["NCHW", "ND"])
        interface_check.check_param(proposal_selected, [4],
                                    ["float16"],
                                    ["NCHW", "ND"])

    def _check_params_shape(self, boxes, scores, proposal_selected,
                            max_output_size, kind_start_index):
        """
        boxes_shape: (batch_num, 4, kind_num, boxes_num)
        scores_shape: (batch_num, 1, kind_num, boxes_num)
        proposal_selected_shape: (batch_num, kind_num, max_output_size, 8)
        gm_size <= 1024 ** 3
        """
        boxes_shape = boxes.get("shape")
        scores_shape = scores.get("shape")
        proposal_selected_shape = proposal_selected.get("shape")
        batch_num, _, kind_num, boxes_num = boxes_shape
        if scores_shape != (batch_num, 1, kind_num, boxes_num):
            raise RuntimeError("scores shape dose not support")
        if proposal_selected_shape != (batch_num, kind_num,
                                       max_output_size, 8):
            raise RuntimeError("proposal_selected shape dose not support")
        # count by space and module limit
        if boxes_num < max_output_size:
            raise RuntimeError("boxes_num dose not support")
        if kind_num + kind_start_index > 2048:
            raise RuntimeError("kind_index dose not support")
        gm_size = self._get_gm_size(batch_num, kind_num,
                                    boxes_num, max_output_size)
        if gm_size > 1024 ** 3:
            raise RuntimeError("data size is too larger")

    def _get_gm_size(self, batch_num, kind_num, boxes_num,
                     max_output_size, use_ratio=0.8):
        data_size = 2
        gm_size = (batch_num * kind_num * boxes_num * 4 * data_size +
                   batch_num * kind_num * boxes_num * data_size +
                   batch_num * kind_num * max_output_size * 8 * data_size)
        gm_size = gm_size / use_ratio
        return gm_size

    def _init_class(self):
        get_boxes_num = GetBoxesNom(
            self.cont, self.data_type, self.ub_size)
        merge_sort = MergeSort(self.cont, self.data_type,
                               self.ub_size)
        nms = NonMaximumSuppression(
            self.cont, self.data_type, self.ub_size)
        nms.set_def_data(
            self.boxes_def_num, self.scores_def_num, self.label_def_num)
        return get_boxes_num, merge_sort, nms

    def mode1_compute(self):
        batch_num_all = self.batch_num * self.kind_num
        if (self.data_type == "float16" and
                (self.max_output_size == 1 or self.max_output_size == 3)):
            self.ai_core_use = 1
        each_core_batch_num = self.method.ceil_div(
            batch_num_all, self.ai_core_use)
        self.ai_core_use, last_core_batch_num = self.method.get_loop_info(
            batch_num_all, each_core_batch_num)
        with self.tik_inst.for_range(0, self.ai_core_use,
                                     block_num=self.ai_core_use) as core_index:
            batch_index = each_core_batch_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                self._mode1_compute_each_core(batch_index, each_core_batch_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_core(batch_index, last_core_batch_num)
        self.tik_inst.BuildCCE(
            inputs=[self.boxes, self.scores],
            outputs=[self.proposal_selected],
            kernel_name=self.kernel_name)

    def _mode1_compute_each_core(self, batch_index_start, batch_num):
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self._mode1_compute_each_loop(batch_index_start + batch_index)

    def _score_move_in_fun(self, scores_ub, start_index, data_num,
                           batch_index, kind_index):
        block_num = self.method.ceil_div(data_num, self.block_data_num)
        self.tik_inst.data_move(
            scores_ub, self.scores[batch_index, 0, kind_index, start_index],
            0, 1, block_num, 0, 0)

    def _mode1_compute_each_loop(self, batch_index):
        kind_index = batch_index % self.kind_num
        batch_index = batch_index // self.kind_num
        num_max = self.get_boxes_num.get_boxes_num(
            self.boxes_num, self.score_threshold, self.neg_fp16_inf,
            self.fp16_inf, self._score_move_in_fun,
            batch_index, kind_index)
        with self.tik_inst.if_scope(num_max > 0):
            top_proposal_l1 = self.tik_inst.Tensor(
                self.data_type, (1, self.ub_pro_num_max, self.pro_data_num),
                self.tik.scope_cbuf, "top_proposal_l1")
            self.merge_sort.get_top_proposal(
                top_proposal_l1, 0, self.boxes_num, self.sort_num,
                self._get_ub_proposal, batch_index, kind_index)
            with self.tik_inst.if_scope(num_max > self.sort_num):
                num_max.set_as(self.sort_num)
            select_pro_ub, select_num_scalar = \
                self.nms.non_maximum_suppression(
                    top_proposal_l1, 0, num_max, self.max_output_size,
                    self.iou_threshold, 0.0, self.weights, self.biases)
            self._proposal_move_out(select_pro_ub, batch_index, kind_index)

        with self.tik_inst.else_scope():
            select_proposal_ub = self.tik_inst.Tensor(
                self.data_type, (self.max_output_size_format, 8),
                self.tik.scope_ubuf, "select_proposal_ub")
            temp_ub = self.tik_inst.Tensor(
                self.data_type, (self.max_output_size_format, ),
                self.tik.scope_ubuf, "temp_ub")
            self.nms.init_pro_data(select_proposal_ub, temp_ub,
                                   self.max_output_size_format)
            self._proposal_move_out(select_proposal_ub, batch_index,
                                    kind_index)

    def _get_ub_proposal(self, boxes_num, start_index,
                         batch_index, kind_index):
        proposal_num = self.method.get_align_num(
            boxes_num, self.pro_repeat_num)
        ub_proposal_1 = self.tik_inst.Tensor(
            self.data_type, (proposal_num, self.pro_data_num),
            self.tik.scope_ubuf, "ub_proposal_1")
        with self.tik_inst.new_stmt_scope():
            proposal_data_ub = self.tik_inst.Tensor(
                self.data_type, (proposal_num,),
                self.tik.scope_ubuf, "proposal_data_ub")
            self._group_boxes_scores(ub_proposal_1, proposal_data_ub,
                                     batch_index, kind_index, start_index,
                                     boxes_num, proposal_num)
        ub_proposal_2 = self.tik_inst.Tensor(
            self.data_type, (proposal_num, self.pro_data_num),
            self.tik.scope_ubuf, "ub_proposal_2")
        return ub_proposal_1, ub_proposal_2

    def _group_boxes_scores(self, ub_proposal, proposal_data_ub, batch_index,
                            kind_index, start_index, boxes_num, proposal_num):
        block_move_in = self.method.ceil_div(boxes_num, self.block_data_num)
        mask_h, mask_l, index_last = self.method.get_mask(
            boxes_num, self.repeat_data_num, self.block_data_num)
        # boxes move in
        self.method.vector_dup(proposal_data_ub, self.boxes_def_num)
        with self.tik_inst.for_range(0, 4) as index_channel:
            self.tik_inst.data_move(
                proposal_data_ub,
                self.boxes[batch_index, index_channel,
                           kind_index, start_index],
                0, 1, block_move_in, 0, 0)
            if mask_h != 0 or mask_l != 0:
                self.tik_inst.vector_dup(
                    [mask_h, mask_l], proposal_data_ub[index_last],
                    self.boxes_def_num, 1, 1, 8)
            self.method.vector_concat(ub_proposal, proposal_data_ub,
                                      index_channel, proposal_num)
        # scores move in
        self.method.vector_dup(proposal_data_ub, self.scores_def_num)
        self.tik_inst.data_move(
            proposal_data_ub,
            self.scores[batch_index, 0, kind_index, start_index],
            0, 1, block_move_in, 0, 0)
        if mask_h != 0 or mask_l != 0:
            self.tik_inst.vector_dup(
                [mask_h, mask_l], proposal_data_ub[index_last],
                self.scores_def_num, 1, 1, 8)
        self.method.vector_concat(ub_proposal, proposal_data_ub,
                                  4, proposal_num)
        # label move in
        self.method.vector_dup(proposal_data_ub, self.label_def_num)
        self.method.vector_dup(proposal_data_ub,
                               kind_index + self.kind_start_index)
        self.method.vector_concat(ub_proposal, proposal_data_ub,
                                  5, proposal_num)

    def _proposal_move_out(self, select_proposal_ub, batch_index, kind_index):
        last_output_num = self.max_output_size % self.block_pro_num
        if last_output_num == 0:
            output_block_num = self.max_output_size // self.block_pro_num
            self.tik_inst.data_move(
                self.proposal_selected[batch_index, kind_index, 0, 0],
                select_proposal_ub, 0, 1, output_block_num, 0, 0)
        elif self.ai_core_use == 1:
            output_block_num = self.method.ceil_div(self.max_output_size,
                                                    self.block_pro_num)
            self.tik_inst.data_move(
                self.proposal_selected[batch_index, kind_index, 0, 0],
                select_proposal_ub, 0, 1, output_block_num, 0, 0)
        else:
            last_proposal = last_output_num + self.block_pro_num
            last_output_index_start = self.max_output_size - last_proposal
            self.tik_inst.data_move(
                self.proposal_selected[batch_index, kind_index, 0, 0],
                select_proposal_ub[last_output_index_start, 0],
                0, 1, 2, 0, 0)

            output_block_num = self.max_output_size // self.block_pro_num
            align_proposal_num = output_block_num * self.block_pro_num
            self.tik_inst.data_move(
                select_proposal_ub[align_proposal_num, 0],
                self.proposal_selected[batch_index, kind_index,
                                       last_output_num, 0],
                0, 1, 1, 0, 0)
            self.tik_inst.data_move(
                self.proposal_selected[batch_index, kind_index, 0, 0],
                select_proposal_ub, 0, 1, output_block_num, 0, 0)
            gm_index_last = self.max_output_size - self.block_pro_num
            self.tik_inst.data_move(
                self.proposal_selected[batch_index, kind_index,
                                       gm_index_last, 0],
                select_proposal_ub[align_proposal_num, 0], 0, 1, 1, 0, 0)

    def tik_output_debug(self):
        return self.tik_inst


def batch_multi_class_nms(boxes, scores, proposal_selected,
                          max_output_size, iou_threshold,
                          score_threshold, kind_start_index,
                          sort_num, weights, biases,
                          kernel_name="BMCNMS", test=False):
    obj = BatchMultiClassNonMaxSuppression(
        boxes, scores, proposal_selected, max_output_size, iou_threshold,
        score_threshold, kind_start_index, sort_num, weights, biases,
        kernel_name)
    obj.mode1_compute()
    if not test:
        return 0
    obj.tik_output_debug()
    return 0
