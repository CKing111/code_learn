from uti import interface_check
from version import get_version
from non_max_suppression.common_method import CommonMethod
from non_max_suppression.merge_sort import MergeSort


class MergeProposal(object):

    def __init__(self, proposal, box_select, kernel_name):
        """
        Args:
            proposal:
                dictionary("shape": (batch_num, kind_num, boxes_num, 8),
                "dtype": "float16", "format": "NCHW");
            box_select:
                dictionary("shape": (batch_num, 6, 1, boxes_num),
                "dtype": "float16", "format": "NCHW");
            kernel_name: str
        """
        interface_check.check_kernelname(kernel_name)
        self._check_param(proposal, box_select)
        self._check_param_shape(proposal, box_select)

        self.cont = get_version.get_aicore_container(("Ascend610",))
        self.tik, self.tik_inst = self.cont.tik, self.cont.tinst
        self.method = CommonMethod(self.cont)

        self.data_type = proposal.get("dtype")
        self.data_size, self.block_data_num, self.repeat_data_num = \
            self.method.get_type_const(self.data_type)
        self.block_pro_num = self.method.get_block_pro_num(self.data_type)

        self.ub_size = self.cont.const_ub_max_byte
        self.l1_size = self.cont.const_l1_max_byte
        self.repeat_time_max = self.cont.const_vector_proc_max_rpt
        self.ai_core_use = self.cont.const_aicore_num
        self.pro_data_num = self.cont.const_proposal_data_num
        self.pro_repeat_num = self.cont.const_proposal_repeat_num

        self.merge_sort = MergeSort(self.cont, self.data_type, self.ub_size)

        self.batch_num, self.kind_num, self.box_num = proposal.get("shape")[:3]
        self.proposal_num = self.method.get_align_num(self.box_num,
                                                      self.pro_repeat_num)
        (self.ub_pro_num_max, self.sort_min_num) = \
            self.merge_sort.get_pro_num_info()

        self.kernel_name = kernel_name
        self.proposal = self.tik_inst.Tensor(
            self.data_type,
            (self.batch_num, self.kind_num, self.box_num * self.pro_data_num),
            self.tik.scope_gm, "proposal")
        self.box_select = self.tik_inst.Tensor(
            self.data_type, (self.batch_num, 6, self.box_num),
            self.tik.scope_gm, "box_select")

    @staticmethod
    def _check_param(proposal, box_select):
        interface_check.check_param(proposal, [4], ["float16"],
                                    ["NCHW", "ND"])
        interface_check.check_param(box_select, [4], ["float16"],
                                    ["NCHW", "ND"])

    @staticmethod
    def _check_param_shape(proposal, box_select):
        proposal_shape = proposal.get("shape")
        boxes_selected_shape = box_select.get("shape")
        batch_num, kind_num, boxes_num = proposal_shape[:3]
        max_boxes_num = 1024
        kind_max_num = 2048
        if boxes_num > max_boxes_num:
            raise RuntimeError("boxes num is too larger")
        if kind_num > kind_max_num:
            raise RuntimeError("kind num is too larger")
        if boxes_selected_shape != (batch_num, 6, 1, boxes_num):
            raise RuntimeError("box_select shape is not supported")
        gm_size = ((batch_num * kind_num * boxes_num * 8 +
                    batch_num * boxes_num * 6) * 2) / 0.8
        if gm_size > 1024 ** 3:
            raise RuntimeError("data size is too large")

    def mode1_compute(self):
        if self.box_num < self.block_data_num:
            self.ai_core_use = 1
        each_core_batch_num = self.method.ceil_div(
            self.batch_num, self.ai_core_use)
        self.ai_core_use, last_core_batch_num = self.method.get_loop_info(
            self.batch_num, each_core_batch_num)
        with self.tik_inst.for_range(
                0, self.ai_core_use, block_num=self.ai_core_use) as core_index:
            batch_index = each_core_batch_num * core_index
            with self.tik_inst.if_scope(core_index != self.ai_core_use - 1):
                self._mode1_compute_each_core(batch_index, each_core_batch_num)
            with self.tik_inst.else_scope():
                self._mode1_compute_each_core(batch_index, last_core_batch_num)
        self.tik_inst.BuildCCE(
            inputs=[self.proposal],
            outputs=[self.box_select],
            kernel_name=self.kernel_name)

    def _mode1_compute_each_core(self, batch_index_start, batch_num):
        with self.tik_inst.for_range(0, batch_num) as batch_index:
            self._mode1_compute_each_loop(batch_index_start + batch_index)

    def _mode1_compute_each_loop(self, batch_index):
        ub_proposal_1 = self.tik_inst.Tensor(
            self.data_type, (self.ub_pro_num_max, self.pro_data_num),
            self.tik.scope_ubuf, "ub_proposal_1")
        self._move_proposal_in(ub_proposal_1, 0, batch_index, 0)
        self._merge_proposal(ub_proposal_1, batch_index)
        self._result_move_out(ub_proposal_1, batch_index)

    def _result_move_out(self, ub_proposal_1, batch_index):
        # move boxes
        with self.tik_inst.new_stmt_scope():
            result_ub = self.tik_inst.Tensor(
                self.data_type, (self.proposal_num, ),
                self.tik.scope_ubuf, "result_ub")
            with self.tik_inst.for_range(0, 6) as mode_number:
                self.method.vector_extract(
                    result_ub, ub_proposal_1, mode_number, self.proposal_num)
                self._result_move_out_each_label(
                    result_ub, batch_index, mode_number)

    def _result_move_out_each_label(self, result_ub, batch_index, label_index):
        if self.box_num <= self.block_data_num:
            self.tik_inst.data_move(
                self.box_select[batch_index, label_index, 0],
                result_ub, 0, 1, 1, 0, 0)
        else:
            block_num = self.box_num // self.block_data_num
            self.tik_inst.data_move(
                self.box_select[batch_index, label_index, 0],
                result_ub, 0, 1, block_num, 0, 0)
            if self.box_num % self.block_data_num != 0:
                # data move last two block
                index_ub = (block_num - 1) * self.block_data_num
                last_num = self.box_num - index_ub
                self.tik_inst.data_move(
                    self.proposal[batch_index, 0, 0], result_ub[index_ub],
                    0, 1, 2, 0, 0)
                index_gm_last_block = last_num - self.block_data_num
                self.tik_inst.data_move(
                    result_ub,
                    self.proposal[batch_index, 0, index_gm_last_block],
                    0, 1, 1, 0, 0)
                index_gm = self.box_num - self.block_data_num
                self.tik_inst.data_move(
                    self.box_select[batch_index, label_index, index_gm],
                    result_ub, 0, 1, 1, 0, 0)

    def _merge_proposal(self, ub_proposal_1, batch_index):
        each_loop_kind_num = self.ub_pro_num_max // self.proposal_num - 1
        loop_times, last_loop_kind_num = self.method.get_loop_info(
            self.kind_num - 1, each_loop_kind_num)
        if loop_times != 0:
            with self.tik_inst.new_stmt_scope():
                ub_proposal_2 = self.tik_inst.Tensor(
                    self.data_type, (self.ub_pro_num_max, self.pro_data_num),
                    self.tik.scope_ubuf, "ub_proposal_2")
                with self.tik_inst.for_range(0, loop_times) as loop_index:
                    kind_index = each_loop_kind_num * loop_index + 1
                    with self.tik_inst.if_scope(loop_index != loop_times - 1):
                        self._merge_proposal_each_loop(
                            ub_proposal_1, ub_proposal_2, batch_index,
                            kind_index, each_loop_kind_num)
                    with self.tik_inst.else_scope():
                        self._merge_proposal_each_loop(
                            ub_proposal_1, ub_proposal_2, batch_index,
                            kind_index, last_loop_kind_num)

    def _merge_proposal_each_loop(self, ub_proposal_1, ub_proposal_2,
                                  batch_index, kind_index_start, kind_num):
        proposal_num_all = self.proposal_num * (1 + kind_num)
        # sort boxes
        with self.tik_inst.for_range(0, kind_num) as kind_index:
            self._move_proposal_in(
                ub_proposal_1, kind_index + 1,
                batch_index, kind_index + kind_index_start)
        ub_proposal_1_merged, ub_proposal_2_merged = \
            self.merge_sort.merge_sort_element(
                ub_proposal_1, ub_proposal_2,
                proposal_num_all, self.proposal_num)
        if ub_proposal_1_merged.name != ub_proposal_1.name:
            block_num_move = self.method.ceil_div(
                self.proposal_num, self.block_pro_num)
            self.tik_inst.data_move(ub_proposal_1, ub_proposal_2, 0,
                                    1, block_num_move, 0, 0)

    def _move_proposal_in(self, proposal_ub, ub_kind_index,
                          gm_batch_index, gm_kind_index):
        block_num_input = self.method.ceil_div(self.box_num,
                                               self.block_pro_num)
        self.tik_inst.data_move(
            proposal_ub[ub_kind_index * self.proposal_num, 0],
            self.proposal[gm_batch_index, gm_kind_index, 0],
            0, 1, block_num_input, 0, 0)

    def tik_output_debug(self):
        return self.tik_inst


def merge_proposal(proposal, box_select,
                   kernel_name="MergeProposal", test=False):
    obj = MergeProposal(proposal, box_select, kernel_name)
    obj.mode1_compute()
    if not test:
        return 0
    obj.tik_output_debug()
    return 0
