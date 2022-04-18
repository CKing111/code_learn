from te import tik


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor           #'//':整数除法。
    return result


class OneCoreNMS():
    def __init__(self, tik_instance, nms_param):            #声明输入
        self.total_input_proposal_num = nms_param[0]    #输出总数
        self.total_output_proposal_num = nms_param[1]   #输出总数
        self.burst_input_proposal_num = nms_param[2]    #
        self.down_factor = nms_param[3]
        self.tik_instance = tik_instance
        # variables                                         #用tik语言声明变量（scalar标量）
        self.selected_proposals = self.tik_instance.Scalar("uint16")
        self.selected_proposals.set_as(0)                               #.set_as : tik的tensor设置,初始化参数为0
        self.handling_proposals = self.tik_instance.Scalar("uint16")
        self.handling_proposals.set_as(0)
        self.handled_proposal_cnt = self.tik_instance.Scalar("uint16")
        self.handled_proposal_cnt.set_as(0)
        self.total_proposal_num = self.tik_instance.Scalar("int32")
        self.total_proposal_num.set_as(
            ceil_div(self.total_input_proposal_num, 16) * 16)
        self.ask_for_ub()
        
    def ask_for_ub(self):
        out_num_align_up = ceil_div(self.total_output_proposal_num, 16) * 16            #向上对齐输出数量标量
        

        #sel-* 为要去计算iou值的目标边界框
        #temp-* 为与sel对比的目标

        #声明tik的Tensor，减少ub
        self.sel_reduced_proposals_ub = self.tik_instance.Tensor(                      
                                                "float16",
                                                [out_num_align_up, 8],           #shape
                                                name="selReducedProposals_ub",
                                                scope=tik.scope_ubuf)            #tensor类型，目前只支持scope_ubuf
        
        #将一个Scalar变量或一个立即数，复制多次并填充到向量
        #(参与元素128个bits，输入张量，，被复制的源操作数，迭代次数，步长（B）)
        self.tik_instance.vector_dup(128, self.sel_reduced_proposals_ub, 0,
                                     out_num_align_up // 16, 1, 8)

        #声明范围张量
        self.sel_area_ub = self.tik_instance.Tensor("float16",
                                                   [out_num_align_up],
                                                   name="selArea_ub",
                                                   scope=tik.scope_ubuf)
        #填充操作
        self.tik_instance.vector_dup(16, self.sel_area_ub, 0,
                                     out_num_align_up // 16, 1, 1)
        #声明张量（子向量）
        self.sup_vec_ub = self.tik_instance.Tensor("uint16", [
            ceil_div(self.total_output_proposal_num, 128) * 128],
                                                  name="supVec_ub",
                                                  scope=tik.scope_ubuf)
        # all suppressed
        self.tik_instance.vector_dup(128, self.sup_vec_ub[0], 1,
                                     self.sup_vec_ub.shape[0] // 128, 1, 8)
        # change with burst
        self.temp_reduced_proposals = self.tik_instance.Tensor("float16", [
                                            self.burst_input_proposal_num, 8],
                                            name="tempReducedProposals",
                                            scope=tik.scope_ubuf)
        self.temp_area_ub = self.tik_instance.Tensor("float16", [
            self.burst_input_proposal_num], name="tempArea_ub",
                                                    scope=tik.scope_ubuf)
        self.temp_iou_ub = self.tik_instance.Tensor("float16", [
            out_num_align_up + self.burst_input_proposal_num, 16],
                                                   name="tempIoU_ub",
                                                   scope=tik.scope_ubuf)
        self.temp_join_ub = self.tik_instance.Tensor("float16", [
            out_num_align_up + self.burst_input_proposal_num, 16],
                                                    name="tempJoin_ub",
                                                    scope=tik.scope_ubuf)
        self.temp_supmatrix_ub = self.tik_instance.Tensor("uint16", [
            out_num_align_up + self.burst_input_proposal_num],
                                                    name="tempSupMatrix_ub",
                                                    scope=tik.scope_ubuf)
        self.temp_supvec_ub = self.tik_instance.Tensor("uint16", [
            ceil_div(self.burst_input_proposal_num, 128) * 128],
                                                      name="tempSupVec_ub",
                                                      scope=tik.scope_ubuf)

    def get_reduced_proposal(self, out_proposal, in_proposal):
        coord_addr = self.tik_instance.Tensor("float16", [4, ceil_div(
            out_proposal.shape[0], 128) * 128],
                                             name="coord_addr",
                                             scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(128, coord_addr, 0.0,
                                     coord_addr.shape[1] // 128, 1, 8)
        self.tik_instance.vextract(coord_addr[0], in_proposal[0],
                                   in_proposal.shape[0] // 16, 0)  # x1
        self.tik_instance.vextract(coord_addr[coord_addr.shape[1] * 1],
                                   in_proposal[0], in_proposal.shape[0] // 16,
                                   1)  # y1
        self.tik_instance.vextract(coord_addr[coord_addr.shape[1] * 2],
                                   in_proposal[0], in_proposal.shape[0] // 16,
                                   2)  # x2
        self.tik_instance.vextract(coord_addr[coord_addr.shape[1] * 3],
                                   in_proposal[0], in_proposal.shape[0] // 16,
                                   3)  # y2
        self.tik_instance.vmuls(128, coord_addr[0], coord_addr[0],
                                self.down_factor, coord_addr.shape[1] // 128,
                                1, 1, 8, 8)  # x1*downFactor
        self.tik_instance.vmuls(128, coord_addr[coord_addr.shape[1] * 1],
                                coord_addr[coord_addr.shape[1] * 1],
                                self.down_factor, coord_addr.shape[1] // 128,
                                1, 1, 8, 8)  # y1*downFactor
        self.tik_instance.vmuls(128, coord_addr[coord_addr.shape[1] * 2],
                                coord_addr[coord_addr.shape[1] * 2],
                                self.down_factor, coord_addr.shape[1] // 128,
                                1, 1, 8, 8)  # x2*downFactor
        self.tik_instance.vmuls(128, coord_addr[coord_addr.shape[1] * 3],
                                coord_addr[coord_addr.shape[1] * 3],
                                self.down_factor, coord_addr.shape[1] // 128, 
                                1, 1, 8, 8)  # y2*downFactor
        self.tik_instance.vconcat(out_proposal[0], coord_addr[0],
                                  out_proposal.shape[0] // 16, 0)  # x1
        self.tik_instance.vconcat(out_proposal[0],
                                  coord_addr[coord_addr.shape[1] * 1],
                                  out_proposal.shape[0] // 16, 1)  # y1
        self.tik_instance.vconcat(out_proposal[0],
                                  coord_addr[coord_addr.shape[1] * 2],
                                  out_proposal.shape[0] // 16, 2)  # x2
        self.tik_instance.vconcat(out_proposal[0],
                                  coord_addr[coord_addr.shape[1] * 3],
                                  out_proposal.shape[0] // 16, 3)  # y2

#流程三

    def nms_single_core_subprocess_three(self):
        with self.tik_instance.for_range(
                                        0, self.handling_proposals) as i:       #历遍需要处理的
            with self.tik_instance.if_scope(                                    #条件判断
                    tik.all(self.temp_supvec_ub[i] == 0,    #临时假设为0
                        self.selected_proposals <           #目标数小于输出总数
                        self.total_output_proposal_num,
                        i + self.handled_proposal_cnt <     #处理数小于总输入
                        self.total_input_proposal_num)):
                # update selReducedProposals_ub
                #将第i个需要处理handling_proposals的临时缩减目标temp_reduced_proposals的第j行数据写入张量sel_reduced_proposals_ub
                with self.tik_instance.for_range(0, 4) as j:    
                    self.sel_reduced_proposals_ub[
                        self.selected_proposals, j].set_as(
                        self.temp_reduced_proposals[i, j])
                # update selArea_ub
                self.sel_area_ub[self.selected_proposals].set_as(
                    self.temp_area_ub[i])
                # update supVec_ub
                self.sup_vec_ub[self.selected_proposals].set_as(
                    self.temp_supvec_ub[i])
                # update counter
                self.selected_proposals.set_as(
                    self.selected_proposals + 1)

#流程二

    def nms_single_core_subprocess_two(self, length, threshold):
        with self.tik_instance.for_range(0, ceil_div(
                            self.handling_proposals, 16)) as i: #历遍处理项
            length.set_as(length + 16)      #修改长度张量
            
            #计算sel和历遍temp的iou值
            # calculate intersection of tempReducedProposals
            # and selReducedProposals
            self.tik_instance.viou(
                                self.temp_iou_ub[0, 0],
                                self.sel_reduced_proposals_ub[0],
                                self.temp_reduced_proposals[
                                    i * 16, 0],
                                ceil_div(self.selected_proposals,
                                        16))
            # calculate intersection of tempReducedProposals
            # and tempReducedProposals(include itself)
            self.tik_instance.viou(self.temp_iou_ub[ceil_div(
                self.selected_proposals, 16) * 16, 0],
                                    self.temp_reduced_proposals[0],
                                    self.temp_reduced_proposals[
                                        i * 16, 0], i + 1)
            # calculate join of tempReducedProposals
            # and selReducedProposals
            self.tik_instance.vaadd(self.temp_join_ub[0, 0],
                                    self.sel_area_ub[0],
                                    self.temp_area_ub[i * 16],
                                    ceil_div(
                                        self.selected_proposals, 16))
            # calculate intersection of tempReducedProposals
            # and tempReducedProposals(include itself)
            self.tik_instance.vaadd(self.temp_join_ub[ceil_div(
                self.selected_proposals, 16) * 16, 0],
                                    self.temp_area_ub,
                                    self.temp_area_ub[i * 16], i + 1)
            # calculate join*(thresh/(1+thresh))
            self.tik_instance.vmuls(128, self.temp_join_ub[0, 0],
                                    self.temp_join_ub[0, 0],
                                    threshold / (1 + threshold),
                                    ceil_div(length, 8), 1, 1, 8, 8)
            # compare and generate suppression matrix
            self.tik_instance.vcmpv_gt(self.temp_supmatrix_ub[0],
                                        self.temp_iou_ub[0, 0],
                                        self.temp_join_ub[0, 0],
                                        ceil_div(length, 8), 1, 1,
                                        8, 8)
            # generate suppression vector
            # clear rpn_cor_ir
            rpn_cor_ir = self.tik_instance.set_rpn_cor_ir(0)
            # non-diagonal
            rpn_cor_ir = self.tik_instance.rpn_cor(
                self.temp_supmatrix_ub[0], self.sup_vec_ub[0], 
                1, 1, ceil_div(self.selected_proposals, 16))
            with self.tik_instance.if_scope(i > 0):
                rpn_cor_ir = self.tik_instance.rpn_cor(
                    self.temp_supmatrix_ub[
                        ceil_div(self.selected_proposals, 16) *
                        16], self.temp_supvec_ub[0], 1, 1, i)
            # diagonal
            self.tik_instance.rpn_cor_diag(
                self.temp_supvec_ub[i * 16],
                self.temp_supmatrix_ub[length - 16], rpn_cor_ir)


#流程一

    def nms_single_core_subprocess_one(self, proposals, threshold, sup_vector):
        open_thread = 2
        if ceil_div(self.total_input_proposal_num, 
                    self.burst_input_proposal_num) == 1:
            open_thread = 1
        with self.tik_instance.for_range(0, ceil_div(
                self.total_input_proposal_num, self.burst_input_proposal_num),
                    thread_num=open_thread) as burst_index:
            fresh_proposals_ub = self.tik_instance.Tensor("float16",
                [ceil_div(self.burst_input_proposal_num, 16) * 16, 8],
                name="fresh_proposals_ub", scope=tik.scope_ubuf)
            # update counter
            with self.tik_instance.if_scope(
                    self.handled_proposal_cnt < self.total_proposal_num -
                    self.burst_input_proposal_num):
                self.handling_proposals.set_as(self.burst_input_proposal_num)
            with self.tik_instance.else_scope():
                self.handling_proposals.set_as(
                    self.total_proposal_num - self.handled_proposal_cnt)

            self.tik_instance.data_move(fresh_proposals_ub[0], proposals[
                burst_index * self.burst_input_proposal_num, 0], 0, 1,
                                        ceil_div(self.handling_proposals * 16,
                                                 32), 0, 0, 0)
            # clear tempSupVec_ub, all suppressed
            self.tik_instance.vector_dup(128, self.temp_supvec_ub[0], 1,
                                         self.temp_supvec_ub.shape[0] // 128, 
                                         1, 8)
            self.tik_instance.vector_dup(128, self.temp_reduced_proposals[0], 
                                        0, self.temp_reduced_proposals.shape[
                                             0] // 16, 1, 8)
            # reduce fresh proposal
            with self.tik_instance.new_stmt_scope():
                self.get_reduced_proposal(self.temp_reduced_proposals,
                                          fresh_proposals_ub)
            # calculate the area of reduced-proposal
            self.tik_instance.vrpac(self.temp_area_ub[0],
                                    self.temp_reduced_proposals[0],
                                    ceil_div(self.handling_proposals, 16))
            # start to update iou and or area from the first 16 proposal
            # and get suppression vector 16 by 16 proposal
            length = self.tik_instance.Scalar("uint16")
            length.set_as(ceil_div(self.selected_proposals, 16) * 16)
            with self.tik_instance.if_scope(
                    self.selected_proposals < self.total_output_proposal_num):
                with self.tik_instance.new_scope():
                    self.nms_single_core_subprocess_two(length, threshold)

                    # move suppresion vector out
                    self.tik_instance.tensor_mov(
                        sup_vector[
                            burst_index * self.burst_input_proposal_num],
                        self.temp_supvec_ub, "", 1,
                        ceil_div(self.burst_input_proposal_num, 16), 0, 0)
                    
                    # find & mov unsuppressed proposals
                    self.nms_single_core_subprocess_three()
                self.handled_proposal_cnt.set_as(
                    self.handled_proposal_cnt + self.handling_proposals)

    def nms_single_core(self, proposals, threshold):
        temp_t = self.tik_instance.Scalar("uint16")
        temp_t.set_as(0)
        self.sup_vec_ub[0].set_as(temp_t)

        padding = ceil_div(self.burst_input_proposal_num,
                           16) * 16 - self.burst_input_proposal_num
        sup_vector = self.tik_instance.Tensor(
            "uint16", (
                ceil_div(
                    proposals.shape[0],
                    self.burst_input_proposal_num) *
                self.burst_input_proposal_num + padding,),
            name="sup_vector",
            scope=tik.scope_ubuf)
        self.nms_single_core_subprocess_one(proposals, threshold, sup_vector)
        return sup_vector
