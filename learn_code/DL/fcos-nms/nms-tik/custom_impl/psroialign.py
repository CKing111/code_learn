# -*- coding: UTF-8 -*-

import os
from te import tik
import numpy as np
from util import OpLog as oplog
from coordinate import coordinate_cal


def four2five(tik_instance, i, j, index, data_in, data_out_cbuf, input_shape,
              group_height, group_width, hw_align,
              class_align, zero_ub, class_num, pooled_width):
    with tik_instance.for_range(0, class_align, thread_num=1) as c_i:
        src_ub = tik_instance.Tensor("float16", (1, 16, hw_align * 16),
                                     tik.scope_ubuf, "src_ub")
        dst_ub = tik_instance.Tensor("float16", (1, 1, hw_align * 16, 16),
                                     tik.scope_ubuf, "dst_ub")
        with tik_instance.for_range(0, 16) as c_idx:
            with tik_instance.if_scope((c_i * 16 + c_idx) < class_num):
                tik_instance.data_move(src_ub[0, c_idx, 0],
                    data_in[input_shape[1] * input_shape[2] *
                        input_shape[3] * index +
                        ((c_i * 16 + c_idx) * group_height *
                        group_width + i * pooled_width + j) *
                        input_shape[2] * input_shape[3]],
                    0, 1, hw_align, 0,
                    0)  # move data right or error
            with tik_instance.else_scope():
                tik_instance.data_move(src_ub[0, c_idx, 0], zero_ub, 0, 1,
                                       hw_align * 16 // 16, 0, 0)
        src_list = [src_ub[0, i0, 0] for i0 in range(16)]
        dst_list = [dst_ub[0, 0, i0, 0] for i0 in range(16)]
        tik_instance.vnchwconv(True, False, dst_list, src_list,
                               hw_align * 16 // 16, 16, 1)
        tik_instance.data_move(data_out_cbuf[0, c_i, 0, 0, 0], dst_ub, 0, 1,
                               input_shape[2] * input_shape[3], 0, 0)


def interp_value_cal(tik_instance, values, x_value, y_value, x_max, y_max,
                    min_value, weight, per_point_ub):
    with tik_instance.if_scope(
            tik.all(x_value >= min_value, x_value < x_max, 
                y_value >= min_value, y_value < y_max)):
        tik_instance.vmuls(16, values, 
                           per_point_ub[0, 0, y_value, x_value, 0], 
                           weight, 1,
                           1, 1, 1, 1)
    with tik_instance.else_scope():
        tik_instance.vmuls(16, values, 
                           per_point_ub[0, 0, y_value, x_value, 0], 
                           0, 1, 1, 1, 1, 1)


def psroialign_interp(tik_instance, sample_per_part, hstart_one_ubuf, i, j,
                      hweight_one_r_ubuf, hweight_one_l_ubuf,
                      wstart_one_ubuf, wweight_one_r_ubuf, wweight_one_l_ubuf,
                      pro, count, sum_value, per_point_ub,
                      input_shape, cnt, cnt_fp16, data_5_4_cbuf, count_fp16):
    values11 = tik_instance.Tensor("float16", (16,), scope=tik.scope_ubuf,
                                   name="values11")
    values12 = tik_instance.Tensor("float16", (16,), scope=tik.scope_ubuf,
                                   name="values12")
    values21 = tik_instance.Tensor("float16", (16,), scope=tik.scope_ubuf,
                                   name="values21")
    values22 = tik_instance.Tensor("float16", (16,), scope=tik.scope_ubuf,
                                   name="values22")
    hstart_scalar = tik_instance.Scalar("int32", name="hstart_scalar")
    wstart_scalar = tik_instance.Scalar("int32", name="wstart_scalar")
    weightw_l = tik_instance.Scalar("float16", name="weightw_l")
    weightw_r = tik_instance.Scalar("float16", name="weightw_r")
    weighth_l = tik_instance.Scalar("float16", name="weighth_l")
    weighth_r = tik_instance.Scalar("float16", name="weighth_r")
    
    psroialign_interp_process_one(
        tik_instance, sample_per_part, hstart_one_ubuf,
        i, j, hweight_one_r_ubuf, hweight_one_l_ubuf, wstart_one_ubuf,
        wweight_one_r_ubuf, wweight_one_l_ubuf, pro, count, sum_value, 
        per_point_ub, input_shape, values11, values12, values21, values22,
        hstart_scalar, wstart_scalar, weightw_l, weightw_r, weighth_l,
        weighth_r)

    with tik_instance.if_scope(count == 0):
        tik_instance.vector_dup(16, sum_value, 0, 1, 1, 1)
        tik_instance.data_move(data_5_4_cbuf[pro, 0, i, j, 0], sum_value, 
                                0, 1, 1, 0, 0)
    with tik_instance.else_scope():
        cnt[0].set_as(count)
        tik_instance.vconv(1, '', cnt_fp16, cnt, 1, 1, 1, 1, 1)
        tik_instance.vconv(1, '', cnt_fp16, cnt, 1, 1, 1, 1, 1)
        count_fp16.set_as(cnt_fp16[0])
        tik_instance.vector_dup(16, cnt_fp16, count_fp16, 1, 1, 1)
        tik_instance.vrec(16, cnt_fp16, cnt_fp16, 1, 1, 1, 1, 1)
        tik_instance.vmul(16, sum_value, sum_value, cnt_fp16, 
                        1, 1, 1, 1, 1, 1, 1)
        tik_instance.data_move(data_5_4_cbuf[pro, 0, i, j, 0], sum_value, 
                                0, 1, 1, 0, 0)


def psroialign_interp_process_one(
        tik_instance, sample_per_part, hstart_one_ubuf,
        i, j, hweight_one_r_ubuf, hweight_one_l_ubuf, wstart_one_ubuf,
        wweight_one_r_ubuf, wweight_one_l_ubuf, pro, count, sum_value, 
        per_point_ub, input_shape, values11, values12, values21, values22, 
        hstart_scalar, wstart_scalar, weightw_l, weightw_r, weighth_l,
        weighth_r):
    with tik_instance.for_range(0, sample_per_part) as h_smp:
        hstart_scalar.set_as(hstart_one_ubuf[h_smp, i])
        with tik_instance.if_scope(
                tik.all(hstart_scalar > -1, hstart_scalar < input_shape[2])):
            weighth_r.set_as(hweight_one_r_ubuf[pro, h_smp])
            weighth_l.set_as(hweight_one_l_ubuf[pro, h_smp])

            psroialign_interp_process_two(
                tik_instance, sample_per_part, j, wstart_one_ubuf,
                wweight_one_r_ubuf, wweight_one_l_ubuf, pro, count,
                sum_value, per_point_ub, input_shape, values11, 
                values12, values21, values22, hstart_scalar,
                wstart_scalar, weightw_l, weightw_r,
                weighth_l, weighth_r)


def psroialign_interp_process_two(
    tik_instance, sample_per_part, j, wstart_one_ubuf,
                wweight_one_r_ubuf, wweight_one_l_ubuf, pro, count,
                sum_value, per_point_ub, input_shape, values11, 
                values12, values21, values22, hstart_scalar,
                wstart_scalar, weightw_l, weightw_r,
                weighth_l, weighth_r):
    with tik_instance.for_range(0, sample_per_part) as w_smp:
        wstart_scalar.set_as(wstart_one_ubuf[w_smp, j])
        weightw_r.set_as(wweight_one_r_ubuf[pro, w_smp])
        weightw_l.set_as(wweight_one_l_ubuf[pro, w_smp])
        with tik_instance.if_scope(tik.all(wstart_scalar > -1,
                                            wstart_scalar <
                                            input_shape[3])):
            count.set_as(count + 1)
            tik_instance.vmuls(16, values11, per_point_ub[
                0, 0, hstart_scalar, wstart_scalar, 0], weightw_r, 1,
                                1, 1, 1, 1)
            interp_value_cal(tik_instance, values12,
                                wstart_scalar + 1, hstart_scalar,
                                input_shape[3],
                                input_shape[2], 0, weightw_l,
                                per_point_ub)
            interp_value_cal(tik_instance, values21, wstart_scalar,
                                hstart_scalar + 1, input_shape[3],
                                input_shape[2], 0, weightw_r,
                                per_point_ub)
            interp_value_cal(tik_instance, values22,
                                wstart_scalar + 1, hstart_scalar + 1,
                                input_shape[3],
                                input_shape[2], 0, weightw_l,
                                per_point_ub)
            tik_instance.vadd(16, values11, values11, values12, 1, 1,
                                1, 1, 1, 1, 1)
            tik_instance.vadd(16, values21, values21, values22, 1, 1,
                                1, 1, 1, 1, 1)
            tik_instance.vmuls(16, values11, values11, weighth_r, 1,
                                1, 1, 1, 1)
            tik_instance.vmuls(16, values21, values21, weighth_l, 1,
                                1, 1, 1, 1)
            tik_instance.vadd(16, values11, values11, values21, 1, 1,
                                1, 1, 1, 1, 1)
            tik_instance.vadd(16, sum_value, sum_value, values11, 1, 
                                1, 1, 1, 1, 1, 1)
    

def psroialign_interp_perbatch(tik_instance, input_shape, class_align,
                               data_out_cbuf, sample_per_part,
                               pooled_width_align16,
                               rois_num, wstart_one_cbuf, hstart_one_cbuf,
                               hweight_one_r_ubuf, hweight_one_l_ubuf,
                               data_5_4_cbuf,
                               wweight_one_r_ubuf, wweight_one_l_ubuf, i, j):
    with tik_instance.new_stmt_scope():
        per_point_ub = tik_instance.Tensor("float16", (
                                           input_shape[0], class_align,
                                           input_shape[2],
                                           input_shape[3], 16),
                                           tik.scope_ubuf,
                                           "per_point_ub")  # 32k
        tik_instance.data_move(per_point_ub, data_out_cbuf, 0, 1,
                               input_shape[2] * input_shape[3] * class_align,
                               0, 0)
        wstart_one_ubuf = tik_instance.Tensor("int32", (
                                              sample_per_part,
                                              pooled_width_align16 * 16),
                                              scope=tik.scope_ubuf,
                                              name="wstart_one_ubuf")
        hstart_one_ubuf = tik_instance.Tensor("int32", (
                                              sample_per_part,
                                              pooled_width_align16 * 16),
                                              scope=tik.scope_ubuf,
                                              name="hstart_one_ubuf")
        with tik_instance.for_range(0, rois_num) as pro:
            with tik_instance.for_range(0, sample_per_part) as ss_idx:
                tik_instance.data_move(wstart_one_ubuf[ss_idx, 0],
                                       wstart_one_cbuf[ss_idx, pro, 0], 
                                       0, 1, 1, 1, 1)
                tik_instance.data_move(hstart_one_ubuf[ss_idx, 0],
                                       hstart_one_cbuf[ss_idx, pro, 0], 
                                       0, 1, 1, 1, 1)
            count = tik_instance.Scalar("uint8", name="count")
            count_fp16 = tik_instance.Scalar("float16", name="count_fp16")
            cnt = tik_instance.Tensor("uint8", (16,), scope=tik.scope_ubuf,
                                      name="cnt")
            cnt_fp16 = tik_instance.Tensor("float16", (16,),
                                           scope=tik.scope_ubuf, name="cnt")
            count.set_as(0)
            sum_value = tik_instance.Tensor("float16", (16,), 
                                            scope=tik.scope_ubuf,
                                            name="sum_value")
            tik_instance.vector_dup(16, sum_value, 0, 1, 1, 1)
            psroialign_interp(tik_instance, sample_per_part, hstart_one_ubuf,
                              i, j, hweight_one_r_ubuf,
                              hweight_one_l_ubuf,
                              wstart_one_ubuf, wweight_one_r_ubuf,
                              wweight_one_l_ubuf, pro, count, sum_value,
                              per_point_ub,
                              input_shape, cnt, cnt_fp16, data_5_4_cbuf,
                              count_fp16)


def psroialign_four2five(tik_instance, input_shape, index, pooled_width,
                         pooled_height, data_in, data_out_cbuf,
                         group_height, group_width, class_align,
                         sample_per_part, pooled_width_align16, rois_num,
                         wstart_one_cbuf,
                         hstart_one_cbuf, hweight_one_r_ubuf,
                         hweight_one_l_ubuf, data_5_4_cbuf, class_num,
                         wweight_one_r_ubuf, wweight_one_l_ubuf):
    hw_align = (input_shape[2] * input_shape[3] + 15) // 16
    zero_ub = tik_instance.Tensor("float16", (hw_align * 16,),
                                  scope=tik.scope_ubuf, name="zero_ub")  # 2k
    tik_instance.vector_dup(128, zero_ub, 0, hw_align // 8, 1, 8)
    with tik_instance.for_range(0, pooled_height) as i:
        with tik_instance.for_range(0, pooled_width) as j:
            with tik_instance.new_stmt_scope():
                four2five(tik_instance, i, j, index,
                          data_in, data_out_cbuf, input_shape, group_height,
                          group_width, hw_align, class_align,
                          zero_ub, class_num, pooled_width)
            psroialign_interp_perbatch(tik_instance, input_shape, class_align,
                                       data_out_cbuf, sample_per_part,
                                       pooled_width_align16,
                                       rois_num, wstart_one_cbuf,
                                       hstart_one_cbuf, hweight_one_r_ubuf,
                                       hweight_one_l_ubuf, data_5_4_cbuf,
                                       wweight_one_r_ubuf, wweight_one_l_ubuf,
                                       i, j)


def psroialign_all(tik_instance, sample_per_part, max_roi_num,
                   pooled_width_align16, sample_per_part_align16,
                   pooled_height_align16, input_shape, class_align,
                   pooled_height, pooled_width, index, roi_align16, data_in,
                   group_height, group_width, roi_shape, heat_map_a,
                   class_num, rois_gm, roi_align2, rois_num, data_5_4_cbuf):
    wstart_one_cbuf = tik_instance.Tensor(
        "int32", (sample_per_part, max_roi_num, pooled_width_align16 * 16),
        scope=tik.scope_cbuf, name="wstart_one_cbuf")
    # shape of wweight_one_l_cbuf is (4,300,16)
    wweight_one_l_ubuf = tik_instance.Tensor(
        "float16", (max_roi_num, sample_per_part_align16 * 16),
        scope=tik.scope_ubuf, name="wweight_one_l_ubuf")
    # shape of wweight_one_r_cbuf is (4,300,16)
    wweight_one_r_ubuf = tik_instance.Tensor(
        "float16", (max_roi_num, sample_per_part_align16 * 16),
        scope=tik.scope_ubuf, name="wweight_one_r_ubuf")
    # shape of hstart_one_cbuf is (4,300,16)
    hstart_one_cbuf = tik_instance.Tensor(
        "int32", (sample_per_part, max_roi_num, pooled_height_align16 * 16),
        scope=tik.scope_cbuf, name="hstart_one_cbuf")
    # shape of hweight_one_l_cbuf is (4,300,16)
    hweight_one_l_ubuf = tik_instance.Tensor(
        "float16", (max_roi_num, sample_per_part_align16 * 16),
        scope=tik.scope_ubuf, name="hweight_one_l_ubuf")
    # shape of hweight_one_r_cbuf is (4,300,16)
    hweight_one_r_ubuf = tik_instance.Tensor(
        "float16", (max_roi_num, sample_per_part_align16 * 16),
        scope=tik.scope_ubuf, name="hweight_one_r_ubuf")
    # shape of data_out_cbuf is (1, 1, 32, 32, 16)
    data_out_cbuf = tik_instance.Tensor(
        "float16", (1, class_align, input_shape[2], input_shape[3], 16),
        scope=tik.scope_cbuf, name="data_out_cbuf")
    r_ub = tik_instance.Tensor("float16", (16,), scope=tik.scope_ubuf,
                               name="r_ub")
    r_ub_int32 = tik_instance.Tensor("int32", (16,), scope=tik.scope_ubuf,
                                     name="r_ub_int32")
    with tik_instance.new_stmt_scope():
        rois_ub = tik_instance.Tensor(
            "float16", (roi_align16 * 16, 8), name="rois_ub",
            scope=tik.scope_ubuf)  # 4.75k
        tik_instance.data_move(rois_ub, rois_gm[index, 0, 0], 0, 1,
                               roi_align2, 0, 0)
        tik_instance.data_move(r_ub, rois_gm[index, 0, 0], 0, 1, 1, 0, 0)
        tik_instance.vconv(8, 'floor', r_ub_int32, r_ub, 1, 1, 1, 1, 1)
        rois_num.set_as(r_ub_int32[5])
        coordinate_cal(tik_instance, roi_shape, pooled_height, pooled_width,
                       sample_per_part, heat_map_a, rois_ub, rois_num, 
                       wstart_one_cbuf, wweight_one_l_ubuf,
                       wweight_one_r_ubuf, hstart_one_cbuf,
                       hweight_one_l_ubuf, hweight_one_r_ubuf)
    psroialign_four2five(tik_instance, input_shape, index, pooled_width,
                         pooled_height, data_in, data_out_cbuf, group_height,
                         group_width, class_align, sample_per_part,
                         pooled_width_align16, rois_num, wstart_one_cbuf,
                         hstart_one_cbuf, hweight_one_r_ubuf,
                         hweight_one_l_ubuf, data_5_4_cbuf, class_num,
                         wweight_one_r_ubuf, wweight_one_l_ubuf)


def check_param(input_shape, class_num, group_height, group_width,
                pooled_height,
                pooled_width, roi_shape, sample_per_part, heat_map_a):
    oplog.check_eq(class_num, 10,
                   "class_num(%d) is not equal (%d)" % (class_num, 10))
    oplog.check_eq(sample_per_part, 4,
                   "sample_per_part(%d) is not equal (%d)" % (
                   sample_per_part, 4))
    oplog.check_eq(group_width, 7,
                   "group_width(%d) is not equal (%d)" % (group_width, 7))
    oplog.check_eq(group_height, 7,
                   "group_height(%d) is not equal (%d)" % (group_height, 7))
    oplog.check_eq(pooled_height, 7,
                   "pooled_height(%d) is not equal (%d)" % (pooled_height, 7))
    oplog.check_eq(pooled_width, 7,
                   "pooled_width(%d) is not equal (%d)" % (pooled_width, 7))
    oplog.check_gt(heat_map_a, 0,
                   "heat_map_a(%d) is out of range" % (heat_map_a))
    oplog.check_le(roi_shape[0], 4,
                   "batch(%d) is out of range" % (roi_shape[0]))
    oplog.check_gt(roi_shape[0], 0,
                   "batch(%d) is out of range" % (roi_shape[0]))
    oplog.check_le(input_shape[0], 4,
                   "batch(%d) is out of range" % (input_shape[0]))
    oplog.check_gt(input_shape[0], 0,
                   "batch(%d) is out of range" % (input_shape[0]))
    oplog.check_eq(roi_shape[1], 300,
                   "max_roi_num(%d) is not equal (%d)" % (roi_shape[1], 300))
    oplog.check_eq(roi_shape[2], 8, "per roi must be 8")
    oplog.check_eq(roi_shape[3], 1, "the last dim of roi must be 1")
    oplog.check_eq(len(roi_shape), 4, "the shape of roi must be 4-D")
    oplog.check_eq(input_shape[2], 32,
                   "H(%d) is not equal (%d)" % (input_shape[2], 32))
    oplog.check_eq(input_shape[3], 32,
                   "W(%d) is not equal (%d)" % (input_shape[3], 32))
    oplog.check_eq(len(input_shape), 4, "the shape of roi must be 4-D")


def psroialign_process(tik_instance, class_num, group_height, group_width,
                       pooled_height, pooled_width, sample_per_part,
                       heat_map_a, input_shape, roi_shape, max_roi_num,
                       class_align, pooled_height_align16, 
                       pooled_width_align16, sample_per_part_align16, 
                       roi_align2, roi_align16, rois_gm, data_in, data_out,
                       data_5_4_cbuf, aicore_use, rois_num):
    with tik_instance.for_range(0, aicore_use, block_num=aicore_use) as index:
        psroialign_all(tik_instance, sample_per_part, max_roi_num,
                       pooled_width_align16, sample_per_part_align16,
                       pooled_height_align16, input_shape, class_align,
                       pooled_height, pooled_width, index, roi_align16,
                       data_in, group_height, group_width, roi_shape,
                       heat_map_a, class_num, rois_gm, roi_align2, rois_num,
                       data_5_4_cbuf)
        tmp_5_4_ub = tik_instance.Tensor("float16", (1, 64, 16),
                                         scope=tik.scope_ubuf,
                                         name="tmp_5_4_ub")
        trans_tmp_5_4_ub = tik_instance.Tensor("float16", (4, 16, 16),
                                               scope=tik.scope_ubuf,
                                               name="trans_tmp_5_4_ub")
        zeros = tik_instance.Tensor("float16", (16,), scope=tik.scope_ubuf,
                                    name="zeros")
        tik_instance.vector_dup(16, zeros, 0, 1, 1, 1)
        with tik_instance.for_range(0, rois_num) as r_idx:
            tik_instance.data_move(tmp_5_4_ub[0, 0, 0],
                                   data_5_4_cbuf[r_idx, 0, 0, 0, 0], 
                                   0, 1, 49, 0, 0)
            with tik_instance.for_range(0, 15) as hw_idx:
                tik_instance.data_move(tmp_5_4_ub[0, 49 + hw_idx, 0], 
                                       zeros, 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, 4) as hw_idx:
                tik_instance.vtranspose(trans_tmp_5_4_ub[hw_idx, 0, 0],
                                        tmp_5_4_ub[0, 16 * hw_idx, 0])
            with tik_instance.for_range(0, class_num) as m_idx:
                with tik_instance.for_range(0, 4) as tt_idx:
                    tik_instance.data_move(
                        data_out[index * max_roi_num + r_idx, 
                            m_idx, tt_idx * 16],
                        trans_tmp_5_4_ub[tt_idx, m_idx, 0], 0, 1, 1, 0, 0)
    

def psroialign(datain_dic, rois_dict, output_dic, class_num, group_height,
               group_width, pooled_height,
               pooled_width, sample_per_part, heat_map_a,
               kernel_name="psroialign"):
    input_shape = datain_dic.get("shape")
    roi_shape = rois_dict.get("shape")

    check_param(input_shape, class_num, group_height, group_width,
                pooled_height, pooled_width, roi_shape, sample_per_part,
                heat_map_a)

    tik_instance = tik.Tik(tik.Dprofile())
    roi_shape = [1, roi_shape[0], roi_shape[1], roi_shape[2]]
    max_roi_num = roi_shape[2]
    class_align = (class_num + 15) // 16
    pooled_height_align16 = (pooled_height + 15) // 16
    pooled_width_align16 = (pooled_width + 15) // 16
    sample_per_part_align16 = (sample_per_part + 15) // 16
    roi_align2 = (roi_shape[2] + 1) // 2
    roi_align16 = (roi_shape[2] + 15) // 16
    data_num = (input_shape[0] * input_shape[1] * input_shape[2] *
                input_shape[3] + 15) // 16
    batch = input_shape[0]
    rois_gm = tik_instance.Tensor("float16", (
                                  roi_shape[1] + 1, roi_shape[2],
                                  roi_shape[3]),
                                  name="rois_gm",
                                  scope=tik.scope_gm)
    data_in = tik_instance.Tensor("float16", (data_num * 16,),
                                  scope=tik.scope_gm, name="data_in")
    data_out = tik_instance.Tensor("float16", (
                                   (max_roi_num + 1) * batch,
                                   class_num, pooled_height * pooled_width),
                                   scope=tik.scope_gm, name="data_out")
    data_5_4_cbuf = tik_instance.Tensor("float16", (
                                        max_roi_num, class_align,
                                        pooled_height, pooled_width, 16),
                                        scope=tik.scope_cbuf,
                                        name="data_5_4_cbuf")
    aicore_use = batch
    rois_num = tik_instance.Scalar("int32", name="rois_num")

    psroialign_process(tik_instance, class_num, group_height, group_width,
                       pooled_height, pooled_width, sample_per_part,
                       heat_map_a, input_shape, roi_shape, max_roi_num,
                       class_align, pooled_height_align16, 
                       pooled_width_align16, sample_per_part_align16, 
                       roi_align2, roi_align16, rois_gm, data_in, data_out,
                       data_5_4_cbuf, aicore_use, rois_num)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[data_in, rois_gm],
                          outputs=[data_out], enable_l2=False)
    return tik_instance

