# -*- coding: UTF-8 -*-

from te import tik


def compare_with_scalar_vecdup(tik_instance, rois_num, scalar_tensor,
                               scalar_value, align_64, res):
    with tik_instance.if_scope(rois_num % 128 == 0):
        tik_instance.vector_dup(128, scalar_tensor, scalar_value, align_64, 1,
                                8)
    with tik_instance.else_scope():
        with tik_instance.if_scope(rois_num > 128):
            tik_instance.vector_dup(128, scalar_tensor, scalar_value,
                                    align_64, 1, 8)
            tik_instance.vector_dup(res, scalar_tensor[align_64 * 128],
                                    scalar_value, 1, 1, 8)
        with tik_instance.else_scope():
            tik_instance.vector_dup(rois_num, scalar_tensor, scalar_value, 1,
                                    1, 8)


def compare_with_scalar_vmax(tik_instance, rois_num, source_ub, scalar_tensor,
                             align_64, res):
    with tik_instance.if_scope(rois_num % 128 == 0):
        tik_instance.vmax(128, source_ub, source_ub, scalar_tensor, align_64,
                          1, 1, 1, 8, 8, 8)
    with tik_instance.else_scope():
        with tik_instance.if_scope(rois_num > 128):
            tik_instance.vmax(128, source_ub, source_ub, scalar_tensor,
                              align_64, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(res, source_ub[align_64 * 128],
                              source_ub[align_64 * 128],
                              scalar_tensor[align_64 * 128],
                              1, 1, 1, 1, 8, 8, 8)
        with tik_instance.else_scope():
            tik_instance.vmax(rois_num, source_ub, source_ub, scalar_tensor,
                              1, 1, 1, 1, 8, 8, 8)


def compare_with_scalar(tik_instance, source_ub, scalar, rois_num):  # 304
    scalar_value = tik_instance.Scalar("float16", "scalar_value")
    scalar_value.set_as(scalar)
    scalar_tensor = tik_instance.Tensor("float16", (source_ub.shape[0],),
                                        name="scalar_tensor",
                                        scope=tik.scope_ubuf)
    align_64 = rois_num // 128
    res = rois_num - align_64 * 128
    compare_with_scalar_vecdup(tik_instance, rois_num, scalar_tensor,
                               scalar_value, align_64, res)
    compare_with_scalar_vmax(tik_instance, rois_num, source_ub, scalar_tensor,
                             align_64, res)


def coordinate_cal_origin_roi_extract(tik_instance, rois_num, coor_x1,
                                      coor_y1, coor_x2, coor_y2, heat_map_a,
                                      roi_num_align_128, res1):
    with tik_instance.if_scope(rois_num % 128 == 0):
        tik_instance.vmuls(128, coor_x1, coor_x1, 1.0 / heat_map_a,
                           rois_num // 128, 1, 1, 8, 8)
        tik_instance.vmuls(128, coor_y1, coor_y1, 1.0 / heat_map_a,
                           rois_num // 128, 1, 1, 8, 8)
        tik_instance.vmuls(128, coor_x2, coor_x2, 1.0 / heat_map_a,
                           rois_num // 128, 1, 1, 8, 8)
        tik_instance.vmuls(128, coor_y2, coor_y2, 1.0 / heat_map_a,
                           rois_num // 128, 1, 1, 8, 8)
    with tik_instance.else_scope():
        with tik_instance.if_scope(roi_num_align_128 >= 1):
            tik_instance.vmuls(128, coor_x1, coor_x1, 1.0 / heat_map_a,
                               roi_num_align_128, 1, 1, 8, 8)
            tik_instance.vmuls(res1, coor_x1[roi_num_align_128 * 128],
                               coor_x1[roi_num_align_128 * 128],
                               1.0 / heat_map_a, 1, 1, 1, 8, 8)
            tik_instance.vmuls(128, coor_y1, coor_y1, 1.0 / heat_map_a,
                               roi_num_align_128, 1, 1, 8, 8)
            tik_instance.vmuls(res1, coor_y1[roi_num_align_128 * 128],
                               coor_y1[roi_num_align_128 * 128],
                               1.0 / heat_map_a, 1, 1, 1, 8, 8)
            tik_instance.vmuls(128, coor_x2, coor_x2, 1.0 / heat_map_a,
                               roi_num_align_128, 1, 1, 8, 8)
            tik_instance.vmuls(res1, coor_x2[roi_num_align_128 * 128],
                               coor_x2[roi_num_align_128 * 128],
                               1.0 / heat_map_a, 1, 1, 1, 8, 8)
            tik_instance.vmuls(128, coor_y2, coor_y2, 1.0 / heat_map_a,
                               roi_num_align_128, 1, 1, 8, 8)
            tik_instance.vmuls(res1, coor_y2[roi_num_align_128 * 128],
                               coor_y2[roi_num_align_128 * 128],
                               1.0 / heat_map_a, 1, 1, 1, 8, 8)
        with tik_instance.else_scope():
            tik_instance.vmuls(rois_num, coor_x1, coor_x1, 1.0 / heat_map_a,
                               1, 1, 1, 8, 8)
            tik_instance.vmuls(rois_num, coor_y1, coor_y1, 1.0 / heat_map_a,
                               1, 1, 1, 8, 8)
            tik_instance.vmuls(rois_num, coor_x2, coor_x2, 1.0 / heat_map_a,
                               1, 1, 1, 8, 8)
            tik_instance.vmuls(rois_num, coor_y2, coor_y2, 1.0 / heat_map_a,
                               1, 1, 1, 8, 8)


def coordinate_cal_roi_wh_cal(tik_instance, rois_num, rois_width, coor_x2,
                              coor_x1, rois_height, coor_y2, coor_y1,
                              roi_num_align_128, res1):
    with tik_instance.if_scope(rois_num % 128 == 0):
        tik_instance.vsub(128, rois_width[0], coor_x2, coor_x1,
                          rois_num // 128, 1, 1, 1, 8, 8, 8)
        tik_instance.vsub(128, rois_height[0], coor_y2, coor_y1,
                          rois_num // 128, 1, 1, 1, 8, 8, 8)
    with tik_instance.else_scope():
        with tik_instance.if_scope(rois_num > 128):
            tik_instance.vsub(128, rois_height[0], coor_y2, coor_y1,
                              roi_num_align_128, 1, 1, 1, 8,
                              8, 8)
            tik_instance.vsub(res1, rois_height[roi_num_align_128 * 128],
                              coor_y2[roi_num_align_128 * 128],
                              coor_y1[roi_num_align_128 * 128],
                              1, 1, 1, 1, 8, 8, 8)
            tik_instance.vsub(128, rois_width[0], coor_x2[0], coor_x1[0],
                              roi_num_align_128, 1, 1, 1, 8, 8,
                              8)
            tik_instance.vsub(res1, rois_width[roi_num_align_128 * 128],
                              coor_x2[roi_num_align_128 * 128],
                              coor_x1[roi_num_align_128 * 128],
                              1, 1, 1, 1, 8, 8, 8)
        with tik_instance.else_scope():
            tik_instance.vsub(rois_num, rois_height[0], coor_y2[0],
                              coor_y1[0], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vsub(rois_num, rois_width[0], coor_x2[0], coor_x1[0],
                              1, 1, 1, 1, 8, 8, 8)


def coordinate_cal_bin(tik_instance, rois_num, bin_height, rois_height,
                       bin_width, rois_width, persample_bin_height,
                       persample_bin_width, res1, roi_num_align_128,
                       sample_per_part, pooled_height, pooled_width,
                       sample_part_ub, temp_scalar,
                       one_persample_bin_height, one_persample_bin_width):
    with tik_instance.if_scope(rois_num % 128 == 0):
        tik_instance.vmuls(128, bin_height[0], rois_height[0],
                           1.0 / (pooled_height), roi_num_align_128, 1, 1, 8,
                           8)
        tik_instance.vmuls(128, bin_width[0], rois_width[0],
                           1.0 / (pooled_width), roi_num_align_128, 1, 1, 8,
                           8)
        tik_instance.vmuls(128, persample_bin_height[0], rois_height[0],
                           1.0 / (sample_per_part * pooled_height),
                           roi_num_align_128, 1, 1, 8, 8)
        tik_instance.vmuls(128, persample_bin_width[0], rois_width[0],
                           1.0 / (sample_per_part * pooled_width),
                           roi_num_align_128, 1, 1, 8, 8)
        with tik_instance.for_range(0, sample_per_part) as smp:
            temp_scalar.set_as(sample_part_ub[smp])
            tik_instance.vmuls(128, one_persample_bin_width[smp, 0],
                               persample_bin_width[0], temp_scalar,
                               roi_num_align_128, 1, 1, 8, 8)
            tik_instance.vmuls(128, one_persample_bin_height[smp, 0],
                               persample_bin_height[0], temp_scalar,
                               roi_num_align_128, 1, 1, 8, 8)
    with tik_instance.else_scope():
        coordinate_cal_bin_rois_num_process(tik_instance, 
                       rois_num, bin_height, rois_height,
                       bin_width, rois_width, persample_bin_height,
                       persample_bin_width, res1, roi_num_align_128,
                       sample_per_part, pooled_height, pooled_width,
                       sample_part_ub, temp_scalar,
                       one_persample_bin_height, one_persample_bin_width)


def coordinate_cal_bin_rois_num_process(tik_instance, 
                       rois_num, bin_height, rois_height,
                       bin_width, rois_width, persample_bin_height,
                       persample_bin_width, res1, roi_num_align_128,
                       sample_per_part, pooled_height, pooled_width,
                       sample_part_ub, temp_scalar,
                       one_persample_bin_height, one_persample_bin_width):
    with tik_instance.if_scope(rois_num > 128):
        tik_instance.vmuls(
            128, bin_height[0], rois_height[0], 1.0 / (pooled_height), 
            roi_num_align_128, 1, 1, 8, 8)
        tik_instance.vmuls(res1, bin_height[roi_num_align_128 * 128],
                            rois_height[roi_num_align_128 * 128],
                            1.0 / (pooled_height), 1, 1, 1, 8, 8)
        tik_instance.vmuls(128, bin_width[0], rois_width[0],
                            1.0 / (pooled_width), roi_num_align_128, 1, 1,
                            8, 8)
        tik_instance.vmuls(res1, bin_width[roi_num_align_128 * 128],
                            rois_width[roi_num_align_128 * 128],
                            1.0 / (pooled_width), 1, 1, 1, 8, 8)
        tik_instance.vmuls(128, persample_bin_height[0], rois_height[0],
                            1.0 / (sample_per_part * pooled_height),
                            roi_num_align_128, 1, 1, 8, 8)
        tik_instance.vmuls(
            res1, persample_bin_height[roi_num_align_128 * 128],
            rois_height[roi_num_align_128 * 128], 
            1.0 / (sample_per_part * pooled_height), 1, 1, 1, 8, 8)
        tik_instance.vmuls(128, persample_bin_width[0], rois_width[0],
                            1.0 / (sample_per_part * pooled_width),
                            roi_num_align_128, 1, 1, 8, 8)
        tik_instance.vmuls(
            res1, persample_bin_width[roi_num_align_128 * 128],
            rois_width[roi_num_align_128 * 128],
            1.0 / (sample_per_part * pooled_width), 1, 1, 1, 8, 8)
        with tik_instance.for_range(0, sample_per_part) as smp:
            temp_scalar.set_as(sample_part_ub[smp])
            tik_instance.vmuls(128, one_persample_bin_height[smp, 0],
                                persample_bin_height[0], temp_scalar,
                                roi_num_align_128, 1, 1, 8, 8)
            tik_instance.vmuls(res1, one_persample_bin_height[
                smp, roi_num_align_128 * 128], persample_bin_height[
                    roi_num_align_128 * 128], temp_scalar, 1, 1, 1, 8, 8)
            tik_instance.vmuls(128, one_persample_bin_width[smp, 0],
                                persample_bin_width[0], temp_scalar,
                                roi_num_align_128, 1, 1, 8, 8)
            tik_instance.vmuls(res1, one_persample_bin_width[
                smp, roi_num_align_128 * 128], persample_bin_width[
                    roi_num_align_128 * 128], temp_scalar, 1, 1, 1, 8, 8)
    with tik_instance.else_scope():
        coordinate_cal_bin_rois_num_less_than_128(tik_instance, 
                       rois_num, bin_height, rois_height,
                       bin_width, rois_width, persample_bin_height,
                       persample_bin_width,
                       sample_per_part, pooled_height, pooled_width,
                       sample_part_ub, temp_scalar,
                       one_persample_bin_height, one_persample_bin_width)


def coordinate_cal_bin_rois_num_less_than_128(tik_instance, 
                       rois_num, bin_height, rois_height,
                       bin_width, rois_width, persample_bin_height,
                       persample_bin_width,
                       sample_per_part, pooled_height, pooled_width,
                       sample_part_ub, temp_scalar,
                       one_persample_bin_height, one_persample_bin_width):
    tik_instance.vmuls(rois_num, bin_height[0], rois_height[0],
                        1.0 / (pooled_height), 1, 1, 1, 8, 8)
    tik_instance.vmuls(rois_num, bin_width[0], rois_width[0],
                        1.0 / (pooled_width), 1, 1, 1, 8, 8)
    tik_instance.vmuls(rois_num, persample_bin_height[0],
                        rois_height[0],
                        1.0 / (sample_per_part * pooled_height), 1, 1,
                        1, 8, 8)
    tik_instance.vmuls(rois_num, persample_bin_width[0],
                        rois_width[0],
                        1.0 / (sample_per_part * pooled_width), 1, 1,
                        1, 8, 8)
    with tik_instance.for_range(0, sample_per_part) as smp:
        temp_scalar.set_as(sample_part_ub[smp])
        tik_instance.vmuls(rois_num, one_persample_bin_width[smp, 0],
                            persample_bin_width[0], temp_scalar, 1, 1,
                            1, 8, 8)
        tik_instance.vmuls(rois_num, one_persample_bin_height[smp, 0],
                            persample_bin_height[0], temp_scalar, 1, 1,
                            1, 8, 8)


def coordinate_cal_worh(tik_instance, temp_scalar, bin_height, idx, hstart,
                        y_tensor, coor_y1, pooled_height_align16,
                        hstart_int, persample_bin_height, sample_per_part,
                        tmp_coor_ub,
                        sample_part_ub, tmp_coor_ub_int32, tmp_coor_ub_fp16,
                        hweight_one_l_ubuf, hweight_one_r_ubuf,
                        one_persample_bin_height, hstart_one_cbuf):
    temp_scalar = tik_instance.Scalar("float16", "temp_scalar")
    temp_scalar.set_as(bin_height[idx])
    tik_instance.vmuls(16, hstart[0], y_tensor, temp_scalar,
                       pooled_height_align16, 1, 1, 1, 1)
    temp_scalar.set_as(coor_y1[idx])
    tik_instance.vadds(16, hstart[0], hstart[0], temp_scalar,
                       pooled_height_align16, 1, 1, 1, 1)
    tik_instance.vconv(16, 'floor', hstart_int[0], hstart[0],
                       pooled_height_align16, 1, 1, 2, 1)
    tik_instance.vconv(16, '', hstart[0], hstart_int[0],
                       pooled_height_align16, 1, 1, 1, 2, 1.0)
    temp_scalar.set_as(persample_bin_height[idx])
    tik_instance.vmuls(sample_per_part, tmp_coor_ub, sample_part_ub,
                       temp_scalar, 1, 1, 1, 1, 1)
    temp_scalar.set_as(hstart[0])
    tik_instance.vadds(sample_per_part, tmp_coor_ub, tmp_coor_ub, temp_scalar,
                       1, 1, 1, 1, 1)
    tik_instance.vconv(sample_per_part, 'floor', tmp_coor_ub_int32,
                       tmp_coor_ub, 1, 1, 1, 2, 1)
    tik_instance.vconv(sample_per_part, '', tmp_coor_ub_fp16,
                       tmp_coor_ub_int32, 1, 1, 1, 1, 2, 1.0)
    tik_instance.vsub(sample_per_part, hweight_one_l_ubuf[idx, 0],
                      tmp_coor_ub, tmp_coor_ub_fp16, 1, 1, 1, 1, 1, 1, 1)
    tik_instance.vmuls(sample_per_part, hweight_one_r_ubuf[idx, 0],
                       hweight_one_l_ubuf[idx, 0], -1, 1, 1, 1, 1, 1)
    tik_instance.vadds(sample_per_part, hweight_one_r_ubuf[idx, 0],
                       hweight_one_r_ubuf[idx, 0], 1, 1, 1, 1, 1, 1)
    with tik_instance.for_range(0, sample_per_part) as cc_idx:
        temp_scalar.set_as(one_persample_bin_height[cc_idx, idx])
        tik_instance.vadds(16, tmp_coor_ub, hstart[0], temp_scalar,
                           pooled_height_align16, 1, 1, 1, 1)
        tik_instance.vconv(16, 'floor', tmp_coor_ub_int32, tmp_coor_ub,
                           pooled_height_align16, 1, 1, 2, 1)
        tik_instance.data_move(hstart_one_cbuf[cc_idx, idx, 0], 
                               tmp_coor_ub_int32,
                               0, 1, pooled_height_align16, 1, 1)


def coordinate_wgtandstart(tik_instance, bin_height, bin_width, coor_y1,
                           coor_x1, persample_bin_height,
                           persample_bin_width, sample_per_part,
                           sample_part_ub,
                           hweight_one_l_ubuf, wweight_one_l_ubuf,
                           hweight_one_r_ubuf, wweight_one_r_ubuf,
                           one_persample_bin_height,
                           one_persample_bin_width, hstart_one_cbuf,
                           wstart_one_cbuf, pooled_height, pooled_width,
                           rois_num):
    pooled_height_align16 = (pooled_height + 15) // 16
    pooled_width_align16 = (pooled_width + 15) // 16
    y_tensor = tik_instance.Tensor("float16", (pooled_height_align16 * 16,),
                                   scope=tik.scope_ubuf, name="y_tensor")
    x_tensor = tik_instance.Tensor("float16", (pooled_width_align16 * 16,),
                                   scope=tik.scope_ubuf, name="x_tensor")
    temp_scalar = tik_instance.Scalar("float16", "temp_scalar")
    for idx in range(0, pooled_height):
        temp_scalar.set_as(idx)
        y_tensor[idx] = temp_scalar
    for idx in range(0, pooled_width):
        temp_scalar.set_as(idx)
        x_tensor[idx] = temp_scalar
    hstart_int = tik_instance.Tensor("int32", (pooled_height_align16 * 16,),
                                     scope=tik.scope_ubuf, name="hstart_int")
    wstart_int = tik_instance.Tensor("int32", (pooled_width_align16 * 16,),
                                     scope=tik.scope_ubuf, name="wstart_int")
    hstart = tik_instance.Tensor("float16", (pooled_height_align16 * 16,),
                                 scope=tik.scope_ubuf, name="hstart")
    wstart = tik_instance.Tensor("float16", (pooled_width_align16 * 16,),
                                 scope=tik.scope_ubuf, name="wstart")
    tmp_coor_ub = tik_instance.Tensor("float16",
                                      (pooled_height_align16 * 16,),
                                      scope=tik.scope_ubuf,
                                      name="tmp_coor_ub")
    tmp_coor_ub_int32 = tik_instance.Tensor("int32",
                                            (pooled_height_align16 * 16,),
                                            scope=tik.scope_ubuf,
                                            name="tmp_coor_ub_int32")
    tmp_coor_ub_fp16 = tik_instance.Tensor("float16",
                                           (pooled_height_align16 * 16,),
                                           scope=tik.scope_ubuf,
                                           name="tmp_coor_ub_fp16")
    # 计算7*7每个格子向下取整的采样点
    with tik_instance.for_range(0, rois_num) as idx:
        coordinate_cal_worh(tik_instance, temp_scalar, bin_height, idx,
                            hstart, y_tensor, coor_y1,
                            pooled_height_align16, hstart_int,
                            persample_bin_height, sample_per_part,
                            tmp_coor_ub, sample_part_ub, tmp_coor_ub_int32,
                            tmp_coor_ub_fp16, hweight_one_l_ubuf,
                            hweight_one_r_ubuf,
                            one_persample_bin_height, hstart_one_cbuf)
        coordinate_cal_worh(tik_instance, temp_scalar, bin_width, idx, wstart,
                            x_tensor, coor_x1, pooled_width_align16,
                            wstart_int, persample_bin_width, sample_per_part,
                            tmp_coor_ub, sample_part_ub, tmp_coor_ub_int32,
                            tmp_coor_ub_fp16, wweight_one_l_ubuf,
                            wweight_one_r_ubuf,
                            one_persample_bin_width, wstart_one_cbuf)


def coordinate_cal(tik_instance, roi_shape, pooled_height, pooled_width,
                   sample_per_part, heat_map_a, rois_ub, rois_num,
                   wstart_one_cbuf, wweight_one_l_ubuf, wweight_one_r_ubuf,
                   hstart_one_cbuf, hweight_one_l_ubuf,
                   hweight_one_r_ubuf):
    roi_align16 = (roi_shape[2] + 15) // 16
    coor_x1 = tik_instance.Tensor("float16", (roi_align16 * 16,),
                                  scope=tik.scope_ubuf, name="coor_x1")
    coor_y1 = tik_instance.Tensor("float16", (roi_align16 * 16,),
                                  scope=tik.scope_ubuf, name="coor_y1")
    coor_x2 = tik_instance.Tensor("float16", (roi_align16 * 16,),
                                  scope=tik.scope_ubuf, name="coor_x2")
    coor_y2 = tik_instance.Tensor("float16", (roi_align16 * 16,),
                                  scope=tik.scope_ubuf, name="coor_y2")
    roi_num_align_128 = rois_num // 128
    res1 = rois_num - roi_num_align_128 * 128
    max_roi_num = roi_shape[2]
    tik_instance.vextract(coor_x1, rois_ub, roi_align16, 0)
    tik_instance.vextract(coor_y1, rois_ub, roi_align16, 1)
    tik_instance.vextract(coor_x2, rois_ub, roi_align16, 2)
    tik_instance.vextract(coor_y2, rois_ub, roi_align16, 3)
    coordinate_cal_origin_roi_extract(tik_instance, rois_num, coor_x1,
                                      coor_y1, coor_x2, coor_y2, heat_map_a,
                                      roi_num_align_128, res1)
    rois_height = tik_instance.Tensor("float16", (max_roi_num,),
                                      name="rois_height",
                                      scope=tik.scope_ubuf)
    rois_width = tik_instance.Tensor("float16", (max_roi_num,),
                                     name="rois_width", scope=tik.scope_ubuf)
    coordinate_cal_roi_wh_cal(tik_instance, rois_num, rois_width, coor_x2,
                              coor_x1, rois_height, coor_y2, coor_y1,
                              roi_num_align_128, res1)
    coordinate_cal_process(tik_instance, pooled_height, pooled_width,
                           sample_per_part, rois_num, wstart_one_cbuf,
                           wweight_one_l_ubuf, wweight_one_r_ubuf,
                           hstart_one_cbuf, hweight_one_l_ubuf,
                           hweight_one_r_ubuf, coor_x1, coor_y1, res1,
                           roi_num_align_128, max_roi_num,
                           rois_height, rois_width)


def coordinate_cal_process(tik_instance, pooled_height, pooled_width,
                           sample_per_part, rois_num, wstart_one_cbuf,
                           wweight_one_l_ubuf, wweight_one_r_ubuf,
                           hstart_one_cbuf, hweight_one_l_ubuf,
                           hweight_one_r_ubuf, coor_x1, coor_y1, res1, 
                           roi_num_align_128, max_roi_num, 
                           rois_height, rois_width):
    compare_with_scalar(tik_instance, rois_height, 0.1, rois_num)
    compare_with_scalar(tik_instance, rois_width, 0.1, rois_num)
    bin_width = tik_instance.Tensor("float16", (max_roi_num,),
                                    name="bin_width", scope=tik.scope_ubuf)
    bin_height = tik_instance.Tensor("float16", (max_roi_num,),
                                     name="bin_height", scope=tik.scope_ubuf)
    persample_bin_width = tik_instance.Tensor("float16", (max_roi_num,),
                                              name="persample_bin_width",
                                              scope=tik.scope_ubuf)
    persample_bin_height = tik_instance.Tensor("float16", (max_roi_num,),
                                               name="persample_bin_height",
                                               scope=tik.scope_ubuf)
    max_roi_num_align16 = (max_roi_num + 15) // 16
    one_persample_bin_width = tik_instance.Tensor("float16", (
        sample_per_part, max_roi_num_align16 * 16),
        name="one_persample_bin_width", scope=tik.scope_ubuf)
    one_persample_bin_height = tik_instance.Tensor("float16", (
        sample_per_part, max_roi_num_align16 * 16),
        name="one_persample_bin_height", scope=tik.scope_ubuf)
    sample_part_ub = tik_instance.Tensor("float16", (16,),
                                         scope=tik.scope_ubuf,
                                         name="sample_part_ub")
    temp_scalar = tik_instance.Scalar("float16", "temp_scalar")
    for i_idx in range(0, sample_per_part):
        k = i_idx + 0.5
        temp_scalar.set_as(k)
        sample_part_ub[i_idx] = temp_scalar
    coordinate_cal_bin(tik_instance, rois_num, bin_height, rois_height,
                       bin_width, rois_width, persample_bin_height,
                       persample_bin_width, res1, roi_num_align_128,
                       sample_per_part, pooled_height, pooled_width,
                       sample_part_ub, temp_scalar,
                       one_persample_bin_height, one_persample_bin_width)
    coordinate_wgtandstart(tik_instance, bin_height, bin_width, coor_y1,
                           coor_x1, persample_bin_height,
                           persample_bin_width, sample_per_part,
                           sample_part_ub,
                           hweight_one_l_ubuf, wweight_one_l_ubuf,
                           hweight_one_r_ubuf, wweight_one_r_ubuf,
                           one_persample_bin_height,
                           one_persample_bin_width, hstart_one_cbuf,
                           wstart_one_cbuf, pooled_height, pooled_width,
                           rois_num)
