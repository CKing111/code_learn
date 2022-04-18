import inspect

from te import tik
from util import OpLog as oplog


def glog_print_value(func):
    '''
        debug wrapper: print func variables
    '''
    def inner(*args, **kwargs):
        func_args = inspect.getcallargs(func, *args, **kwargs)
        print("func: " + func.__name__)
        for key, value in func_args.items():
            print(key + ": ", value)
        res = func(*args, **kwargs)
        return res
    return inner


def data_addr_align32(element_num, element_size):
    '''
        return offline 32 align block
    '''
    size = (element_num * element_size - 1) // 32 + 1
    return size * 32 // element_size


def set_mask(tik_instance, idx, repeat_times, mask_full, mask_tail):
    # get cmpmask
    with tik_instance.if_scope(idx != repeat_times - 1):
        mask = mask_full
    with tik_instance.else_scope():
        mask = mask_tail
    return mask


def vector_dup_align256(tik_instance, ubuf, ubuf_off, element_num,
                        element_bytes, dup_data):
    '''
        support 16bits and 32 bits
        element_num: element number
        element_bytes: 2 bytes for 16bits and 4 bytes for 32 bits
    '''
    oplog.check(element_bytes != 2 and element_bytes != 4,
                "ERROR: vector_dup instruction support 2 or 4 only")
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        with tik_instance.if_scope(i != repeat_times - 1):
            tik_instance.vector_dup(mask_full, ubuf[mask_full * i + ubuf_off],
                                    dup_data, 1, 1, 8)
        with tik_instance.else_scope():
            tik_instance.vector_dup(mask_tail, ubuf[mask_full * i + ubuf_off],
                                    dup_data, 1, 1, 8)


def vector_cmp_assign(tik_instance, cmp_ubuf1, cmp_off1, cmp_ubuf2, cmp_off2,
                      element_num, element_bytes, sel_ubuf1, sel_off1,
                      sel_ubuf2,
                      sel_off2, result_ubuf, result_off):
    """
    result_ubuf equals to sel_ubuf1 when cmp_ubuf1 less than cmp_ubuf2
    """
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)
    with tik_instance.for_range(0, repeat_times) as i:
        # get cmpmask
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        cmpmask = tik_instance.vcmp_lt(mask,
                                       cmp_ubuf1[cmp_off1 + i * mask_full],
                                       cmp_ubuf2[cmp_off2 + i * mask_full],
                                       1, 1)
        # select num
        tik_instance.vsel(mask, 0, result_ubuf[result_off + i * mask_full],
                          cmpmask,
                          sel_ubuf1[sel_off1 + i * mask_full],
                          sel_ubuf2[sel_off2 + i * mask_full], 1, 1, 1, 1, 8,
                          8, 8)


def vector_cmp_assign_gt(tik_instance, cmp_ubuf1, cmp_off1, cmp_ubuf2,
                         cmp_off2, element_num, element_bytes, sel_ubuf1,
                         sel_off1, sel_ubuf2, sel_off2, result_ubuf,
                         result_off):
    '''
    result_ubuf equals to sel_ubuf1 when cmp_ubuf1 less than cmp_ubuf2
    '''
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        # get cmpmask
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        cmpmask = tik_instance.vcmp_gt(mask,
                                       cmp_ubuf1[cmp_off1 + i * mask_full],
                                       cmp_ubuf2[cmp_off2 + i * mask_full],
                                       1, 1)
        # select num  #todo: bugs ?
        tik_instance.vsel(mask, 0, result_ubuf[result_off + i * mask_full],
                          cmpmask,
                          sel_ubuf1[sel_off1 + i * mask_full],
                          sel_ubuf2[sel_off2 + i * mask_full], 1, 1, 1, 1, 8,
                          8, 8)


def vmuls_vadds_align256(tik_instance, ubuf, off, mul_scalar, add_scalar,
                         element_num, element_bytes):
    '''
        vector vmul scalar, vadd scalar
    '''
    repeat_times, mask_full, mask_tail = get_full_and_tail(
                                                        tik_instance,
                                                        element_num,
                                                        element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vmuls(mask, ubuf[off + i * mask_full],
                           ubuf[off + i * mask_full], mul_scalar, 1, 1, 1, 8,
                           8)
        tik_instance.vadds(mask, ubuf[off + i * mask_full],
                           ubuf[off + i * mask_full], add_scalar, 1, 1, 1, 8,
                           8)


def vmuls_vadd_align256(tik_instance, ubuf, off, mul_scalar, adder_ub,
                        element_num, element_bytes):
    '''
        vector mul scalar, add vector
    '''
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vmuls(mask, ubuf[off + i * mask_full],
                           ubuf[off + i * mask_full], mul_scalar, 1, 1, 1, 8,
                           8)
        tik_instance.vadd(mask, ubuf[off + i * mask_full],
                          ubuf[off + i * mask_full],
                          adder_ub[i * mask_full], 1, 1, 1, 1, 8, 8, 8)


def vexp_mul_scalar(tik_instance, ubuf, off, mul_scalar, element_num,
                    element_bytes):
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vexp(mask, ubuf[off + i * mask_full],
                          ubuf[off + i * mask_full], 1, 1, 1, 8, 8)
        tik_instance.vmuls(mask, ubuf[off + i * mask_full],
                           ubuf[off + i * mask_full], mul_scalar, 1, 1, 1, 8,
                           8)


def vdup_width_plane(tik_instance, temp_ub, height, width, element_bytes):
    '''
        output structure:
        ## 0 1 2 3 4 5 6...31
        ## 0 1 2 3 4 5 6...31
    '''
    oplog.check_ne(tik_instance, None, "tik_instance is None")

    with tik_instance.new_stmt_scope():
        line_tmp_ub = tik_instance.Tensor("float16", (width,),
                                          name='line_tmp_ub',
                                          scope=tik.scope_ubuf)
        for w_idx in range(width):
            bais_scalar = tik_instance.Scalar(dtype="float16",
                                              name="bais_scalar")
            bais_scalar.set_as(w_idx)
            line_tmp_ub[w_idx].set_as(bais_scalar)
        with tik_instance.for_range(0, height) as h_idx:
            mask = width  # 32 element
            tik_instance.data_move(temp_ub[width * h_idx], line_tmp_ub, 0, 1,
                                   mask * element_bytes // 32, 0, 0)


def vdup_height_plane(tik_instance, temp_ub,
                      height, width, element_bytes):
    '''
        output structure:
            ## 0 0 0 0 0
            ## 1 1 1 1 1
    '''
    oplog.check_ne(tik_instance, None, "tik_instance is None")

    for h_idx in range(height):
        height_scalar = tik_instance.Scalar(dtype="float16",
                                            name="height_scalar")
        height_scalar.set_as(h_idx)
        vector_dup_align256(tik_instance, temp_ub, (width * h_idx), width, 2,
                            height_scalar)


def vadds_align256(tik_instance,
                   ubuf,
                   off,
                   add_scalar,
                   element_num,
                   element_bytes):
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vadds(mask, ubuf[off + i * mask_full],
                           ubuf[off + i * mask_full], add_scalar, 1, 1, 1, 8,
                           8)


def get_full_and_tail(tik_instance, element_num, element_bytes):
    oplog.check_ne(tik_instance, None, "tik_instance is None")

    repeat_times = (element_num * element_bytes - 1) // 256 + 1
    mask_full = 256 // element_bytes
    mask_tail = element_num - mask_full * (repeat_times - 1)

    return repeat_times, mask_full, mask_tail


def vmuls_align256(tik_instance, ubuf, off, mul_scalar,
                   element_num, element_bytes):
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vmuls(mask, ubuf[off + i * mask_full],
                           ubuf[off + i * mask_full], mul_scalar, 1, 1, 1, 8,
                           8)


def vsubs_align256(tik_instance, dst_ub, dst_off, src0_ub, src0_off, src1_ub,
                   src1_off, element_num, element_bytes):
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vadds(mask, dst_ub[dst_off + i * mask_full],
                           src0_ub[src0_off + i * mask_full],
                           src1_ub[src1_off + i * mask_full],
                           1, 1, 1, 1, 8, 8)


def vadd_align256(tik_instance, dst_ub, dst_off, src0_ub, src0_off, src1_ub,
                  src1_off, element_num, element_bytes):
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vadd(mask, dst_ub[dst_off + i * mask_full],
                          src0_ub[src0_off + i * mask_full],
                          src1_ub[src1_off + i * mask_full],
                          1, 1, 1, 1, 8, 8, 8)


def vrelu_align256(tik_instance, src_ub, src_off, dst_ub, dst_off,
                   element_num,
                   element_bytes):
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vrelu(mask, dst_ub[dst_off + i * mask_full],
                           src_ub[src_off + i * mask_full],
                           1, 1, 1, 8, 8)


def data_move_align256(tik_instance, src, src_off, dst, dst_off, element_num,
                       element_bytes):
    oplog.check_ne(tik_instance, None, "tik_instance is None")
    burst = element_num * element_bytes // 32
    oplog.check_lt(burst, 65535, "data_move more than 65535")

    tik_instance.data_move(dst[dst_off], src[src_off], 0, 1, burst, 0, 0)


def vsub_align256(tik_instance, dst_ub, dst_off, src0_ub, src0_off, src1_ub,
                  src1_off, element_num, element_bytes):
    repeat_times, mask_full, mask_tail = get_full_and_tail(tik_instance,
                                                           element_num,
                                                           element_bytes)

    with tik_instance.for_range(0, repeat_times) as i:
        mask = set_mask(tik_instance, i, repeat_times, mask_full, mask_tail)
        tik_instance.vsub(mask, dst_ub[dst_off + i * mask_full],
                          src0_ub[src0_off + i * mask_full],
                          src1_ub[src1_off + i * mask_full],
                          1, 1, 1, 1, 8, 8, 8)


def cmp_gt_float(tik_instance,
                 float1,
                 float2,
                 intype):
    '''
        float1 and float2 should be scalar or tensorp[x]
        return > 0 if float1 > float2 else return < 0
    '''
    element_num = 32 if intype == "float16" else 16
    element_bytes = 2 if intype == "float16" else 4
    stride = 8 if intype == "float16" else 4

    cmp_tmp_ub = tik_instance.Tensor(intype, (element_num,),
                                     name="cmp_tmp_ub",
                                     scope=tik.scope_ubuf)
    vector_dup_align256(tik_instance, cmp_tmp_ub, 0, element_num,
                        element_bytes, 1.0)
    vadds_align256(tik_instance, cmp_tmp_ub, 0, 1, element_num // 2 + 1,
                   element_bytes)

    cmp_tmp_ub[0].set_as(float1)
    cmp_tmp_ub[element_num // 2].set_as(float2)
    vsub_align256(tik_instance, cmp_tmp_ub, 0, cmp_tmp_ub, 0, cmp_tmp_ub,
                  element_num // 2, element_num // 2,
                  element_bytes)
    tik_instance.vrec(element_num // 2, cmp_tmp_ub, cmp_tmp_ub, 1, 1, 1,
                      stride, stride)

    one_ub = tik_instance.Tensor("int32", (16,), name="one_ub",
                                 scope=tik.scope_ubuf)
    vector_dup_align256(tik_instance, one_ub, 0, element_num,
                        element_bytes, 0)

    if intype == "float32":
        zero_ub = tik_instance.Tensor("float16", (16,), name="zero_ub",
                                      scope=tik.scope_ubuf)
        vector_dup_align256(tik_instance, zero_ub, 0, element_num,
                            element_bytes, 0)
        tik_instance.vconv(1, "", zero_ub, cmp_tmp_ub, 1, 1, 1, stride, 8)
        tik_instance.vconv(1, "floor", one_ub, zero_ub, 1, 1, 1, 4, 8)
    else:
        tik_instance.vconv(1, "floor", one_ub, cmp_tmp_ub, 1, 1, 1, 4, 8)

    dst_scalar = tik_instance.Scalar("int32", name="dst_scalar")
    dst_scalar.set_as(one_ub[0])
    return dst_scalar


def vconv_fp16_scalar_to_int32(tik_instance, input_fp16_scalar, int32_scalar):
    with tik_instance.new_stmt_scope():
        int32_ub = tik_instance.Tensor("int32", (8,), name="int32_ub",
                                       scope=tik.scope_ubuf)
        float16_ub = tik_instance.Tensor("float16", (16,), name="float16_ub",
                                         scope=tik.scope_ubuf)
        float16_ub[0].set_as(input_fp16_scalar)

        tik_instance.vconv(1, "floor", int32_ub, float16_ub, 1, 1, 1, 8, 4)
        int32_scalar.set_as(int32_ub[0])


def vconv_int32_scalar_to_fp16(tik_instance, input_int32_scalar, fp16_scalar):
    with tik_instance.new_stmt_scope():
        scalar_int32 = tik_instance.Scalar("int32", name="scalar_int32",
                                           init_value=0)
        scalar_int32.set_as(input_int32_scalar)
        int32_ub = tik_instance.Tensor("int32", (8,), name="int32_ub",
                                       scope=tik.scope_ubuf)
        float16_ub = tik_instance.Tensor("float16", (16,), name="float16_ub",
                                         scope=tik.scope_ubuf)
        int32_ub[0].set_as(scalar_int32)
        tik_instance.vconv(1, "", float16_ub, int32_ub, 1, 1, 1, 4, 8,
                           deqscale=1.0)
        fp16_scalar.set_as(float16_ub[0])


def copy_block_to_ub(tik_instance, dst, dst_off, src, src_off, off_scalar,
                     element_bytes):
    with tik_instance.new_stmt_scope():
        cmp_scalar = tik_instance.Scalar("int32", name="cmp_scalar",
                                         init_value=-1)

        # cmp_scalar use as tmp scalar
        cmp_scalar.set_as(off_scalar % 8)
        with tik_instance.if_scope(cmp_scalar == 0):
            off_scalar.set_as(off_scalar // 8 - 1)
            cmp_scalar.set_as(7)

        with tik_instance.else_scope():
            cmp_scalar.set_as(off_scalar % 8 - 1)
            off_scalar.set_as(off_scalar // 8)

        data_move_align256(tik_instance, src, src_off, dst,
                           dst_off + cmp_scalar, 1, element_bytes)


def cnt_proposals(tik_instance, proposals_ub, proposls_num, target_ub):
    with tik_instance.new_stmt_scope():
        with tik_instance.for_range(0, proposls_num) as cnt:
            target_ub[cnt].set_as(proposals_ub[8 * cnt + 4])
