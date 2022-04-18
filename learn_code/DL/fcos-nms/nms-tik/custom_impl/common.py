from te import tik

FLOAT16 = 'float16'


def cos(tik_instance, dst, src0):
    cos_first_order = 4
    cos_last_order = 16
    cos_factor = -1.0 / 2.0
    tmp_ub = tik_instance.Tensor(FLOAT16, src0.shape, scope=tik.scope_ubuf,
                                 name="tmp_ub")
    tmp_ub_mul = tik_instance.Tensor(FLOAT16, src0.shape,
                                     scope=tik.scope_ubuf, name="tmp_ub_mul")
    vec_mul(tik_instance, FLOAT16, tmp_ub, src0, src0)
    vec_muls(tik_instance, FLOAT16, tmp_ub_mul, tmp_ub, cos_factor)
    vec_adds(tik_instance, FLOAT16, dst, tmp_ub_mul, 1)
    i = cos_first_order
    while i < cos_last_order:
        vec_mul(tik_instance, FLOAT16, tmp_ub_mul, tmp_ub, tmp_ub_mul)
        vec_muls(tik_instance, FLOAT16, tmp_ub_mul, tmp_ub_mul,
                 -1.0 / (i * (i - 1)))
        vec_add(tik_instance, FLOAT16, dst, dst, tmp_ub_mul)
        i = i + 2
    return dst


PI = 3.14159265358979


def sin(tik_instance, dst, src0):
    sin_first_order = 5
    sin_last_order = 13
    sin_factor = -1.0 / 6.0
    tmp_ub = tik_instance.Tensor(FLOAT16, src0.shape, scope=tik.scope_ubuf,
                                 name="tmp_ub")
    tmp_ub_mul = tik_instance.Tensor(FLOAT16, src0.shape,
                                     scope=tik.scope_ubuf, name="tmp_ub_mul")
    tmp_cos = tik_instance.Tensor(FLOAT16, src0.shape, scope=tik.scope_ubuf,
                                  name="tmp_cos")
    src_1 = tik_instance.Tensor(FLOAT16, src0.shape, scope=tik.scope_ubuf,
                                name="src_1")
    vec_dup(tik_instance, FLOAT16, src_1, PI / 2)
    vec_sub(tik_instance, FLOAT16, src_1, src0, src_1)
    tmp_cos = cos(tik_instance, tmp_cos, src_1)
    vec_mul(tik_instance, FLOAT16, tmp_ub, src0, src0)
    vec_muls(tik_instance, FLOAT16, tmp_ub_mul, tmp_ub, sin_factor)
    vec_add(tik_instance, FLOAT16, dst, tmp_ub_mul, src0)
    i = sin_first_order
    while i < sin_last_order:
        vec_mul(tik_instance, FLOAT16, tmp_ub_mul, tmp_ub, tmp_ub_mul)
        vec_muls(tik_instance, FLOAT16, tmp_ub_mul, tmp_ub_mul,
                 -1.0 / (i * (i - 1)))
        vec_add(tik_instance, FLOAT16, dst, dst, tmp_ub_mul)
        i = i + 2
    threshold_ub = tik_instance.Tensor(FLOAT16, (128,), name="threshold_ub",
                                       scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, threshold_ub, PI / 2, 1, 1, 8)
    vec_cmp_gt(tik_instance, src0, tmp_cos, dst, threshold_ub)
    return dst


def compute_param(dtype, tensor):
    if dtype == "float16" or dtype == "int16":
        mask = 128
    elif dtype == "float32" or dtype == "int32":
        mask = 64
    repeat = tensor.shape[0] // mask // 255
    tail = tensor.shape[0] % (mask * 255)
    tail_repeat = tail // mask
    tail_mask = tail % mask

    return mask, repeat, tail_repeat, tail_mask


def vec_dup(tik_instance, dtype, dst, scalar):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vector_dup(mask, dst[255 * mask * i], scalar, 255, 1,
                                    8)
    if tail_repeat > 0:
        tik_instance.vector_dup(mask, dst[255 * mask * repeat], scalar,
                                tail_repeat, 1, 8)
    if tail_mask > 0:
        tik_instance.vector_dup(tail_mask,
                                dst[(255 * repeat + tail_repeat) * mask],
                                scalar, 1, 1, 8)


def vconv_int32tofp16(tik_instance, dtype, src, dst):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, src)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vconv(mask, '', dst[255 * mask * i],
                               src[255 * mask * i], 255, 1, 1, 4, 8, 1.0)
    if tail_repeat > 0:
        tik_instance.vconv(mask, '', dst[255 * mask * repeat],
                           src[255 * mask * repeat], tail_repeat, 1, 1, 4, 8,
                           1.0)
    if tail_mask > 0:
        tik_instance.vconv(tail_mask, '',
                           dst[(255 * repeat + tail_repeat) * mask],
                           src[(255 * repeat + tail_repeat) * mask], 1, 1, 1,
                           4, 8, 1.0)


def vec_muls(tik_instance, dtype, dst, src, scalar):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vmuls(mask, dst[255 * mask * i], src[255 * mask * i],
                               scalar, 255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vmuls(mask, dst[255 * mask * repeat],
                           src[255 * mask * repeat],
                           scalar, tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vmuls(tail_mask,
                           dst[(255 * repeat + tail_repeat) * mask],
                           src[(255 * repeat + tail_repeat) * mask],
                           scalar, 1, 1, 1, 8, 8)


def vec_mul(tik_instance, dtype, dst, src0, src1):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vmul(mask, dst[255 * mask * i], src0[255 * mask * i],
                              src1[255 * mask * i], 255, 1, 1, 1, 8, 8, 8)
    if tail_repeat > 0:
        tik_instance.vmul(mask, dst[255 * mask * repeat],
                          src0[255 * mask * repeat],
                          src1[255 * mask * repeat], tail_repeat, 1, 1, 1, 8,
                          8, 8)
    if tail_mask > 0:
        tik_instance.vmul(tail_mask, dst[(255 * repeat + tail_repeat) * mask],
                          src0[(255 * repeat + tail_repeat) * mask],
                          src1[(255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 1, 8, 8, 8)


def vec_adds(tik_instance, dtype, dst, src, scalar):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vadds(mask, dst[255 * mask * i], src[255 * mask * i],
                               scalar, 255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vadds(mask, dst[255 * mask * repeat],
                           src[255 * mask * repeat],
                           scalar, tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vadds(tail_mask,
                           dst[(255 * repeat + tail_repeat) * mask],
                           src[(255 * repeat + tail_repeat) * mask],
                           scalar, 1, 1, 1, 8, 8)


def vec_add(tik_instance, dtype, dst, src0, src1):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vadd(mask, dst[255 * mask * i], src0[255 * mask * i],
                              src1[255 * mask * i], 255, 1, 1, 1, 8, 8, 8)
    if tail_repeat > 0:
        tik_instance.vadd(mask, dst[255 * mask * repeat],
                          src0[255 * mask * repeat],
                          src1[255 * mask * repeat], tail_repeat, 1, 1, 1, 8,
                          8, 8)
    if tail_mask > 0:
        tik_instance.vadd(tail_mask, dst[(255 * repeat + tail_repeat) * mask],
                          src0[(255 * repeat + tail_repeat) * mask],
                          src1[(255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 1, 8, 8, 8)


def vec_exp(tik_instance, dtype, dst, src):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vexp(mask, dst[255 * mask * i], src[255 * mask * i],
                              255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vexp(mask, dst[255 * mask * repeat],
                          src[255 * mask * repeat],
                          tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vexp(tail_mask, dst[(255 * repeat + tail_repeat) * mask],
                          src[(255 * repeat + tail_repeat) * mask], 1, 1, 1,
                          8, 8)


def vec_abs(tik_instance, dtype, dst, src):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vabs(mask, dst[255 * mask * i], src[255 * mask * i],
                              255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vabs(mask, dst[255 * mask * repeat],
                          src[255 * mask * repeat],
                          tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vabs(tail_mask, dst[(255 * repeat + tail_repeat) * mask],
                          src[(255 * repeat + tail_repeat) * mask], 1, 1, 1,
                          8, 8)


def vec_sub(tik_instance, dtype, dst, src0, src1):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vsub(mask, dst[255 * mask * i], src0[255 * mask * i],
                              src1[255 * mask * i], 255, 1, 1, 1, 8, 8, 8)
    if tail_repeat > 0:
        tik_instance.vsub(mask, dst[255 * mask * repeat],
                          src0[255 * mask * repeat],
                          src1[255 * mask * repeat], tail_repeat, 1, 1, 1, 8,
                          8, 8)
    if tail_mask > 0:
        tik_instance.vsub(tail_mask, dst[(255 * repeat + tail_repeat) * mask],
                          src0[(255 * repeat + tail_repeat) * mask],
                          src1[(255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 1, 8, 8, 8)


def vec_cmp_gt(tik_instance, cmp_data, filter_data_1, filter_data_2,
               threshold_ub):
    length = cmp_data.shape[0]
    repeat = length // 128
    tail = length % 128
    with tik_instance.for_range(0, repeat) as i:
        cmpmask = tik_instance.vcmp_gt(128, cmp_data[i * 128], threshold_ub,
                                       1, 1)
        tik_instance.vsel(128, 0, filter_data_2[i * 128], cmpmask,
                          filter_data_1[i * 128],
                          filter_data_2[i * 128], 1, 1, 1, 1, 8, 8, 8)
    if tail > 0:
        cmpmask = tik_instance.vcmp_gt(tail, cmp_data[repeat * 128],
                                       threshold_ub, 1, 1)
        tik_instance.vsel(tail, 0, filter_data_2[repeat * 128], cmpmask,
                          filter_data_1[repeat * 128],
                          filter_data_2[repeat * 128], 1, 1, 1, 1, 8, 8, 8)


def vec_vln(tik_instance, dtype, dst, src):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vln(mask, dst[255 * mask * i], src[255 * mask * i],
                             255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vln(mask, dst[255 * mask * repeat],
                         src[255 * mask * repeat],
                         tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vln(tail_mask, dst[(255 * repeat + tail_repeat) * mask],
                         src[(255 * repeat + tail_repeat) * mask], 1, 1, 1, 8,
                         8)


def vec_vrec(tik_instance, dtype, dst, src):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vrec(mask, dst[255 * mask * i], src[255 * mask * i],
                              255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vrec(mask, dst[255 * mask * repeat],
                          src[255 * mask * repeat],
                          tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vrec(tail_mask, dst[(255 * repeat + tail_repeat) * mask],
                          src[(255 * repeat + tail_repeat) * mask], 1, 1, 1,
                          8, 8)


def vec_vsel(tik_instance, cmp_data, filter_data_1, filter_data_2, cmpmask):
    length = cmp_data.shape[0]
    repeat = length // 128
    tail = length % 128
    with tik_instance.for_range(0, repeat) as i:
        tik_instance.vsel(128, 0, cmp_data[i * 128], cmpmask,
                          filter_data_1[i * 128],
                          filter_data_2[i * 128], 1, 1, 1, 1, 8, 8, 8)
    if tail > 0:
        tik_instance.vsel(tail, 0, cmp_data[repeat * 128], cmpmask,
                          filter_data_1[repeat * 128],
                          filter_data_2[repeat * 128], 1, 1, 1, 1, 8, 8, 8)
