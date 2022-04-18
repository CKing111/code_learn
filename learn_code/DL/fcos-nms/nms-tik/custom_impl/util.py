import re
from te import tik

MAX_KERNEL_NAME_LEN = 200


def factor(value):
    if value == 1 or value == 0:
        return 1
    else:
        fac = 1
        for i in range(1, value + 1):
            fac = fac * i
        return fac


def type_byte(dtype):
    if dtype == "float32":
        return 4
    elif dtype == "float16":
        return 2
    elif dtype == "int8":
        return 1


def singleton(cls):
    _instance = {}
    if cls not in _instance:
        _instance[cls] = cls()
    return _instance[cls]


@singleton
class OpLog(object):
    '''
    OpLog: operator log class
    '''
    def __init__(self):
        pass

    def check(self, condition, error_msg):
        if condition:
            raise RuntimeError(error_msg)

    def check_eq(self, obj, value, error_msg):
        if obj != value:
            raise RuntimeError(error_msg)

    def check_ne(self, obj, value, error_msg):
        if obj == value:
            raise RuntimeError(error_msg)

    def check_le(self, obj, value, error_msg):
        if obj > value:
            raise RuntimeError(error_msg)

    def check_lt(self, obj, value, error_msg):
        if obj >= value:
            raise RuntimeError(error_msg)

    def check_ge(self, obj, value, error_msg):
        if obj < value:
            raise RuntimeError(error_msg)

    def check_gt(self, obj, value, error_msg):
        if obj <= value:
            raise RuntimeError(error_msg)

    def check_kernelname(self, kernel_name):
        if kernel_name is None:
            raise RuntimeError("kernel_name can not be None, but got %s" %
                            type(kernel_name))

        if not isinstance(kernel_name, str):
            raise RuntimeError("kernel_name must be string, but got %s" %
                            type(kernel_name))

        if len(kernel_name) > MAX_KERNEL_NAME_LEN:
            raise RuntimeError(
                "kernel_name len must be less than %d, but got %d" %
                (MAX_KERNEL_NAME_LEN, len(kernel_name)))

        pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        if not pattern.match(kernel_name):
            raise RuntimeError(
                "kernel_name can only contain letters, \
                numbers and underscores, \
                and begin with underscores or letters")


def compute_param(dtype, tensor):
    if dtype == "float16":
        mask = 128
    elif dtype == "float32":
        mask = 64
    repeat = tensor.shape[1] // mask // 255
    tail = tensor.shape[1] % (mask * 255)
    tail_repeat = tail // mask
    tail_mask = tail % mask
    return mask, repeat, tail_repeat, tail_mask


def vec_dup(tik_instance, dtype, dst, scalar, dst_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vector_dup(mask, dst[dst_row, 255 * mask * i],
                                    scalar, 255, 1, 8)
    if tail_repeat > 0:
        tik_instance.vector_dup(mask, dst[dst_row, 255 * mask * repeat],
                                scalar, tail_repeat, 1, 8)
    if tail_mask > 0:
        tik_instance.vector_dup(tail_mask, dst[
            dst_row, (255 * repeat + tail_repeat) * mask], scalar, 1, 1, 8)


def vec_exp(tik_instance, dtype, dst, src, dst_row, src_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vexp(mask, dst[dst_row, 255 * mask * i],
                              src[src_row, 255 * mask * i], 255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vexp(mask, dst[dst_row, 255 * mask * repeat],
                          src[src_row, 255 * mask * repeat],
                          tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vexp(tail_mask,
                          dst[dst_row, (255 * repeat + tail_repeat) * mask],
                          src[src_row, (255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 8, 8)


def vec_maxs(tik_instance, dtype, dst, src, scalar, dst_row, src_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    threshold = tik_instance.Tensor(dtype, (mask,), name="threshold",
                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(mask, threshold, scalar, 1, 1, 8)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vmax(mask, dst[dst_row, 255 * mask * i],
                              src[src_row, 255 * mask * i],
                              threshold, 255, 1, 1, 1, 8, 8, 0)
    if tail_repeat > 0:
        tik_instance.vmax(mask, dst[dst_row, 255 * mask * repeat],
                          src[src_row, 255 * mask * repeat],
                          threshold, tail_repeat, 1, 1, 1, 8, 8, 0)
    if tail_mask > 0:
        tik_instance.vmax(tail_mask,
                          dst[dst_row, (255 * repeat + tail_repeat) * mask],
                          src[src_row, (255 * repeat + tail_repeat) * mask],
                          threshold, 1, 1, 1, 1, 8, 8, 0)


def vec_mins(tik_instance, dtype, dst, src, scalar, dst_row, src_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    threshold = tik_instance.Tensor(dtype, (mask,), name="threshold",
                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(mask, threshold, scalar, 1, 1, 8)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vmin(mask, dst[dst_row, 255 * mask * i],
                              src[src_row, 255 * mask * i], threshold, 255, 1,
                              1, 1, 8, 8, 0)
    if tail_repeat > 0:
        tik_instance.vmin(mask, dst[dst_row, 255 * mask * repeat],
                          src[src_row, 255 * mask * repeat],
                          threshold, tail_repeat, 1, 1, 1, 8, 8, 0)
    if tail_mask > 0:
        tik_instance.vmin(tail_mask,
                          dst[dst_row, (255 * repeat + tail_repeat) * mask],
                          src[src_row, (255 * repeat + tail_repeat) * mask],
                          threshold, 1, 1, 1, 1, 8, 8, 0)


def vec_adds(tik_instance, dtype, dst, src, scalar, dst_row, src_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vadds(mask, dst[dst_row, 255 * mask * i],
                               src[src_row, 255 * mask * i],
                               scalar, 255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vadds(mask, dst[dst_row, 255 * mask * repeat],
                           src[src_row, 255 * mask * repeat],
                           scalar, tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vadds(tail_mask,
                           dst[dst_row, (255 * repeat + tail_repeat) * mask],
                           src[src_row, (255 * repeat + tail_repeat) * mask],
                           scalar, 1, 1, 1, 8, 8)


def vec_sub(tik_instance, dtype, dst, src0, src1, dst_row, src0_row,
            src1_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vsub(mask, dst[dst_row, 255 * mask * i],
                              src0[src0_row, 255 * mask * i],
                              src1[src1_row, 255 * mask * i], 255, 1, 1, 1, 8,
                              8, 8)
    if tail_repeat > 0:
        tik_instance.vsub(mask, dst[dst_row, 255 * mask * repeat],
                          src0[src0_row, 255 * mask * repeat],
                          src1[src1_row, 255 * mask * repeat], tail_repeat, 1,
                          1, 1, 8, 8, 8)
    if tail_mask > 0:
        tik_instance.vsub(tail_mask,
                          dst[dst_row, (255 * repeat + tail_repeat) * mask],
                          src0[src0_row, (255 * repeat + tail_repeat) * mask],
                          src1[src1_row, (255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 1, 8, 8, 8)


def vec_add(tik_instance, dtype, dst, src0, src1, dst_row, src0_row,
            src1_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vadd(mask, dst[dst_row, 255 * mask * i],
                              src0[src0_row, 255 * mask * i],
                              src1[src1_row, 255 * mask * i], 255, 1, 1, 1, 8,
                              8, 8)
    if tail_repeat > 0:
        tik_instance.vadd(mask, dst[dst_row, 255 * mask * repeat],
                          src0[src0_row, 255 * mask * repeat],
                          src1[src1_row, 255 * mask * repeat], tail_repeat, 1,
                          1, 1, 8, 8, 8)
    if tail_mask > 0:
        tik_instance.vadd(tail_mask,
                          dst[dst_row, (255 * repeat + tail_repeat) * mask],
                          src0[src0_row, (255 * repeat + tail_repeat) * mask],
                          src1[src1_row, (255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 1, 8, 8, 8)


def vec_mul(tik_instance, dtype, dst, src0, src1, dst_row, src0_row,
            src1_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vmul(mask, dst[dst_row, 255 * mask * i],
                              src0[src0_row, 255 * mask * i],
                              src1[src1_row, 255 * mask * i], 255, 1, 1, 1, 8,
                              8, 8)
    if tail_repeat > 0:
        tik_instance.vmul(mask, dst[dst_row, 255 * mask * repeat],
                          src0[src0_row, 255 * mask * repeat],
                          src1[src1_row, 255 * mask * repeat], tail_repeat, 1,
                          1, 1, 8, 8, 8)
    if tail_mask > 0:
        tik_instance.vmul(tail_mask,
                          dst[dst_row, (255 * repeat + tail_repeat) * mask],
                          src0[src0_row, (255 * repeat + tail_repeat) * mask],
                          src1[src1_row, (255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 1, 8, 8, 8)


def vec_muls(tik_instance, dtype, dst, src, scalar, dst_row, src_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vmuls(mask, dst[dst_row, 255 * mask * i],
                               src[src_row, 255 * mask * i],
                               scalar, 255, 1, 1, 8, 8)
    if tail_repeat > 0:
        tik_instance.vmuls(mask, dst[dst_row, 255 * mask * repeat],
                           src[src_row, 255 * mask * repeat],
                           scalar, tail_repeat, 1, 1, 8, 8)
    if tail_mask > 0:
        tik_instance.vmuls(tail_mask,
                           dst[dst_row, (255 * repeat + tail_repeat) * mask],
                           src[src_row, (255 * repeat + tail_repeat) * mask],
                           scalar, 1, 1, 1, 8, 8)


def vec_div(tik_instance, dtype, dst, src0, src1, dst_row, src0_row,
            src1_row):
    mask, repeat, tail_repeat, tail_mask = compute_param(dtype, dst)
    if repeat > 0:
        for i in range(repeat):
            tik_instance.vrec(mask, src1[src1_row, 255 * mask * i],
                              src1[src1_row, 255 * mask * i], 255, 1, 1, 8, 8)
            tik_instance.vmul(mask, dst[dst_row, 255 * mask * i],
                              src0[src0_row, 255 * mask * i],
                              src1[src1_row, 255 * mask * i], 255, 1, 1, 1, 8,
                              8, 8)
    if tail_repeat > 0:
        tik_instance.vrec(mask, src1[src1_row, 255 * mask * repeat],
                          src1[src1_row, 255 * mask * repeat],
                          tail_repeat, 1, 1, 8, 8)
        tik_instance.vmul(mask, dst[dst_row, 255 * mask * repeat],
                          src0[src0_row, 255 * mask * repeat],
                          src1[src1_row, 255 * mask * repeat], tail_repeat, 1,
                          1, 1, 8, 8, 8)
    if tail_mask > 0:
        tik_instance.vrec(mask,
                          src1[src1_row, (255 * repeat + tail_repeat) * mask],
                          src1[src1_row, (255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 8, 8)
        tik_instance.vmul(tail_mask,
                          dst[dst_row, (255 * repeat + tail_repeat) * mask],
                          src0[src0_row, (255 * repeat + tail_repeat) * mask],
                          src1[src1_row, (255 * repeat + tail_repeat) * mask],
                          1, 1, 1, 1, 8, 8, 8)


def vec_cmp(tik_instance, cmp_type, cmp_data, filter_data, threshold,
            cmp_rows, filter_row):
    threshold_ub = tik_instance.Tensor("float16", (128,), name="threshold_ub",
                                       scope=tik.scope_ubuf)
    zeros_ub = tik_instance.Tensor("float16", (128,), name="zeros_ub",
                                   scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, threshold_ub, threshold, 1, 1, 8)
    tik_instance.vector_dup(128, zeros_ub, 0, 1, 1, 8)

    length = cmp_data.shape[1]
    repeat = length // 128
    tail = length % 128
    with tik_instance.for_range(0, repeat) as i:
        if cmp_type == "GT":
            cmpmask = tik_instance.vcmp_gt(128, cmp_data[cmp_rows, i * 128],
                                           threshold_ub, 1, 1)
        elif cmp_type == "EQ":
            cmpmask = tik_instance.vcmp_eq(128, cmp_data[cmp_rows, i * 128],
                                           threshold_ub, 1, 1)
        elif cmp_type == "LT":
            cmpmask = tik_instance.vcmp_lt(128, cmp_data[cmp_rows, i * 128],
                                           threshold_ub, 1, 1)
        tik_instance.vsel(128, 0, filter_data[filter_row, i * 128], cmpmask,
                          zeros_ub,
                          filter_data[filter_row, i * 128], 1, 1, 1, 1, 8, 8,
                          8)
    if cmp_type == "GT":
        cmpmask = tik_instance.vcmp_gt(tail, cmp_data[cmp_rows, repeat * 128],
                                       threshold_ub, 1, 1)
    elif cmp_type == "EQ":
        cmpmask = tik_instance.vcmp_eq(tail, cmp_data[cmp_rows, repeat * 128],
                                       threshold_ub, 1, 1)
    elif cmp_type == "LT":
        cmpmask = tik_instance.vcmp_lt(tail, cmp_data[cmp_rows, repeat * 128],
                                       threshold_ub, 1, 1)
    tik_instance.vsel(tail, 0, filter_data[filter_row, repeat * 128], cmpmask,
                      zeros_ub, filter_data[filter_row, repeat * 128], 1, 1,
                      1, 1, 8, 8, 8)
