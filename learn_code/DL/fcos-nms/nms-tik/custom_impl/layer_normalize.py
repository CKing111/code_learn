from uti import interface_check
from version import get_version
from .layer_normalize_split_d import LayerNormalizeBase
from .layer_normalize_split_d import LayerNormalizeSplitD
from .layer_normalize_split_n import LayerNormalizeSplitN


def check_params(input_tensor, weight, bias, output_tensor):
    support_shape_length = [shape_length for shape_length in range(1, 9)]
    support_dtype = ["float16", "float32"]
    support_format = ["NCHW", "ND"]
    interface_check.check_param(input_tensor, support_shape_length,
                                support_dtype, support_format)
    interface_check.check_param(weight, support_shape_length, support_dtype,
                                support_format)
    interface_check.check_param(bias, support_shape_length, support_dtype,
                                support_format)
    interface_check.check_param(output_tensor, support_shape_length,
                                support_dtype, support_format)


def check_attrs(eps, elementwise, dims):
    if not isinstance(eps, float) or eps <= 0:
        raise RuntimeError(
            "[ERROR][LayerNormalize] param eps is not supported")
    if not isinstance(elementwise, bool):
        raise RuntimeError(
            "[ERROR][LayerNormalize] param elementwise is not supported")
    if not isinstance(dims, int):
        raise RuntimeError(
            "[ERROR][LayerNormalize] param dims is not supported")


def check_shape(input_tensor, weight, bias, output_tensor, elementwise, dims):
    input_tensor_shape = input_tensor.get("shape")
    output_tensor_shape = output_tensor.get("shape")
    weight_shape = weight.get("shape")
    bias_shape = bias.get("shape")
    if dims < - len(input_tensor_shape) or dims >= len(input_tensor_shape):
        raise RuntimeError(
            "[ERROR][LayerNormalize] param dims is not supported")
    if output_tensor_shape != input_tensor_shape:
        raise RuntimeError(
            "[ERROR][LayerNormalize] output shape is not supported")
    if elementwise:
        if weight_shape != input_tensor_shape[dims:]:
            raise RuntimeError(
                "[ERROR][LayerNormalize] weight shape is not supported")
        if bias_shape != input_tensor_shape[dims:]:
            raise RuntimeError(
                "[ERROR][LayerNormalize] bias shape is not supported")


def check_params_attrs_name(input_tensor, weight, bias, output_tensor, eps,
                            elementwise, dims, kernel_name):
    interface_check.check_kernelname(kernel_name)
    check_params(input_tensor, weight, bias, output_tensor)
    check_attrs(eps, elementwise, dims)
    check_shape(input_tensor, weight, bias, output_tensor, elementwise, dims)


def check_batch_data(batch_num, data_num, format_num):
    max_data_num = 60000000
    if data_num < format_num:
        raise RuntimeError("[ERROR][LayerNormalize] data num is not supported")
    if batch_num * data_num > max_data_num:
        raise RuntimeError("[ERROR][LayerNormalize] data num is not supported")


def select_layer_normalize(batch_num, data_num, format_num, cont, ai_core_use):
    if batch_num <= ai_core_use:
        ub_size = cont.const_ub_max_byte
    else:
        ub_size = cont.const_ub_max_byte // 2
    if data_num <= LayerNormalizeBase.mode_n_split_max_num(ub_size, format_num,
                                                           cont):
        return LayerNormalizeSplitN
    else:
        return LayerNormalizeSplitD


def layer_normalize(input_tensor, weight, bias, output_tensor, eps,
                    elementwise, dims, kernel_name="LayerNormalize",
                    test=False):
    check_params_attrs_name(input_tensor, weight, bias, output_tensor, eps,
                            elementwise, dims, kernel_name)
    cont = get_version.get_aicore_container(("Ascend610", ))
    format_num, if_fp16 = \
        LayerNormalizeBase.get_ub_format_num(input_tensor, weight, bias,
                                             output_tensor, elementwise, cont)
    batch_num, data_num, dims = \
        LayerNormalizeBase.split_shape(input_tensor, dims)
    check_batch_data(batch_num, data_num, format_num)
    ai_core_use = cont.const_aicore_num

    class_layer_normalize = select_layer_normalize(batch_num, data_num,
                                                   format_num, cont,
                                                   ai_core_use)
    obj_layer_normalize = class_layer_normalize(
        input_tensor, weight, bias, output_tensor, eps, elementwise, dims,
        kernel_name,
        batch_num, data_num, format_num, if_fp16, ai_core_use, cont)

    obj_layer_normalize.mode_compute()
    if test:
        obj_layer_normalize.tik_output_debug()
