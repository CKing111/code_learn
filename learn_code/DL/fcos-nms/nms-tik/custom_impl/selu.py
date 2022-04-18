import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te import tvm

# define selu oprator's required constants
ALPHA = 1.67326324235
SCALE = 1.05070098736
# define product of scale and alpha
SCALE_ALPHA_PRODUCT = 1.75809934085
# define a scalar, value = -1, the calculation of exp need minus one
SCALAR_NEGATIVE_ONE = -1


@tbe_platform.fusion_manager.fusion_manager.register("selu")
def selu_compute(input_x, input_y, alpha, gamma, kernel_name="selu"):
    """
    Computes scaled exponential linear: `gamma * alpha * (exp(features) - 1)`
    if < 0, `gamma * features` otherwise.
    alpha =  1.67326324235437728481704299167176
    gamma =  1.0507009873554804934193349852946

    Parameters
    ----------
    input_x: TVM tensors
        input tensor has shape and dtype attribute
    y: TVM tensor
        outputtensor has shape and dtype attributes
    alpha: float
    gamma: float   
    kernel_name : str
        cce kernel name, default value is "selu"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """
    # define product of scale and alpha
    gamma_alpha_product = alpha * gamma

    # if input_dtype is float16,convert it to float32
    input_data = input_x
    dtype = input_data.dtype
    if dtype in ("float16", "float32"):
        input_data = tbe.cast_to(input_data, "float32")
        type_tmp = "float32"
    else:
        input_data = tbe.cast_to(input_data, "float16")
        type_tmp = "float16"

    # generate tensor_zero to be compared
    tensor_zero = tbe.vmuls(input_data, tvm.const(0, dtype=type_tmp))
    # generate negative_res and positive_res to compute
    # When the element value is greater than 0 and less than 0
    negative_res = tbe.vmin(input_data, tensor_zero)
    positive_res = tbe.vmax(input_data, tensor_zero)
    exp_res = tbe.vexp(negative_res)
    sub_res = tbe.vadds(exp_res, tvm.const(SCALAR_NEGATIVE_ONE,
                                            dtype=type_tmp))
    negative_muls_res = tbe.vmuls(sub_res, tvm.const(gamma_alpha_product,
                                                    dtype=type_tmp))
    if dtype == "int8":
        negative_muls_res = tbe.ceil(negative_muls_res)

    positive_muls_res = tbe.vmuls(positive_res, tvm.const(gamma,
                                                        dtype=type_tmp))
    res = tbe.vadd(negative_muls_res, positive_muls_res)
    # if input_dtype is float16, has converted to float32,
    # output should convert back
    if dtype in ("float16", "int8", "int32"):
        res = tbe.cast_to(res, dtype)

    return res


# @para_check.check_op_params(para_check.REQUIRED_INPUT,
# para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def selu(in_x, in_y, alpha, gamma, kernel_name="selu"):
    """
    Generate selu_cce operator use selu_compute

    Parameter
    ----------
    x: dict
        dict{"shape":tuple or list,"dtype":str}
        shape of data, assume src_shape equals dst_shape,
        the data type, src_dtype equals dst_dtype,
         support fp16, fp32, int8, int32
    y: dict
        dict with keys(shape and dtype) of output
    alpha: float
        alpha attr
    gamma: float
        gamma attr
    kernel_name: str
        cce kernel name, default value is "selu"

    Returns
    ------
    None
    """

    # get dtype and shape attributes
    dtype = in_x.get("dtype")
    shape = in_x.get("shape")
    # check_kernel_name & shape
    input_dtype = dtype.lower()
    para_check.check_shape(shape, param_name="in_x")
    # check input tensor data_type
    check_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(input_dtype, check_list, param_name="in_x")

    reshape_input = (functools.reduce(lambda in_x, in_y: in_x * in_y,
                                     shape[:]),)
    input_data = tvm.placeholder(reshape_input, name="input_data",
                                 dtype=input_dtype)
    res = selu_compute(input_data, in_y, alpha, gamma, kernel_name)
    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {"name": alpha,
              "name": gamma,
              "name": kernel_name,
              "tensor_list": [input_data, res]}
    tbe.cce_build_code(auto_sch, config)
