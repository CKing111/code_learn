#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ExpandD
"""

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_shape
from topi import generic

NUM_ONE = 1


@fusion_manager.register("expand_d")
def expand_compute(_x,
                   input_y,
                   shape):
    """
    Process expand operator.

    Parameters:
    ----------
    _x : the input tensor.
    y : the dict of output.
    shape : the desired output shape.
    kernel_name : cce kernel name, default value is "expand_d".

    Returns:
    -------
    output_tensor : tensor after expand.
    """
    dtype = _x.dtype
    shape_in = _x.shape

    # te.lang.cce.broadcast supports float16, float32, int32.
    # so convert int8, uint8 to float16
    if dtype in ('int8', 'uint8'):
        _x = te.lang.cce.cast_to(_x, 'float16')

    python_shape_in = [int(x) for x in shape_in]
    if list(python_shape_in) == list(shape):
        if dtype == "int32":
            value_one = tvm.const(NUM_ONE, dtype=dtype)
            value_one_tensor = te.lang.cce.broadcast(value_one, shape)
            output_tensor = te.lang.cce.vmul(_x, value_one_tensor)
        else:
            output_tensor = te.lang.cce.vmuls(_x, NUM_ONE)
    else:
        output_tensor = te.lang.cce.broadcast(_x, shape, dtype)

    # convert float16 back to int8, uint8
    if dtype in ('int8', 'uint8'):
        return te.lang.cce.cast_to(output_tensor, dtype, f1628IntegerFlag=True)
    return output_tensor


def _check_shape_compatibility(shape_in, shape):
    """
    Check if the shape of input tensor is compatible with output tensor.

    Parameters:
    ----------
    shape_in : shape of input tensor.

    shape : shape of output tensor.

    Returns:
    -------
    comp_shape_in : new shape_in compatible with shape.
    """

    try:
        comp_shape_in, comp_shape, shape_max = broadcast_shapes(
            shape_in, shape, param_name_input1="x", param_name_input2="shape")
    except RuntimeError:
        raise ValueError('shape_in is not compatible with shape_out.')

    return comp_shape_in, comp_shape, shape_max


def expand_d(input_x,
             input_y,
             shape,
             kernel_name="expand_d"):
    """
    Broadcast an array for a compatible shape.

    Parameters:
    ----------
    x : the dict of input. support data type: 
        float32, float16, int8, uint8, int32.
    y : the dict of output.
    shape : the other shape which needed to be broadcasted .
    kernel_name : cce kernel name, default value is "expand_d".

    Returns:
    -------
    None
    """
    check_list = ('float32', 'float16', 'int8', 'uint8', 'int32')
    x_dtype = input_x.get('dtype').lower()
    check_dtype(x_dtype, check_list, param_name="x")

    shape_in = input_x.get('shape')
    check_shape(shape_in, param_name="x")
    check_shape(shape, param_name="shape")

    compatible_shape_in, _, shape_max = _check_shape_compatibility(shape_in,
                                                                   shape)
    var = tvm.placeholder(compatible_shape_in, x_dtype, name='data_input')

    res = expand_compute(var, input_y, shape_max)
    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [var, res]}
    te.lang.cce.cce_build_code(sch, config)
