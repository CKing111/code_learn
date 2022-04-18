"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""
import numpy as np
import onnx

def stringify(l):
    return '_'.join([str(x) for x in l])

def save_data(op_name, input_shape, output_shape, input_data, expect_data):
    input_data_path = f"./input_data/{op_name}_in_{stringify(input_shape)}.bin"
    gt_save_path = f"./ground_truth/{op_name}_gt_{stringify(output_shape)}.bin"
    input_data.tofile(input_data_path)
    expect_data.tofile(gt_save_path)

def generate_model(input_shape, output_shape, model_save_path, **kwargs):

    graph_def = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node(
                op_type='ReduceSumSquare',
                inputs=['input_x'],
                outputs=['output_y'],
                axes=kwargs['axes'],
                keepdims=kwargs['keepdims']
            )
        ],
        name='reduce_sum_square',
        inputs=[
            onnx.helper.make_tensor_value_info('input_x', onnx.TensorProto.FLOAT, input_shape)
        ],
        outputs=[
            onnx.helper.make_tensor_value_info('output_y', onnx.TensorProto.FLOAT, output_shape)
        ],
    )
    model_def = onnx.helper.make_model(graph_def,
                                       producer_name='reduce_sum_square')
    model_def.opset_import[0].version = 11
    onnx.checker.check_model(model_def)

    onnx.save(model_def, model_save_path)

def reduce_sum_square(input_shape, output_shape, axes=None, keepdims=1, input_dtype=np.float32):
    input_data = np.random.random_sample(size=(3,4,3)).astype(input_dtype)+1
    print(input_data)
    data = np.square(input_data)
    if axes is None:
        expect_data = np.sum(data,axis = axes,keepdims = True)
    else:
        expect_data = np.sum(data,axis = tuple(axes),keepdims = True)

    print(expect_data)
    save_data("reduce_sum_square", input_shape, output_shape, input_data, expect_data)



if __name__ == "__main__":

    op_name = "reduce_sum_square"
    input_shape = [3, 4, 3]
    output_shape = [3, 4, 1]
    axes = [2]
    keepdims = 1

    model_save_path = f"./model/{op_name}_{stringify(input_shape)}.onnx"

    generate_model(input_shape, output_shape, model_save_path, axes=axes, keepdims=keepdims)

    reduce_sum_square(input_shape, output_shape, axes=axes, keepdims=keepdims, input_dtype=np.float32)

    print("Done")
