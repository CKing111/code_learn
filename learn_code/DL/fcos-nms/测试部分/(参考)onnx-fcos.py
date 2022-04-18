import argparse
import os.path as osp
import warnings
 
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv import DictAction
 
from mmdet.core import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)
from onnxsim import simplify
 
 
 
if __name__ == '__main__':
 
    config_path = "configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py"
    checkpoint_path = "checkpoints/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth"
    output_file = 'fcos_ori.onnx'
 
 
 
    orig_model = build_model_from_cfg(config_path, checkpoint_path)
 
    normalize_cfg = {'mean': [0,0,0], 'std': [1,1,1]}
    input_config = {
        'input_shape': (1,3,256,256),
        'input_path': 'tests/data/color.jpg',
        'normalize_cfg': normalize_cfg
    }
    model, tensor_data = generate_inputs_and_wrap_model(config_path, checkpoint_path, input_config)
 
    # dynamic_ax = {'images': {0:"batch_size", 2: "image_height", 3: "image_width"},
    #               "fm1": {0:"batch_size", 2: "fm1_height", 3: "fm1_width"},
    #               "fm2": {0:"batch_size", 2: "fm2_height", 3: "fm2_width"},
    #               "fm3": {0:"batch_size", 2: "fm3_height", 3: "fm3_width"},
    #               "fm4": {0:"batch_size", 2: "fm4_height", 3: "fm4_width"},
    #               "fm5": {0:"batch_size", 2: "fm5_height", 3: "fm5_width"}}
    dynamic_ax = {'input':[0,2,3],"fm1":[0,2,3],"fm2":[0,2,3],"fm3":[0,2,3],"fm4":[0,2,3],"fm5":[0,2,3]}
    input_names = ["input"]
    output_names = ["fm1","fm2","fm3","fm4","fm5"]
    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=11,
        dynamic_axes=dynamic_ax)
    print("convert to onnx success!")
 
    # model_simp, ok = simplify(onnx.load(output_file))
    # assert ok,"simp failed!"
    # onnx.save(model_simp,"fcos_simp.onnx")
