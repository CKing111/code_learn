import numpy as np

import onnx
from onnx import defs, checker, helper, numpy_helper, mapping
from onnx import ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorProto, OperatorSetIdProto
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node

dynamic_batch = False

def append_nms(graph, unused_node=[]):
    ngraph = GraphProto()
    ngraph.name = graph.name

    ngraph.input.extend([i for i in graph.input if i.name not in unused_node])
    ngraph.initializer.extend([i for i in graph.initializer if i.name not in unused_node])
    ngraph.value_info.extend([i for i in graph.value_info if i.name not in unused_node])
    ngraph.node.extend([i for i in graph.node if i.name not in unused_node])

    output_info = [i for i in graph.output]
    ngraph.value_info.extend(output_info)
    # print(graph.output)

    score_node = 'scores'
    bbox_node = 'boxes'

    # ngraph.value_info.append(make_tensor_value_info(score_node, TensorProto.FLOAT, [1, 2134, 3]))
    # ngraph.value_info.append(make_tensor_value_info(bbox_node, TensorProto.FLOAT, [1, 2134, 1, 4]))

    nms = make_node(
        'TRT_NonMaxSuppression',
        [bbox_node, score_node],
        ['num_detections', 'nmsed_boxes', 'nmsed_scores', 'nmsed_classes'],
        'batch_nms',
    )
    nms.attribute.append(make_attribute('backgroundLabelId', -1))
    nms.attribute.append(make_attribute('iouThreshold', 0.5))
    nms.attribute.append(make_attribute('isNormalized', False))
    nms.attribute.append(make_attribute('keepTopK', 200))
    nms.attribute.append(make_attribute('numClasses', 2)) #
    nms.attribute.append(make_attribute('shareLocation', True))
    nms.attribute.append(make_attribute('scoreThreshold', 0.1))
    nms.attribute.append(make_attribute('topK', 1000))
    ngraph.node.append(nms)

    if dynamic_batch:
        num_detection = make_tensor_value_info('num_detections', TensorProto.INT32, ["-1", 1])
        nmsed_box = make_tensor_value_info('nmsed_boxes', TensorProto.FLOAT, ["-1", 200, 4])
        nmsed_score = make_tensor_value_info('nmsed_scores', TensorProto.FLOAT, ["-1", 200, 1])
        nmsed_class = make_tensor_value_info('nmsed_classes', TensorProto.FLOAT, ["-1", 200, 1])
    else:
        num_detection = make_tensor_value_info('num_detections', TensorProto.INT32, [1, 1])
        nmsed_box = make_tensor_value_info('nmsed_boxes', TensorProto.FLOAT, [1, 200, 4])
        nmsed_score = make_tensor_value_info('nmsed_scores', TensorProto.FLOAT, [1, 200, 1])
        nmsed_class = make_tensor_value_info('nmsed_classes', TensorProto.FLOAT, [1, 200, 1])

    ngraph.output.extend([num_detection, nmsed_box, nmsed_score, nmsed_class])

    print(ngraph.output)
    print(ngraph)

    return ngraph

if __name__ == '__main__':
    model = onnx.load('./yolov5s_416x320_no_nms.onnx')

    model_attrs = dict(
        ir_version = model.ir_version,
        opset_imports = model.opset_import,
        producer_version = model.producer_version,
        model_version = model.model_version
    )

    model = make_model(append_nms(model.graph), **model_attrs)
    print(model.graph)
    checker.check_model(model)
    onnx.save(model, 'yolov5s_416x320_0.01.onnx')


