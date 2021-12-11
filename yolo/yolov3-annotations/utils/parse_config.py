import numpy as np


# cfg解析函数：
#   将cfg的layer，setting等解析成dict的形式，返回一个包含这些dict的list；
#   lsit的每个元素（dict）对应cfg文件的一个 [] 开头的block（如net等），第一个元素就是该block的性质如{'type': 'net'...}
def parse_model_cfg(path):
    # Parses the yolo-v3 layer configuration file and returns module definitions
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            else:
                mdefs[-1][key] = val.strip()

    return mdefs

# 像mmdetection一样，将配置文件转码return成dict的键值对形式，便于索引查询
def parse_data_cfg(path):
    # Parses the data configuration file
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
