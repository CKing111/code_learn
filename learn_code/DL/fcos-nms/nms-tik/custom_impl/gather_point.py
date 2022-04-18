from ascend import AContainer1951
from uti import interface_check
from version.get_version import get_aicore_container
from . import pointsampling_base


class GatherPoint:

    def __init__(self, container: AContainer1951, input_data, point_data,
                 output, kernel_n):
        self.tik_inst = container.tinst
        interface_check.check_kernelname(kernel_n)
        interface_check.check_param(input_data, [4], ["float16"], ["NCHW"])
        interface_check.check_param(point_data, [4], ["int32"], ["NCHW"])
        interface_check.check_param(output, [4], ["float16"], ["NCHW"])
        input_shape = input_data.get('shape')
        point_shape = point_data.get('shape')
        output_shape = output.get('shape')
        self.kernel_name = kernel_n
        self.src_b = input_shape[0]
        self.src_c = input_shape[1]
        self.src_k = input_shape[2]
        self.src_p = input_shape[3]
        self.src_n = input_shape[3] * self.src_k
        self.point_b, self.point_num, self.point_h, self.point_w = point_shape
        self.dst_b, self.dst_c, self.dst_h, self.dst_w = output_shape

        self._para_check()
        self.point_sample = pointsampling_base.PointSample(container,
                                                           self.src_b,
                                                           self.src_c,
                                                           self.src_n,
                                                           self.point_num,
                                                           kernel_n)

    def _para_check(self):
        if self.src_n < self.point_num:
            raise RuntimeError(
                "The number of sampling points should less than input points.")
        if self.point_b != self.src_b:
            raise RuntimeError("The batch of idx should be same as src_data.")
        if self.src_c != 3:
            raise RuntimeError("Please check the channel of input data.")
        if self.point_h != 1 or self.point_w != 1:
            raise RuntimeError(
                "Please check the channel of point data which should be ND11.")
        if self.src_p != 1 and self.src_k != 1:
            raise RuntimeError(
                "The format of input data should be N3D1 or N31D.")
        if self.dst_c != self.src_c or self.dst_h != self.point_num or \
                self.dst_w != 1 or self.dst_b != self.src_b:
            raise RuntimeError(
                "The dst shape should be [N, 3, point_data'D, 1], please "
                "check.")

    def compute(self):
        self.point_sample.model_compute()

    def tik_output_debug(self):
        return self.tik_inst


def gather_point(input_data, point_data, output, kernel_n="GatherPoint",
                 test=False):
    '''
    Sampling data according to point's idxes which have sampled by
    FarthestPointSample.
    Args:
        param input_data: the dict of inputdata
        param point_data: the dict of idx data(index)
        param output: the dict of output shape
        param kernel_n: the name of kernel
    '''
    container = get_aicore_container(("Ascend610",), c3x_support_list=())
    obj = GatherPoint(container, input_data, point_data, output, kernel_n)
    obj.compute()
    if test:
        return obj.tik_output_debug()
    else:
        return 0
