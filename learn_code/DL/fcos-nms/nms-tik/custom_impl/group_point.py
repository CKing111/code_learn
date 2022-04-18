from ascend import AContainer1951
from uti import interface_check
from version.get_version import get_aicore_container
from . import pointsampling_base


class GroupPoint:

    def __init__(self, container: AContainer1951, input_data, idx_data, output,
                 kernel_n):
        self.tik_inst = container.tinst
        input_shape = input_data.get("shape")
        interface_check.check_kernelname(kernel_n)
        interface_check.check_param(input_data, [4], ["float16"], ["NCHW"])
        interface_check.check_param(idx_data, [4], ["int32"], ["NCHW"])
        interface_check.check_param(output, [4], ["float16"], ["NCHW"])

        self.kernel_name = kernel_n
        self.src_b = input_shape[0]
        self.src_c = input_shape[1]
        self.src_k = input_shape[2]
        self.src_p = input_shape[3]
        self.src_n = input_shape[2] * input_shape[3]

        self.idx_b, self.idx_k, self.idx_h, self.idx_w = idx_data.get('shape')
        self.dst_b, self.dst_c, self.dst_h, self.dst_w = output.get('shape')

        self.point_num = self.idx_h * self.idx_w
        self.src_c16 = ((self.src_c + 15) // 16) * 16
        self._para_check()
        self.point_sample = pointsampling_base.PointSample(container,
                                                           self.src_b,
                                                           self.src_c,
                                                           self.src_n,
                                                           self.point_num,
                                                           kernel_n)

    def _para_check(self):
        if self.src_b != self.idx_b:
            raise RuntimeError("The batch of src and idx should be same.")
        if self.idx_w > self.src_n:
            raise RuntimeError(
                "The number of sampling points should less than input points.")
        if self.src_b * self.src_c * self.src_n < 2 or self.idx_b * \
                self.idx_h * self.idx_k * self.idx_w < 2:
            raise RuntimeError(
                "The number of sampling points should more than one.")
        if self.idx_k != 1:
            raise RuntimeError(" The format of idx data should be N1HW.")
        if self.src_p != 1 and self.src_k != 1:
            raise RuntimeError(
                "The format of input data should be NCD1 or NC1D.")
        if self.dst_b != self.src_b or self.dst_c != self.src_c or \
                self.dst_h != self.idx_h or self.dst_w != self.idx_w:
            raise RuntimeError(
                "The format of output data should be [N, input_data'C, "
                "idx_data'H, idx_data'W].")

    def model_compute(self):
        self.point_sample.model_compute()

    def tik_output_debug(self):
        return self.tik_inst


def group_point(input_data, idx_data, output, kernel_n="GroupPoint",
                test=False):
    '''
    Puts update data into the output shape based on the indice.
    Args:
        param input_data: the input_data
        param idx_data: the idx_data(index)
        param output: the shape and dtype of output shape
        param kernel_n: the name of kernel
    '''
    container = get_aicore_container(("Ascend610",), c3x_support_list=())

    obj = GroupPoint(container, input_data, idx_data, output, kernel_n)
    obj.model_compute()
    if test:
        return obj.tik_output_debug()
    else:
        return 0
