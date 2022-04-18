from util import OpLog as oplog


class Blob(object):
    '''
        NOTE: support same func with caffe blob
    '''

    def __init__(self):
        pass

    '''
        return shape count(n) size
        shape should be as list
    '''

    @staticmethod
    def count(shape, num):
        oplog.check_lt(num, len(shape),
                       "blob count input n should be less than n")
        oplog.check_ge(num, 0, "blob count input n should be great equal n")
        count = 1
        for i in range(num, len(shape)):
            count = count * shape[i]
        return count

    @staticmethod
    def channels(shape):
        oplog.check_eq(len(shape), 4, "blob size should be 4")
        return shape[1]

    @staticmethod
    def batch_size(shape):
        oplog.check_eq(len(shape), 4, "blob size should be 4")
        return shape[0]

    @staticmethod
    def height(shape):
        oplog.check_eq(len(shape), 4, "blob size should be 4")
        return shape[2]

    @staticmethod
    def width(shape):
        oplog.check_eq(len(shape), 4, "blob size should be 4")
        return shape[3]

