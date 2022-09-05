import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, Matpath, Battery_list,split, cols):
        # dataframe = pd.read_csv(filename)  # csv
        # dataframe = self.loadMat(matfile)  # mat
        self.capacity, self.charge, self.discharge = self.get_Battery(self, Matpath, Battery_list)  # 元素容器
        # self.capacity[name] = self.getBatteryCapacity(dataframe)  # 放电时的容量数据
        # self.charge[name] = self.getBatteryValues(dataframe, 'charge')  # 充电数据
        # self.discharge[name] = self.getBatteryValues(dataframe, 'discharge')  # 放电数据

        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test = dataframe.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_Battery(self, Matpath, Battery_list):
        for name in Battery_list:
            print('Load Dataset ' + name + '.mat ...')
            path = Matpath + name + '.mat'
            data = self.loadMat(path)
            # print(len(data))
            #     Battery[name] = getBatteryCapacity(data)   # 放电时的容量数据
            self.capacity[name] = self.getBatteryCapacity(data)  # 放电时的容量数据
            self.charge[name] = self.getBatteryValues(data, 'charge')  # 充电数据
            self.discharge[name] = self.getBatteryValues(data, 'discharge')  # 放电数据
        return self.capacity, self.charge, self.discharge


    # 转换时间格式，将字符串转换成 datatime 格式
    def convert_to_time(self, hmm):
        year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(
            hmm[5])
        return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

    # 加载 mat 文件
    def loadMat(self, matfile):
        data = loadmat(matfile)  # 加载mat文件
        filename = matfile.split("/")[-1].split(".")[0]  # 以‘.’和‘/’为分隔符分割，分割出路径中的文件名
        # nasa_count_data/B0006.mat  ---》  B0006
        col = data[filename]  # dict_keys(['__header__', '__version__', '__globals__', 'B0006'])
        col = col[0][0][0][0]  # numpy   (1, 1 )  --->  (616,)
        size = col.shape[0]

        data = []
        for i in range(size):
            #         print(col.shape)
            k = list(col[i][3][0].dtype.fields.keys())
            # k:dict_keys(['Voltage_measured', 'Current_measured', 'Temperature_measured',
            # 'Current_charge', 'Voltage_charge', 'Time'])
            # “电压测量”、“电流测量”、“温度测量”、“电流充电”、“电压充电”、“时间”
            # col[i][0][0]  : 表示当前样本i的作业类别，放电、充电、阻抗
            # col[i][1][0]  : 表示当前样本i的作业温度
            # col[i][2][0]  : 表示当前样本i的作业日期，时间，'2008-04-02 13:08:17'
            # col[i][3][0]  : 表示当前样本i的作业六个参数值

            d1, d2 = {}, {}
            if str(col[i][0][0]) != 'impedance':  # 判断作业类型,非阻抗作业
                for j in range(len(k)):  # 遍历六个特征列
                    t = col[i][3][0][0][j][0];
                    l = [t[m] for m in range(len(t))]
                    d2[k[j]] = l  # 将读取到的数据放入对应的字典keys下
            d1['type'], d1['temp'] = str(col[i][0][0]), int(col[i][1][0])
            d1['time'], d1['data'] = str(self.convert_to_time(col[i][2][0])), d2
            data.append(d1)
        return data

    # 提取锂电池容量
    def getBatteryCapacity(self, Battery):
        # 电池容量在放电数据集 'discharge'的'Capacity'特征下；
        cycle, capacity = [], []
        i = 1
        for Bat in Battery:
            if Bat['type'] == 'discharge':
                capacity.append(Bat['data']['Capacity'][0])
                cycle.append(i)
                i += 1
        return [cycle, capacity]  # 构成了(x, y)坐标系

    # 获取锂电池充电或放电时的测试数据
    def getBatteryValues(self, Battery, Type='charge'):
        data = []
        for Bat in Battery:
            if Bat['type'] == Type:
                data.append(Bat['data'])
        return data

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y
