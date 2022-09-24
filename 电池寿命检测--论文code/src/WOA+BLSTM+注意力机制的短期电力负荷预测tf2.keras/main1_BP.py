# -*- coding: utf-8 -*-
# BP回归,bp就是多层感知器
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import savemat
from model import split_data,result
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
np.random.seed(0)
# In[] 加载数据
data=pd.read_csv('电荷数据.csv').iloc[:,1:].fillna(0).values
time_steps=10
in_,out_=split_data(data,time_steps)

n=range(in_.shape[0])
m=-2#最后两天测试
train_data = in_[n[0:m],]
test_data = in_[n[m:],]
train_label = out_[n[0:m],]
test_label = out_[n[m:],]
# 归一化
ss_X=StandardScaler().fit(train_data)
ss_Y=StandardScaler().fit(train_label)
# ss_X=MinMaxScaler(feature_range=(0,1)).fit(train_data)
# ss_Y=MinMaxScaler(feature_range=(0,1)).fit(train_label)
train_data = ss_X.transform(train_data)
train_label = ss_Y.transform(train_label)

test_data = ss_X.transform(test_data)
test_label = ss_Y.transform(test_label)

in_num=train_data.shape[1]
out_num=train_label.shape[1]

clf = MLPRegressor(max_iter=10,hidden_layer_sizes=(100),random_state=0)
clf.fit(train_data,train_label)
test_pred=clf.predict(test_data)
# In[] 画出测试集的值
# 对测试结果进行反归一化
test_label  = ss_Y.inverse_transform(test_label)
test_pred   = ss_Y.inverse_transform(test_pred)
np.savez('result/bp_result.npz', true=test_label, pred=test_pred)
# In[]计算各种指标
test_pred1=test_pred.reshape(-1,1)
test_label1=test_label.reshape(-1,1)

result(test_label1,test_pred1,'BP')


# plot test_set result
plt.figure()
plt.plot(test_label1, c='r', label='real')
plt.plot(test_pred1, c='b', label='pred')
plt.legend()
plt.xlabel('样本点')
plt.ylabel('功率')
plt.savefig('figure/BP预测结果.jpg')
plt.show()

