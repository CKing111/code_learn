# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from model import result
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
data0=np.load('result/bp_result.npz')['true'].reshape(-1,1)
data1=np.load('result/bp_result.npz')['pred'].reshape(-1,1)
data2=np.load('result/lstm_result.npz')['pred'].reshape(-1,1)
data3=np.load('result/bilstm_result.npz')['pred'].reshape(-1,1)
data4=np.load('result/bilstm_attention_result.npz')['pred'].reshape(-1,1)
data5=np.load('result/woa_bilstm_attention_result.npz')['pred'].reshape(-1,1)


result(data0,data1,'BP')
result(data0,data2,'LSTM')
result(data0,data3,'BiLSTM')
result(data0,data4,'BiLSTM_At')
result(data0,data5,'WOA_BiLSTM_At')


plt.figure()
plt.plot(data0, c='r', label='real')
plt.plot(data1, c='b', label='BP')
plt.plot(data2, c='g', label='LSTM')
plt.plot(data3, c='y', label='BiLSTM')
plt.plot(data4, c='k', label='BiLSTM_At')
plt.plot(data5, c='m', label='WOA_BiLSTM_At')
plt.legend()
plt.xlabel('样本点')
plt.ylabel('功率')
plt.savefig('figure/结果对比.jpg')
plt.show()


















