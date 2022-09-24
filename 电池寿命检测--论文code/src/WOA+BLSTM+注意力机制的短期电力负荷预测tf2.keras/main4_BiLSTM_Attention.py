# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
from model import BiLSTM_Attention, result,split_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.random.set_seed(0)
np.random.seed(0)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
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
ss_X = StandardScaler().fit(train_data)
ss_Y = StandardScaler().fit(train_label)
# ss_X = MinMaxScaler(feature_range=(0, 1)).fit(train_data)
# ss_Y = MinMaxScaler(feature_range=(0, 1)).fit(train_label)
train_data = ss_X.transform(train_data).reshape(train_data.shape[0], time_steps, -1)
train_label = ss_Y.transform(train_label)

test_data = ss_X.transform(test_data).reshape(test_data.shape[0], time_steps, -1)
test_label = ss_Y.transform(test_label)
# In[]定义超参数
num_epochs = 20  # 迭代次数
batch_size = 16  # batchsize
lr = 0.01  # 学习率
hidden1 = 10  # bilstm节点数1
hidden2 = 10  # bilstm节点数2
fc = 10
sequence, feature = train_data.shape[-2:]
output_node = train_label.shape[1]

model = BiLSTM_Attention(sequence, feature,hidden1,hidden2, fc, output_node).build_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
model.summary()
train_again = True  # 为False的时候就直接加载训练好的模型进行测试
# 训练模型
if train_again:
    history = model.fit(train_data, train_label, epochs=num_epochs, validation_data=(test_data, test_label),
                        batch_size=batch_size, verbose=1)
    model.save('model/bilstm_attention_model.h5')
    # 画loss曲线
    plt.figure()
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='testing')
    plt.title('loss curve')
    plt.legend()
    plt.savefig('figure/bilstm_attention_loss_curve.jpg')
else:  # 加载模型
    model = tf.keras.models.load_model('model/bilstm_attention_model.h5')
test_pred = model.predict(test_data)

# 对测试结果进行反归一化
test_label1 = ss_Y.inverse_transform(test_label)
test_pred1 = ss_Y.inverse_transform(test_pred)

np.savez('result/bilstm_attention_result.npz', true=test_label1, pred=test_pred1)
# In[]计算各种指标
test_pred1=test_pred1.reshape(-1,1)
test_label1=test_label1.reshape(-1,1)

result(test_label1,test_pred1,'BiLSTM_attention')

# plot test_set result
plt.figure()
plt.plot(test_label1, c='r', label='real')
plt.plot(test_pred1, c='b', label='pred')
plt.legend()
plt.xlabel('样本点')
plt.ylabel('功率')
plt.savefig('figure/BiLSTM_attention预测结果.jpg')
plt.show()

