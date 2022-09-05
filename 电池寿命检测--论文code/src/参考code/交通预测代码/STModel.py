import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split

# tensorflow and keras
# from keras.utils.vis_utils import plot_model

import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from spektral.layers import GraphConv

df = pd.read_csv('data/training.txt', delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
lagging = 5
lagging_feature = ['lagging%01d' % e for e in range(lagging, 0, -1)]
base_feature = [x for x in df.columns.values.tolist() if x not in ['time_interval_begin', 'link_ID', 'link_ID_int',
                                                                   'date', 'travel_time', 'imputation1',
                                                                   'minute_series', 'area', 'hour_en', 'day_of_week']]
base_feature = [x for x in base_feature if x not in lagging_feature]
train_feature = list(base_feature)
train_feature.extend(lagging_feature)
valid_feature = list(base_feature)
valid_feature.extend(['minute_series', 'travel_time'])

print("train_feature:")
print(train_feature)

df = df.dropna()

df = df.sort_values(by=['time_interval_begin', 'link_ID_en'], ascending=True)



# df_train = df.loc[df['time_interval_begin'] < ('2017-06-01')]
df_train = df.loc[((df['time_interval_begin'].dt.year == 2017) & (df['time_interval_begin'].dt.month < 6))].copy()

df_train_X = df_train[train_feature].values

df_train_X_normal = np.concatenate((df_train_X[:, 0].reshape(-1, 1), df_train_X[:, 2].reshape(-1, 1),
                                    df_train_X[:, -6].reshape(-1, 1), df_train_X[:, -5].reshape(-1, 1),
                                    df_train_X[:, -4].reshape(-1, 1), df_train_X[:, -3].reshape(-1, 1),
                                    df_train_X[:, -2].reshape(-1, 1), df_train_X[:, -1].reshape(-1, 1)), axis=1)


df_train_X_other = np.concatenate((df_train_X[:, 1].reshape(-1, 1), df_train_X[:, 3:-6]), axis=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

model_scaler = StandardScaler()
model_scaler.fit(df_train_X_normal)
df_train_X_normal_scale = model_scaler.transform(df_train_X_normal)

df_train_X_final = np.concatenate((df_train_X_normal_scale, df_train_X_other), axis=1)

df_train_X = df_train_X_final

df_train_X_nodes = []
i = 0
while i < len(df_train_X_final):
    df_train_X_nodes.append(df_train_X[i:i + 132, :])
    i += 132

df_train_X_nodes = np.array(df_train_X_nodes)

df_train_X_timeseries_nodes = df_train_X_nodes.reshape(len(df_train_X_nodes), 132 * 33)

df_train_X_10_nodes = []
i = 0
while i + 10 <= len(df_train_X_timeseries_nodes):
    df_train_X_10_nodes.append(df_train_X_timeseries_nodes[i:i + 10])
    i += 1

df_train_X_10_nodes = np.array(df_train_X_10_nodes)

df_train_Y = df_train['travel_time'].values

df_train_Y_nodes = []
i = 0
while i < len(df_train_Y):
    df_train_Y_nodes.append(df_train_Y[i:i + 132])
    i += 132

df_train_Y_nodes = np.array(df_train_Y_nodes)

df_train_Y_10_nodes = []
i = 0
while i + 10 <= len(df_train_Y_nodes):
    df_train_Y_10_nodes.append(df_train_Y_nodes[i + 9:i + 10])
    i += 1

df_train_Y_10_nodes = np.array(df_train_Y_10_nodes)

ad_matrix = np.load('ad_mat.npy')

ad_mat = ad_matrix.copy()

D = np.zeros(shape=(132, 132))
for i in range(132):
    for j in range(132):
        if ad_mat[i, j] == 1:
            D[i, i] += 1
laplace = D - ad_mat

D_ = np.zeros(shape=(132, 132))
for i in range(132):
    if (D[i, i] > 0):
        D_[i, i] = D[i, i] ** (-1 / 2)

laplace_ = np.matmul(np.matmul(D_, laplace), D_)


from spektral.utils.convolution import *

# spektral.utils.convolution.localpooling_filter

la_gcn_filter = localpooling_filter(ad_mat, symmetric=True)

laplace_matrix = np.tile(laplace_, (23660, 1, 1))

np.isnan(laplace_matrix).any()

df_train_ad_input = []
i = 0
while i + 10 <= len(df_train_X_nodes):
    df_train_ad_input.append(df_train_X_nodes[i + 9:i + 10, :, :].reshape(132, 33))
    i += 1

df_train_ad_input = np.array(df_train_ad_input)

from spektral.layers import GraphConv, GraphAttention

scored_seq_length = 132

# loss functions
def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred):
    score = 0
    for i in range(y_actual.shape[2]):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / y_actual.shape[2]
    return score


def build_model(input_seq_len=132, output_seq_len=scored_seq_length):
    # inputs
    Feature_inputs = L.Input(shape=(10, 4356), name="Nodes_features")
    Feature_inputs_new = L.Dense(660)(Feature_inputs)

    ad_feature_input = L.Input(shape=(132, 33), name="ad_feature_input")
    # adjacency matrix about seq. connectivity
    ad_feature_input_new = L.Dense(64)(ad_feature_input)

    adj_matrix_inputs = L.Input((132, 132), name="adjmatrix")

    attn_adj_matrix_inputs = L.Input((132, 132), name="attn_adj_matrix")
    gcn_1 = GraphConv(
        16,
        activation='tanh',
    )([ad_feature_input_new, adj_matrix_inputs])

    gcn_2 = GraphConv(
        32,
        activation='tanh',
    )([gcn_1, adj_matrix_inputs])

    #     gcn_2 = SeqSelfAttention(attention_activation='relu', attention_type='multiplicative',return_attention=True)(gcn_2)

    #     gcn_2 = GraphAttention(16)([gcn_2, attn_adj_matrix_inputs])
    conv_1 = L.Conv1D(
        16, 16,
        padding='same'
    )(Feature_inputs_new)

    conv_1 = L.LeakyReLU()(conv_1)

    conv_2 = L.Conv1D(
        32, 1,
        padding='same'
    )(conv_1)

    conv_2 = L.LeakyReLU()(conv_2)

    conv_3 = L.Conv1D(
        32, 2,
        padding='same'
    )(conv_1)

    conv_3 = L.LeakyReLU()(conv_3)

    conv_4 = L.Conv1D(
        32, 4,
        padding='same'
    )(conv_1)

    conv_4 = L.LeakyReLU()(conv_4)

    conv_out = L.Concatenate(axis=2)([conv_2, conv_3, conv_4])

    lstm_out = L.Bidirectional(L.LSTM(16, activation='elu', return_sequences=True))(conv_out)
    #     lstm_out, spa_att = SeqSelfAttention(attention_activation='relu', attention_type='multiplicative',
    #                                          return_attention=True)(lstm_out)

    tm_out = L.Flatten()(lstm_out)
    tm_out = L.Dense(132, activation='relu')(tm_out)

    #     linear = L.Dense(32)(ad_feature_input)
    linear = L.Flatten()(ad_feature_input)
    linear = L.Dense(132)(linear)
    tm_out = L.add([tm_out, linear])

    #     gcn_2 = L.Dense(32,activation='relu')(gcn_2)
    gcn_2 = L.Flatten()(gcn_2)

    gcn_2 = L.Dense(132)(gcn_2)

    out = L.Concatenate(axis=1)([tm_out, gcn_2])

    out = L.Dense(132)(out)

    out = L.Reshape((132, 1))(out)

    model = tf.keras.Model(
        inputs=[
            Feature_inputs,
            ad_feature_input,
            adj_matrix_inputs,
            attn_adj_matrix_inputs
        ],
        outputs=out,
    )

    return model


model = build_model()
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint


def scheduler(epoch):
    # 每隔15个epoch，学习率减小为原来的1/10
    if epoch % 15 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


# early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
lr_new = LearningRateScheduler(scheduler)

history = model.fit(
    [df_train_X_10_nodes, df_train_ad_input, laplace_matrix[:23651]], df_train_Y_10_nodes.reshape(23651, 132, 1),
    batch_size=32,
    epochs=60,
    callbacks=[lr_new],
    verbose=1,
    validation_split=0.1)
print(f"Min validation loss history={min(history.history['val_loss'])}")

model.save_weights('model/gcn_4_24.h5')

model.summary()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch', fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.title('train_loss', fontsize=16)
plt.show()
fig.savefig('train_loss.svg', format='svg')

df_test = df.loc[((df['time_interval_begin'].dt.year == 2017) & (df['time_interval_begin'].dt.month == 6)
                  & (df['time_interval_begin'].dt.hour.isin([7, 8, 13, 14, 17, 18])))].copy()

df_test_X = df_test[train_feature].values

df_test_X_normal = np.concatenate((df_test_X[:, 0].reshape(-1, 1), df_test_X[:, 2].reshape(-1, 1),
                                   df_test_X[:, -6].reshape(-1, 1), df_test_X[:, -5].reshape(-1, 1),
                                   df_test_X[:, -4].reshape(-1, 1), df_test_X[:, -3].reshape(-1, 1),
                                   df_test_X[:, -2].reshape(-1, 1), df_test_X[:, -1].reshape(-1, 1)), axis=1)

df_test_X_other = np.concatenate((df_test_X[:, 1].reshape(-1, 1), df_test_X[:, 3:-6]), axis=1)

df_test_X_normal_scale = model_scaler.transform(df_test_X_normal)

df_test_X_final = np.concatenate((df_test_X_normal_scale, df_test_X_other), axis=1)
df_test_X = df_test_X_final

df_test_X_nodes = []
i = 0
while i < 693000:
    df_test_X_nodes.append(df_test_X[i:i + 132, :])
    i += 132

df_test_X_nodes = np.array(df_test_X_nodes)


df_test_X_timeseries_nodes = df_test_X_nodes.reshape(5250, 132 * 33)

df_test_X_10_nodes = []
i = 0
while i + 10 <= 5250:
    df_test_X_10_nodes.append(df_test_X_timeseries_nodes[i:i + 10])
    i += 1

df_test_X_10_nodes = np.array(df_test_X_10_nodes)

df_test_X_10_nodes.shape

df_test_Y = df_test['travel_time'].values

df_test_Y.shape

df_test_Y_nodes = []
i = 0
while i < 693000:
    df_test_Y_nodes.append(df_test_Y[i:i + 132])
    i += 132

df_test_Y_nodes = np.array(df_test_Y_nodes)

df_test_Y_nodes.shape

df_test_Y_10_nodes = []
i = 0
while i + 10 <= 5250:
    df_test_Y_10_nodes.append(df_test_Y_nodes[i + 9:i + 10])
    i += 1

df_test_Y_10_nodes = np.array(df_test_Y_10_nodes)

df_test_Y_10_nodes.shape

df_test_ad_input = []
i = 0
while i + 10 <= 5250:
    df_test_ad_input.append(df_test_X_nodes[i + 9:i + 10, :, :].reshape(132, 33))
    i += 1

df_test_ad_input = np.array(df_test_ad_input)

df_test_ad_input.shape

test_predict = model.predict([df_test_X_10_nodes, df_test_ad_input, laplace_matrix[:5241]])

test_predict.shape

test_predict = test_predict.reshape(-1, 132)
df_test_Y_10_nodes = df_test_Y_10_nodes.reshape(-1, 132)

from sklearn import metrics


def GetMAPE(y_hat, y_test):
    sum = np.mean(np.abs((y_hat - y_test) / y_test)) * 100
    return sum


def GetMAE(y_hat, y_test):
    sum = metrics.mean_absolute_error(y_test, y_hat)
    return sum


def GetRMSE(y_hat, y_test):
    sum = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
    return sum


test_mape = GetMAPE(np.expm1(test_predict), np.expm1(df_test_Y_10_nodes))
# test_mae = GetMAE(test_Y_predict,test_Y_final_split)
test_mae = GetMAE(np.expm1(test_predict), np.expm1(df_test_Y_10_nodes))
test_rmse = GetRMSE(np.expm1(test_predict), np.expm1(df_test_Y_10_nodes))

print('test_mape:',test_mape)
print('test_mae:',test_mae)
print('test_rmse:',test_rmse)


