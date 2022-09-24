import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, LSTM,Bidirectional,Input
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def split_data(data, n):
    in_ = []
    out_ = []    
    N = data.shape[0] - n
    for i in range(N):
        in_.append(data[i:i + n,:])
        out_.append(data[i + n,:96])
    in_ = np.array(in_).reshape(len(in_), -1)
    out_ = np.array(out_).reshape(len(out_), -1)
    return in_, out_
def result(real,pred,name):
    ss_X = MinMaxScaler(feature_range=(-1, 1))
    real = ss_X.fit_transform(real).reshape(-1,)
    pred = ss_X.transform(pred).reshape(-1,)
    
    # mape
    test_mape = np.mean(np.abs((pred - real) / real))
    # rmse
    test_rmse = np.sqrt(np.mean(np.square(pred - real)))
    # mae
    test_mae = np.mean(np.abs(pred - real))
    # R2
    test_r2 = r2_score(real, pred)

    #print(name,':的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2)
    print(name,'的mape:%.4f,rmse:%.4f,mae：%.4f,R2:%.4f'%(test_mape ,test_rmse, test_mae, test_r2))

class AttentionLayer(Layer):
    # BahdanauAttention https://blog.csdn.net/u010960155/article/details/82853632
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):  # 输入：inputs.shape = (batch_size, time_steps, seq_len)
        # Permute将2、1轴翻转后，(batch_size, time_steps, seq_len) -> (batch_size, lstm_units, seq_len)
        # 转换后：x.shape = (batch_size, seq_len, time_steps)
        x = K.permute_dimensions(inputs, (0, 2, 1))

        # 经过一个全连接层和Softmax后，其维度仍为(batch_size, seq_len, time_steps)
        # 其实际内涵为，利用全连接层计算每一个time_steps的权重
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))

        # a * x后获得每一个step中，每个维度在所有step中的权重
        # 再经过Permute将2、1轴翻转后，(batch_size, seq_len, time_steps) -> (batch_size, time_steps, seq_len)
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
class LSTM_(object):
    '''
    seq：时间步维度
    feat_dim：变量维数
    hidden_unit1：BiLSTM节点数
    hidden_unit2：LSTM节点数
    fc:全连接神经元个数
    output_dim：输出维度
    '''
    def __init__(self, seq,feat_dim,hidden_unit1,hidden_unit2,fc,output_dim):
        self.input_dim = seq
        self.feat_dim = feat_dim
        self.units1 = hidden_unit1
        self.units2 = hidden_unit2
        self.fc=fc
        self.output_dim = output_dim
    def build_model(self):
        inp = Input(shape=(self.input_dim, self.feat_dim))        
        # LSTM单元
        hout = LSTM(self.units1, return_sequences=True)(inp)
        hout = LSTM(self.units2, return_sequences=False)(hout)
        # 全连接层和输出层
        dense = Dense(self.fc, activation='relu')(hout)
        out = Dense(self.output_dim, activation=None)(dense)
        model = Model(inputs=inp, outputs=out)
        return model


class BiLSTM(object):
    '''
    seq：时间步维度
    feat_dim：变量维数
    hidden_unit1：biLSTM节点数1
    hidden_unit2：biLSTM节点数2
    fc:全连接神经元个数
    output_dim：输出维度
    '''
    def __init__(self, seq,feat_dim,hidden_unit1,hidden_unit2,fc,output_dim):
        self.input_dim = seq
        self.feat_dim = feat_dim
        self.units1 = hidden_unit1
        self.units2 = hidden_unit2
        self.fc=fc
        self.output_dim = output_dim
    def build_model(self):
        inp = Input(shape=(self.input_dim, self.feat_dim))
        # BiLSTM单元1
        hout = Bidirectional(LSTM(self.units1, return_sequences=True))(inp)
        hout = Bidirectional(LSTM(self.units2, return_sequences=False))(hout)
        # 全连接层和输出层
        dense = Dense(self.fc, activation='relu')(hout)
        out = Dense(self.output_dim, activation=None)(dense)
        model = Model(inputs=inp, outputs=out)
        return model


class BiLSTM_Attention(object):
    '''
    seq：时间步维度
    feat_dim：变量维数
    hidden_unit1：biLSTM节点数1
    hidden_unit2：biLSTM节点数2
    fc:全连接神经元个数
    output_dim：输出维度
    '''
    def __init__(self, seq,feat_dim,hidden_unit1,hidden_unit2,fc,output_dim):
        self.input_dim = seq
        self.feat_dim = feat_dim
        self.units1 = hidden_unit1
        self.units2 = hidden_unit2
        self.fc=fc
        self.output_dim = output_dim
    def build_model(self):
        inp = Input(shape=(self.input_dim, self.feat_dim))
        # BiLSTM单元        
        in_ = Bidirectional(LSTM(self.units1, return_sequences=True))(inp)
        in_ = Bidirectional(LSTM(self.units2, return_sequences=True))(in_)
        # 注意力机制
        attention = AttentionLayer()(in_)
        # 全连接层和输出层
        dense = Dense(self.fc, activation='relu')(attention)
        out = Dense(self.output_dim, activation=None)(dense)
        model = Model(inputs=inp, outputs=out)
        return model
