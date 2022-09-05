import os
import math
import numpy as np
from numpy import newaxis
# from core.utils import Timer
import time
import datetime as dt

from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers, regularizers

import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tensorflow as tf


# from spektral.layers import GraphConv
# from tensorflow.keras.models import Model  # Sequential, Model
# from tensorflow.keras.layers import *  # Dense, Dropout, BatchNormalization, LSTM, Conv1D, Flatten, GRU, Input
# from tensorflow.keras import optimizers, regularizers


class Model_lstm:
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()
        self.layers = []

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        start_sampling = time.time()
        # 遍历每一个layer
        for layer in configs['model']['layers']:
            self.layers.append(layer['type'])
            neurons = layer['neurons'] if 'neurons' in layer else None  # 神经元个数
            dropout_rate = layer['rate'] if 'rate' in layer else None  # 丢失率
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None  # lstm-lstm
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None  # 输入时序步长
            input_dim = layer['input_dim'] if 'input_dim' in layer else None  # 输入特征维度

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        self.model.summary()
        print('[Model] Model Compiled')
        end_sampling = time.time()

    def plot_model(self, filename):
        file_topath = os.path.join(filename, '%s-二特征-10序列.png' % (len(self.layers['lstm']) + 'lstm'))
        tf.keras.utils.plot_model(self.model, to_file=file_topath, show_shapes=True, show_layer_names=True)


def train(self, x, y, epochs, batch_size, save_dir):
    start_sampling = time.time()

    print('[Model] Training Started')
    print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
    ]
    self.model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    self.model.save(save_fname)

    print('[Model] Training Completed. Model saved as %s' % save_fname)
    end_sampling = time.time()


def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
    start_sampling = time.time()

    print('[Model] Training Started')
    print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    callbacks = [
        ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
    ]
    self.model.fit_generator(
        data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        workers=1
    )

    print('[Model] Training Completed. Model saved as %s' % save_fname)
    end_sampling = time.time()


def predict_point_by_point(self, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = self.model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequences_multiple(self, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def predict_sequence_full(self, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
    return predicted
#
#
# def build_model(input_seq_len=132,
#                 output_seq_len=64,
#                 losss='mse',
#                 optimizerr='adam'
#                 ):
#     # inputs
#     Feature_inputs = L.Input(shape=(input_seq_len, 4356), name="Feature_inputs")
#     Feature_inputs2 = L.Input(shape=(input_seq_len, 33), name="Feature_inputs2")
#
#     Feature_inputs_new = L.Dense(660)(Feature_inputs)
#     Feature_inputs_new2 = L.Dense(64)(Feature_inputs2)
#
#     conv_1 = L.Conv1D(
#         16, 1,
#         padding='same', name="conv_1")(Feature_inputs_new)
#     conv_1_1 = L.LeakyReLU()(conv_1)
#
#     conv_2 = L.Conv1D(
#         16, 3,
#         padding='same', name="conv_2")(Feature_inputs_new)
#     conv_2_2 = L.LeakyReLU()(conv_2)
#
#     conv_3 = L.Conv1D(
#         16, 4,
#         padding='same', name="conv_3")(Feature_inputs_new)
#     conv_3_3 = L.LeakyReLU()(conv_3)
#
#     LSTM1 = L.Bidirectional(LSTM(16, input_shape=(input_seq_len, 660), return_sequences=True, name='LSTM1'))(
#         Feature_inputs_new)
#
#     merged_vector = concatenate([conv_1_1, conv_2_2, conv_3_3, LSTM1], axis=-1, name="merged_vector")
#     LSTM2 = L.Bidirectional(LSTM(16, name='LSTM2'))(merged_vector)
#     Flatten1 = Flatten(name="Flatten1")(LSTM2)
#     Dense2 = Dense(16, activation='sigmoid', name="Dense2")(Flatten1)
#     out1 = Dense(1, activation='sigmoid', name="out1")(Dense2)
#
#     Flatten2 = Flatten()(Feature_inputs2)
#     Dense2 = Dense(16)(Flatten2)
#     out2 = Dense(1, name='out2')(Dense2)
#
#     merged_vector2 = concatenate([out1, out2], axis=-1, name="merged_vector2")
#     out3 = Dense(1, activation='sigmoid', name="out3")(merged_vector2)
#
#     model = Model(
#         inputs=[
#             Feature_inputs, Feature_inputs2
#         ],
#         outputs=out3,
#     )
#     model.compile(loss=losss,
#                   optimizer=optimizerr,
#                   #                   optimizer=optimizers.Adam(lr=0.004,decay=0.04),
#                   metrics=['accuracy'])
#     model.summary()
#     return model
#
#
# model = build_model(input_seq_len=10,
#                     output_seq_len=64,
#                     losss='mse',
#                     optimizerr='adam')
# tf.keras.utils.plot_model(model, to_file='1.png', show_shapes=True, show_layer_names=True)
