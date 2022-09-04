import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tensorflow as tf
# from spektral.layers import GraphConv
from tensorflow.keras.models import Model  # Sequential, Model
from tensorflow.keras.layers import *  # Dense, Dropout, BatchNormalization, LSTM, Conv1D, Flatten, GRU, Input
from tensorflow.keras import optimizers, regularizers


def build_model(input_seq_len=132,
                output_seq_len=64,
                losss='mse',
                optimizerr='adam'
                ):
    # inputs
    Feature_inputs = L.Input(shape=(input_seq_len, 4356), name="Feature_inputs")
    Feature_inputs2 = L.Input(shape=(input_seq_len, 33), name="Feature_inputs2")

    Feature_inputs_new = L.Dense(660)(Feature_inputs)
    Feature_inputs_new2 = L.Dense(64)(Feature_inputs2)

    conv_1 = L.Conv1D(
        16, 1,
        padding='same', name="conv_1")(Feature_inputs_new)
    conv_1_1 = L.LeakyReLU()(conv_1)

    conv_2 = L.Conv1D(
        16, 3,
        padding='same', name="conv_2")(Feature_inputs_new)
    conv_2_2 = L.LeakyReLU()(conv_2)

    conv_3 = L.Conv1D(
        16, 4,
        padding='same', name="conv_3")(Feature_inputs_new)
    conv_3_3 = L.LeakyReLU()(conv_3)

    LSTM1 = L.Bidirectional(LSTM(16, input_shape=(input_seq_len, 660), return_sequences=True, name='LSTM1'))(
        Feature_inputs_new)

    merged_vector = concatenate([conv_1_1, conv_2_2, conv_3_3, LSTM1], axis=-1, name="merged_vector")
    LSTM2 = L.Bidirectional(LSTM(16, name='LSTM2'))(merged_vector)
    Flatten1 = Flatten(name="Flatten1")(LSTM2)
    Dense2 = Dense(16, activation='sigmoid', name="Dense2")(Flatten1)
    out1 = Dense(1, activation='sigmoid', name="out1")(Dense2)

    Flatten2 = Flatten()(Feature_inputs2)
    Dense2 = Dense(16)(Flatten2)
    out2 = Dense(1, name='out2')(Dense2)

    merged_vector2 = concatenate([out1, out2], axis=-1, name="merged_vector2")
    out3 = Dense(1, activation='sigmoid', name="out3")(merged_vector2)

    model = Model(
        inputs=[
            Feature_inputs, Feature_inputs2
        ],
        outputs=out3,
    )
    model.compile(loss=losss,
                  optimizer=optimizerr,
                  #                   optimizer=optimizers.Adam(lr=0.004,decay=0.04),
                  metrics=['accuracy'])
    model.summary()
    return model


model = build_model(input_seq_len=10,
                    output_seq_len=64,
                    losss='mse',
                    optimizerr='adam')
tf.keras.utils.plot_model(model, to_file='1.png', show_shapes=True, show_layer_names=True)