"""
Example implementation for using stacked bidirectionnal recurrent units
Displays how to use this structure with padding of sequences
author: Paul KLEIN <kleinplw@gmail.com>
"""

from keras.models import Model
from keras.layers import Input, Dense, Masking, LeakyReLU, TimeDistributed, concatenate, Layer, LSTM, Lambda
from keras import backend as K
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt


class Reverse(Layer):

    def __init__(self, axis, **kwargs):
        super(Reverse, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def build(self, input_shape):
        if len(input_shape) < self.axis:
            raise ValueError('Input Rank inferior to specified axis. '
                             'input_shape: ' + str(input_shape) + ' - '
                             'axis : ' + str(self.axis))

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return K.reverse(x, axes=self.axis)


def get_model(max_len, n_features, hidden_size, dense_sizes, pad_value):
    _inputs = Input((max_len, n_features))
    m = Masking(mask_value=pad_value)(_inputs)
    inp_fw = LSTM(hidden_size, return_sequences=True, name='fw1', go_backwards=False)(m)
    inp_fw = LSTM(hidden_size, return_sequences=True, name='fw2', go_backwards=False)(inp_fw)
    inp_bw = LSTM(hidden_size, return_sequences=True, name='bw1', go_backwards=True)(m)
    inp_bw = LSTM(hidden_size, return_sequences=True, name='bw2', go_backwards=False)(inp_bw)
    # Second unit in the backwards stack has to go forward to process the sequence in the correct order
    # Backwards returns backwards sequence --> needs fliping
    # Using a Lambda layer to do so would currently discard the mask for all the downstream layers
    # inp_bw = Lambda(lambda x: K.reverse(x, axes=1), output_shape=lambda s: s)(inp_bw)
    inp_bw = Reverse(axis=1)(inp_bw)
    prev_inp = concatenate([inp_fw, inp_bw], axis=2)
    for idx, size in enumerate(dense_sizes):
        prev_inp = TimeDistributed(Dense(size, activation='linear'), name='fc' + str(idx))(prev_inp)
        prev_inp = LeakyReLU(alpha=.01)(prev_inp)
    _output = TimeDistributed(Dense(1, activation='linear'), name='output')(prev_inp)
    return Model(inputs=[_inputs], outputs=[_output])

if __name__ == '__main__':

    max_len = 30
    n_features = 2
    hidden_size = 5
    dense_sizes = [10, 5]
    pad_value = -3.

    model = get_model(max_len, n_features, hidden_size, dense_sizes, pad_value)
    model.summary()
    opt = optimizers.Adam(lr=0.01, clipnorm=1.)
    model.compile(loss='mse', optimizer=opt)

    # Crafting toy data
    n_ex = 100
    x = np.random.random((n_ex, max_len, n_features))
    y = np.ones((n_ex, max_len, 1))

    # Adding padding
    for i in range(n_ex):
        p_idx = np.random.randint(0, max_len)
        y[i, :, 0] = np.sum(x[i, :, :], axis=-1)
        x[i, p_idx:, :] = pad_value
        y[i, p_idx:, :] = pad_value

    # Train the network
    model.fit(x, y, batch_size=5, epochs=10, shuffle=True)
    # Display the predictions on the training set
    for i in range(10):
        example = x[i].reshape(1, max_len, n_features)
        label = y[i].reshape(max_len)
        prediction = model.predict(example).reshape(max_len)
        plt.plot(label, label='label')
        plt.plot(prediction, label='prediction')
        plt.legend()
        plt.show()
