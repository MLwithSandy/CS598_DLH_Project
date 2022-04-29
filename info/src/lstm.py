from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

from util import Metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rolled', action='store_true')
    args = parser.parse_args()

    filename = 'lstm_rolled' if args.rolled else 'lstm'

    with np.load('data.npz') as data:
        x_train = data['x_train']
        x_val = data['x_val']
        if args.rolled:
            y_train = data['rol_y_train']
            y_val = data['rol_y_data']
        else:
            y_train = data['reg_y_train']
            y_val = data['reg_y_val']

    embedding_matrix = np.load('embedding_matrix.npy')

    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)

    model = Sequential()
    # model.add(Embedding(len(word_index) + 1, 300, input_length=x_train.shape[1]))
    model.add(Embedding(len(word_index) + 1, embedding_matrix.shape[1],
                        weights=[embedding_matrix], input_length=x_train.shape[1],
                        mask_zero=True, trainable=False))
    model.add(LSTM(100))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    metrics = Metrics('{}.log'.format(filename))
    early_stopping = EarlyStopping(min_delta=.0001, patience=2)
    model.fit(x_train, y_train,
              batch_size=32, epochs=5,
              validation_data=(x_val, y_val),
              callbacks=[metrics, early_stopping])
    model.save('{}.h5'.format(filename))


if __name__ == '__main__':
    main()
