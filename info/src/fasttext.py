from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.python.keras.callbacks import EarlyStopping

from util import Metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rolled', action='store_true')
    args = parser.parse_args()

    filename = __file__[:-3] + '_rolled' if args.rolled else __file__[:-3]

    with np.load('data.npz') as data:
        x_train = data['x_train']
        x_val = data['x_val']
        if args.rolled:
            y_train = data['rol_y_train']
            y_val = data['rol_y_val']
        else:
            y_train = data['reg_y_train']
            y_val = data['reg_y_val']

    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)

    embedding_matrix = np.load('embedding_matrix.npy')

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_matrix.shape[1],
                        weights=[embedding_matrix], input_length=x_train.shape[1],
                        trainable=False))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    metrics = Metrics('{}.log'.format(filename))
    early_stopping = EarlyStopping(min_delta=.0001, patience=2)
    model.fit(x_train, y_train,
              batch_size=32, epochs=50,
              validation_data=(x_val, y_val),
              callbacks=[metrics, early_stopping])
    model.save('{}.h5'.format(filename))


if __name__ == '__main__':
    main()
