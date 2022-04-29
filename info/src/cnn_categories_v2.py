from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle

import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Input,
    Embedding,
    Dense,
    Conv1D,
    GlobalMaxPooling1D,
    concatenate,
)

from util import Metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rolled', action='store_true')
    args = parser.parse_args()

    filename = __file__[:-3] + '_rolled' if args.rolled else __file__[:-3]

    with np.load('data.npz') as data:
        x_train = data['x_train']
        x_val = data['x_val']
        cats_train = data['cats_train']
        cats_val = data['cats_val']
        if args.rolled:
            y_train = data['rol_y_train']
            y_val = data['rol_y_val']
        else:
            y_train = data['reg_y_train']
            y_val = data['reg_y_val']

    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)

    embedding_matrix = np.load('embedding_matrix.npy')

    embedding_input = Input(shape=(x_train.shape[1],), dtype=np.int32)
    category_input = Input(shape=(cats_train.shape[1],), dtype=np.float32)

    embedding_layer = Embedding(len(word_index) + 1, embedding_matrix.shape[1],
                                weights=[embedding_matrix], input_length=x_train.shape[1],
                                trainable=False)(embedding_input)
    x = Conv1D(300, 3, activation='relu')(embedding_layer)
    x = GlobalMaxPooling1D()(x)
    category_layer = Dense(64, activation='relu')(category_input)
    x = concatenate([x, category_layer])
    # x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(y_train.shape[1], activation='sigmoid')(x)

    model = Model(inputs=[embedding_input, category_input], outputs=[output])

    metrics = Metrics('{}.log'.format(filename))
    early_stopping = EarlyStopping(min_delta=.0001, patience=2)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit([x_train, cats_train], y_train,
              batch_size=32, epochs=50,
              validation_data=([x_val, cats_val], y_val),
              callbacks=[metrics, early_stopping])
    model.save('{}.h5'.format(filename))


if __name__ == '__main__':
    main()
