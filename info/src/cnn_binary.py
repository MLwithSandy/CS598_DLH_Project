from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('idx')
    args = parser.parse_args()

    filename = __file__[:-3] + '_' + args.idx

    with np.load('data.npz') as data:
        x_train = data['x_train']
        x_val = data['x_val']
        y_train = data['rol_y_train'][:, int(args.idx)]
        y_val = data['rol_y_val'][:, int(args.idx)]

    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)

    embedding_matrix = np.load('embedding_matrix.npy')

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_matrix.shape[1],
                        weights=[embedding_matrix], input_length=x_train.shape[1],
                        trainable=False))
    model.add(Conv1D(300, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    early_stopping = EarlyStopping(min_delta=.0001)
    checkpoint = ModelCheckpoint(filename + '.h5', save_best_only=True)
    model.fit(x_train, y_train,
              batch_size=32, epochs=25,
              validation_data=(x_val, y_val),
              callbacks=[early_stopping, checkpoint])


if __name__ == '__main__':
    main()
