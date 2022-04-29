from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle as pickle

import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Input,
    Embedding,
    Dense,
    GlobalAveragePooling1D,
    concatenate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('idx')
    args = parser.parse_args()

    filename = __file__[:-3] + '_' + args.idx

    with np.load('data.npz') as data:
        x_train = data['x_train']
        x_val = data['x_val']
        cats_train = data['cats_train']
        cats_val = data['cats_val']
        y_train = data['rol_y_train'][:, int(args.idx)]
        y_val = data['rol_y_val'][:, int(args.idx)]

    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)

    embedding_matrix = np.load('embedding_matrix.npy')

    embedding_input = Input(shape=(x_train.shape[1],), dtype=np.int32)
    category_input = Input(shape=(cats_train.shape[1],), dtype=np.float32)

    embedding_layer = Embedding(len(word_index) + 1, embedding_matrix.shape[1],
                                weights=[embedding_matrix], input_length=x_train.shape[1],
                                trainable=False)(embedding_input)
    x = GlobalAveragePooling1D()(embedding_layer)
    x = concatenate([x, category_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[embedding_input, category_input], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    early_stopping = EarlyStopping(min_delta=.0001, patience=2)
    checkpoint = ModelCheckpoint(filename + '.h5', save_best_only=True)
    model.fit([x_train, cats_train], y_train,
              batch_size=32, epochs=50,
              validation_data=([x_val, cats_val], y_val),
              callbacks=[early_stopping, checkpoint])


if __name__ == '__main__':
    main()
