from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import time

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from models import ModelInput, MODELS


MAX_EPOCHS = 50
TOP_INDEXES = [21, 60, 18, 16,  3, 70,  0, 22,  4, 54]

def loaddata():
    with np.load('data.npz') as data:
        x = data['x']
        cats = data['cats']
    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)
    embedding_matrix = np.load('embedding_matrix.npy')
    return x, cats, len(word_index), embedding_matrix


def load_y(idx):
    with np.load('data.npz') as data:
        y = data['rol_y'][:, idx]
    return y


def evaluate_model(model, examples, targets):
    preds = np.around(model.predict(examples)).astype(np.int32)
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
    return round(p * 100, 4), round(r * 100, 4), round(f1 * 100, 4)


def main():
    x, cats, vocab_dim, embeddings_matrix = loaddata()

    mi = ModelInput()
    mi.embeddings_matrix = embeddings_matrix
    mi.vocab_dim = vocab_dim
    mi.output_dim = 1

    for idx in TOP_INDEXES:
        y = load_y(idx)

        for model_fn in MODELS:
            print('NOW TRAINING:', model_fn.__name__)

            ps = []
            rs = []
            f1s = []
            trainds = []
            testds = []

            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            for i, (train_idx, test_idx) in enumerate(kfold.split(y)):
                if i == 0: continue
                print('FOLD:', i + 1)

                x_train, x_test = x[train_idx], x[test_idx]
                cats_train, cats_test = cats[train_idx], cats[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model_fold_filename = '{}_{}_{}.h5'.format(idx, model_fn.__name__, i)

                mi.x_train = x_train
                mi.cats_train = cats_train

                early_stopper = EarlyStopping(min_delta=.0001, patience=2)
                checkpointer = ModelCheckpoint(model_fold_filename,
                                               save_best_only=True,
                                               save_weights_only=True)

                model = model_fn(mi)
                model.compile(loss='binary_crossentropy', optimizer='adam')

                t0 = time.time()
                model.fit(([x_train, cats_train]
                           if 'categories' in model_fn.__name__
                           else [x_train]),
                          y_train,
                          batch_size=32, epochs=MAX_EPOCHS,
                          validation_split=.1,
                          callbacks=[early_stopper, checkpointer])
                d = time.time() - t0
                trainds.append(d)

                model.load_weights(model_fold_filename)

                t0 = time.time()
                p, r, f1 = evaluate_model(model,
                                          ([x_test, cats_test]
                                           if 'categories' in model_fn.__name__
                                           else [x_test]),
                                          y_test)
                d = time.time() - t0
                ps.append(p)
                rs.append(r)
                f1s.append(f1)
                testds.append(d)
                print('PRECISION: {} - RECALL: {} - F1: {}\n'.format(p, r, f1))

            with open('{}_{}.log'.format(idx, model_fn.__name__), 'a') as f:
                for p, r, f1, traind, testd in zip(ps, rs, f1s, trainds, testds):
                    f.write('{} {} {} {} {}\n'.format(
                        p, r, f1, round(traind, 4), round(testd, 4)))


if __name__ == '__main__':
    main()
