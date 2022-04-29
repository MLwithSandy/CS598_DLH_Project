from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import time

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from models import ModelInput, MODELS


MAX_EPOCHS = 50
USE_ROLLED = True


def loaddata():
    with np.load('data.npz') as data:
        x = data['x']
        cats = data['cats']
        y = data['rol_y'] if USE_ROLLED else data['reg_y']
    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)
    embedding_matrix = np.load('embedding_matrix.npy')
    return x, cats, y, len(word_index), embedding_matrix


def evaluate_model(model, examples, targets):
    preds = np.around(model.predict(examples)).astype(np.int32)
    p = round(precision_score(targets, preds, average='micro') * 100, 4)
    r = round(recall_score(targets, preds, average='micro') * 100, 4)
    f1 = round(f1_score(targets, preds, average='micro') * 100, 4)
    return p, r, f1


def main():
    x, cats, y, vocab_dim, embeddings_matrix = loaddata()

    mi = ModelInput()
    mi.embeddings_matrix = embeddings_matrix
    mi.vocab_dim = vocab_dim
    mi.output_dim = y.shape[1]

    for model_fn in MODELS:
        print('NOW TRAINING:', model_fn.__name__)

        ps = []
        rs = []
        f1s = []
        trainds = []
        testds = []

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_idx, test_idx) in enumerate(kfold.split(y)):
            print('FOLD:', i + 1)

            x_train, x_test = x[train_idx], x[test_idx]
            cats_train, cats_test = cats[train_idx], cats[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model_fold_filename = '{}_{}.h5'.format(model_fn.__name__, i)

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

        with open(model_fn.__name__, 'w') as f:
            for p, r, f1, traind, testd in zip(ps, rs, f1s, trainds, testds):
                f.write('{} {} {} {} {}\n'.format(
                    p, r, f1, round(traind, 4), round(testd, 4)))


if __name__ == '__main__':
    main()
