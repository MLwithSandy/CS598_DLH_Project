from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import argparse

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

from models import ModelInput, MODELS


TOP_INDEXES = [21, 60, 18, 16,  3, 70,  0, 22,  4, 54]


def loaddata():
    with np.load('data.npz') as data:
        x = data['x']
        cats = data['cats']
        y = data['rol_y']
    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)
    embedding_matrix = np.load('embedding_matrix.npy')
    return x, cats, y, len(word_index), embedding_matrix


def evaluate_model(model, examples, targets):
    preds = np.around(model.predict(examples)).astype(np.int32)
    preds = preds[:, TOP_INDEXES]
    targets = targets[:, TOP_INDEXES]
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='micro')
    p = round(p * 100, 4)
    r = round(r * 100, 4)
    f1 = round(f1 * 100, 4)
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    args = parser.parse_args()

    x, cats, y, vocab_dim, embeddings_matrix = loaddata()

    with np.load('icd9_lookup.npz') as f:
        lookup = f['rolled_icd9_lookup']

    mi = ModelInput()
    mi.embeddings_matrix = embeddings_matrix
    mi.vocab_dim = vocab_dim
    mi.output_dim = y.shape[1]

    for model_fn in MODELS:
        if model_fn.__name__ == args.model_name: break
    else:
        print("ERROR")
        return

    ps = []
    rs = []
    f1s = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kfold.split(y)):
        x_train, x_test = x[train_idx], x[test_idx]
        cats_train, cats_test = cats[train_idx], cats[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mi.x_train = x_train
        mi.cats_train = cats_train

        model = model_fn(mi)
        model.load_weights('rolled/{}_{}.h5'.format(args.model_name, i))

        p, r, f1 = evaluate_model(model,
                                  ([x_test, cats_test]
                                   if 'categories' in model_fn.__name__
                                   else [x_test]),
                                  y_test)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        print(p, r, f1)
        break

    ps = np.mean(ps, axis=0)
    rs = np.mean(rs, axis=0)
    f1s = np.mean(f1s, axis=0)
    print(ps, rs, f1s)


if __name__ == '__main__':
    main()
