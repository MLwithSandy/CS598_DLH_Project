from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from tensorflow.python.keras.layers import concatenate
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

from multi_label import loaddata
from models import ModelInput, MODELS


TOP_INDEXES = [21, 60, 18, 16,  3, 70,  0, 22,  4, 54]


def evaluate_model(models, examples, targets):
    preds = [np.around(model.predict(examples)).astype(np.int32)
             for model in models]
    preds = np.concatenate(preds, axis=1)
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

    # x = x[:1000]
    # y = y[:1000]

    mi = ModelInput()
    mi.embeddings_matrix = embeddings_matrix
    mi.vocab_dim = vocab_dim
    mi.output_dim = 1

    for model_fn in MODELS:
        if model_fn.__name__ == args.model_name: break
    else:
        print("ERROR")
        return

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kfold.split(y)):
        x_train, x_test = x[train_idx], x[test_idx]
        cats_train, cats_test = cats[train_idx], cats[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mi.x_train = x_train
        mi.cats_train = cats_train

        filenames = ['{}_{}_{}.h5'.format(idx, model_fn.__name__, i)
                     for idx in TOP_INDEXES]
        models = []
        for filename in filenames:
            model = model_fn(mi)
            model.load_weights(filename)
            models.append(model)

        p, r, f1 = evaluate_model(models,
                                  ([x_test, cats_test]
                                   if 'categories' in model_fn.__name__
                                   else [x_test]),
                                  y_test)
        print(p, r, f1)
        break


if __name__ == '__main__':
    main()
