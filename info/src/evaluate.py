from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from tensorflow.python.keras.models import load_model

from util import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-r', '--rolled', action='store_true')
    parser.add_argument('-c', '--categories', action='store_true')
    args = parser.parse_args()

    model = load_model(args.model)

    with np.load('data.npz') as data:
        x_test = data['x_test']
        if args.categories:
            cats_test = data['cats_test']
        if args.rolled:
            y_test = data['rol_y_test']
        else:
            y_test = data['reg_y_test']

    examples = [x_test, cats_test] if args.categories else [x_test]

    p, r, f1 = evaluate_model(model, examples, y_test)
    print('PRECISION: {} - RECALL: {} - F1: {}\n'.format(p, r, f1))


if __name__ == '__main__':
    main()
