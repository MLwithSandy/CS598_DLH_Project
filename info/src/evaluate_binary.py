from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from tensorflow.python.keras.models import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-c', '--categories', action='store_true')
    args = parser.parse_args()

    model = load_model(args.model)
    idx = args.model.split('_')[-1][:-3]

    with np.load('data.npz') as data:
        x_test = data['x_test']
        if args.categories:
            cats_test = data['cats_test']
        y_test = data['rol_y_test'][:, int(idx)]

    examples = [x_test, cats_test] if args.categories else [x_test]

    print('ACCURACY:', model.evaluate(examples, y_test)[1] * 100)


if __name__ == '__main__':
    main()
