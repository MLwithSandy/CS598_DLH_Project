from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt


def main():
    with open('sequences.pickle', 'rb') as f:
        sequences = pickle.load(f)

    lens = np.asarray(map(len, sequences))
    # max 8513
    # min 1

    bins = np.arange(0, 8521, 20)
    idxs = np.digitize(lens, bins)
    bincount = np.bincount(idxs)

    # for i in xrange(1, len(bins)):
        # print('{} {} <= {} < {}'.format(i, bins[i - 1], bincount[i], bins[i]))

    # seqs = [seq for seq in sequences if len(seq) < 2200 and len(seq) > 9]
    # print(len(seqs) / len(sequences))

    ## Plot
    labels = []
    for i in xrange(1, len(bins)):
        labels.append('{}-{}'.format(bins[i - 1], bins[i]))

    coords = np.arange(40)
    plt.bar(coords, bincount[1:41], align='center', color='orange')
    plt.xticks(coords, labels[:len(coords)], rotation=45, rotation_mode='anchor', ha='right')
    plt.ylabel('Num. Reports')
    plt.xlabel('Num. Tokens')
    plt.tight_layout()
    plt.show()
    # plt.savefig('chart_report_length.png', bbox_inches='tight', dpi=900)


if __name__ == '__main__':
    main()
