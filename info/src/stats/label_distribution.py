from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from util import read_diagnosis, read_notes


def main():
    diagnosis, icd9_lookup = read_diagnosis()
    texts, labels = read_notes(diagnosis)

    labels = [icd9_lookup[label][:3] for sublist in labels for label in sublist]

    cats = [xrange(1, 140), xrange(140, 240), xrange(240, 280), xrange(280, 290), xrange(290, 320),
            xrange(320, 390), xrange(390, 460), xrange(460, 520), xrange(520, 580), xrange(580, 630),
            xrange(630, 680), xrange(680, 710), xrange(710, 740), xrange(740, 760), xrange(760, 780),
            xrange(780, 800), xrange(800, 1000)]
    counts = [0] * (len(cats) + 1)

    for label in labels:
        if label.isdigit():
            label = int(label)
            for i, cat in enumerate(cats):
                if label in cat:
                    counts[i] += 1
                    break
        else:
            counts[-1] += 1

    print(counts)

    plt_labels = ['{}-{}'.format(cat[0], cat[-1]) for cat in cats] + ['E-V']
    ind = xrange(len(counts))

    plt.bar(ind, counts, align='center')
    plt.xticks(ind, plt_labels, rotation=45, rotation_mode='anchor', ha='right')

    plt.ylabel('Num. Occurences')
    plt.xlabel('ICD-9-CM Ranges')

    plt.savefig('chart_labels_distribution.png', bbox_inches='tight', dpi=900)
    # plt.show()
    

if __name__ == '__main__':
    main()
