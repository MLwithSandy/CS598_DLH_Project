"""Extracts texts for annotating with Neji"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util import read_diagnosis, read_notes


def evaluate():
    percents = []
    for i, text in enumerate(texts):
        annotations = []
        with open('../annotations/{}.a1'.format(i), 'rb') as f:
            for line in f:
                if line[0] == 'N':
                    ann = line.split(':')[1].replace('.', '')
                    annotations.append(ann)
        num_common = len(set(labels[i]) & set(annotations))
        percent = num_common / len(set(labels[i]))
        percents.append(percent)
        print('Doc:', i, ' ', num_common, 'of', len(set(labels[i])))

    print('Average correct:', sum(percents) / len(percents))


def main():
    diagnosis, icd9_lookup = read_diagnosis()
    texts, labels = read_notes(diagnosis)

    for i, text_labels in enumerate(labels):
        labels[i] = [icd9_lookup[idx] for idx in text_labels]

    # Write
    for i, text in enumerate(texts):
        with open('../texts/' + str(i), 'wb') as f:
            f.write(text)

    # evaluate()

if __name__ == '__main__':
    main()
