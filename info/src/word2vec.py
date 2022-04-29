from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import cPickle as pickle
import string

from gensim.models.word2vec import Word2Vec


def main():
    filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
    tt = string.maketrans(string.digits + filters,
                          'd' * len(string.digits) + ' ' * len(filters))

    with open('infrequent_word_mapping.pickle', 'rb') as f:
        infrequent_word_mapping = pickle.load(f)

    seqs = []
    with open('notes.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            tokens = row[-1].lower().translate(tt).strip().split()
            for i, word in enumerate(tokens):
                tokens[i] = infrequent_word_mapping.get(word, word)
            seqs.append(tokens)

    print('Building word embeddings.')

    model = Word2Vec(seqs, size=300, workers=4)
    model.wv.save_word2vec_format('word2vec_embeddings.vec', binary=False)


if __name__ == '__main__':
    main()
