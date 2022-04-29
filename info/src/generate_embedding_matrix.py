from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle

import numpy as np


def main():
    # Load pre-trained embeddings
    embeddings = {}
    with open('word2vec_embeddings.vec', 'rb') as f:
        embeddings_dim = int(next(f).split()[1])
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings[word] = coefs

    with open('word_index.pickle', 'rb') as f:
        word_index = pickle.load(f)

    print('Vocabulary:', len(word_index))

    # Compute embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, embeddings_dim))
    unknown_count = 0
    for idx, (word, i) in enumerate(word_index.iteritems()):
        vec = embeddings.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
        else:
            print(word)
            unknown_count += 1

    print('Unknown:', unknown_count)

    np.save('embedding_matrix.npy', embedding_matrix)


if __name__ == '__main__':
    main()
