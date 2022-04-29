"""
Outputs:
* 'infrequent_word_mapping.pickle' if it doesn't exist
* `word_index.pickle`
* `icd9_lookup.npz`
* `shuffled_indices.npy` if it doesn't exist
* `data.npz`

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import cPickle as pickle
import os
import string

import Levenshtein
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# from util import read_diagnosis, read_notes


MIN_WORD_OCCURENCES = 5
MIN_SEQ_LEN = 9
MAX_SEQ_LEN = 2200


def read_data():
    regular_diagnosis = {}
    rolled_diagnosis = {}
    regular_icd9_lookup = []
    rolled_icd9_lookup = []

    with open('diagnosis.csv', 'rb') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            note_id = (row[1], row[2])
            regular_icd9 = row[4]
            rolled_icd9 = regular_icd9[:3]

            if regular_icd9 not in regular_icd9_lookup:
                regular_icd9_lookup.append(regular_icd9)
            if rolled_icd9 not in rolled_icd9_lookup:
                rolled_icd9_lookup.append(rolled_icd9)

            regular_note_diagnosis = regular_diagnosis.get(note_id, [])
            rolled_note_diagnosis = rolled_diagnosis.get(note_id, [])
            regular_idx = regular_icd9_lookup.index(regular_icd9)
            rolled_idx = rolled_icd9_lookup.index(rolled_icd9)

            if regular_idx not in regular_note_diagnosis:
                regular_diagnosis[note_id] = regular_note_diagnosis + [regular_idx]
            if rolled_idx not in rolled_note_diagnosis:
                rolled_diagnosis[note_id] = rolled_note_diagnosis + [rolled_idx]

    texts = []
    regular_labels = []
    rolled_labels = []

    unique_categories = []
    texts_categories = []

    tt = string.maketrans(string.digits, 'd' * len(string.digits))
    with open('notes.csv', 'rb') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            key = (row[1], row[2])
            cat = row[6]

            if cat not in unique_categories:
                unique_categories.append(cat)

            if key in regular_diagnosis:
                text = row[-1].strip().translate(tt)
                if text:
                    texts.append(text)
                    texts_categories.append(unique_categories.index(cat))
                    regular_labels.append(regular_diagnosis[key])
                    rolled_labels.append(rolled_diagnosis[key])

    return (texts, texts_categories, unique_categories,
            regular_labels, rolled_labels,
            regular_icd9_lookup, rolled_icd9_lookup)


def generate_infrequent_word_mapping(infrequent_words, frequent_words):
    if not os.path.exists('infrequent_word_mapping.pickle'):
        infrequent_word_mapping = {}
        for word in infrequent_words:
            dists = np.vectorize(lambda x: Levenshtein.distance(word, x))(frequent_words)
            most_similar_word = frequent_words[np.argmin(dists)]
            infrequent_word_mapping[word] = most_similar_word
        with open('infrequent_word_mapping.pickle', 'wb') as f:
            pickle.dump(infrequent_word_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('infrequent_word_mapping.pickle', 'rb') as f:
            infrequent_word_mapping = pickle.load(f)
    return infrequent_word_mapping


def main():
    (texts, texts_categories, unique_categories,
     regular_labels, rolled_labels,
     regular_icd9_lookup, rolled_icd9_lookup) = read_data()

    print('Average regular labels per report:',
          sum(map(len, regular_labels)) / len(regular_labels))
    print('Average rolled labels per report:',
          sum(map(len, rolled_labels)) / len(rolled_labels))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index

    print('Unique tokens *before* preprocessing:', len(word_index))

    # Segment words on frequently/infrequently occuring
    frequent_words = []
    infrequent_words = []
    for word, count in tokenizer.word_counts.iteritems():
        if count < MIN_WORD_OCCURENCES:
            infrequent_words.append(word)
        else:
            frequent_words.append(word)

    infrequent_word_mapping = generate_infrequent_word_mapping(infrequent_words,
                                                               frequent_words)

    # Replace infrequent word with a frequent similar word
    infrequent_word_index = {}
    for word in infrequent_words:
        most_similar_word = infrequent_word_mapping[word]
        infrequent_word_index[word] = word_index[most_similar_word]
        del word_index[word]

    print('Unique tokens *after* preprocessing:', len(word_index))

    # Reimplementation of `tokenizer.texts_to_sequences`
    sequences = []
    for text in texts:
        seq = text_to_word_sequence(text)
        vec = []
        for word in seq:
            idx = word_index.get(word)
            if idx is not None:
                vec.append(idx)
            else:
                vec.append(infrequent_word_index[word])
        sequences.append(vec)

    # with open('sequences.pickle', 'wb') as f:
        # pickle.dump(sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    # return

    # Sequence must be < MAX_SEQ_LEN and > MIN_SEQ_LEN
    seqs = []
    cats = []
    reg_labels = []
    rol_labels = []
    for seq, cat, reg, rol in zip(sequences, texts_categories, regular_labels, rolled_labels):
        if len(seq) < MAX_SEQ_LEN and len(seq) > MIN_SEQ_LEN:
            seqs.append(seq)
            cats.append(cat)
            reg_labels.append(reg)
            rol_labels.append(rol)
    sequences = seqs
    texts_categories = cats
    regular_labels = reg_labels
    rolled_labels = rol_labels

    lens = map(len, sequences)
    print('Shortest sequence has', min(lens), 'tokens')
    print('Longest sequences has', max(lens), 'tokens')
    print('Average tokens per sequence:', sum(lens) / len(sequences))

    with open('word_index.pickle', 'wb') as f:
        pickle.dump(tokenizer.word_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    texts = pad_sequences(sequences)

    # Encode labels as k-hot
    reg = np.zeros((len(regular_labels), len(regular_icd9_lookup)), dtype=np.int32)
    rol = np.zeros((len(rolled_labels), len(rolled_icd9_lookup)), dtype=np.int32)
    for i, label in enumerate(regular_labels): reg[i][label] = 1
    for i, label in enumerate(rolled_labels): rol[i][label] = 1
    regular_labels = reg
    rolled_labels = rol

    # Encode categories as 1-hot
    cats = np.zeros((len(texts_categories), len(unique_categories)), dtype=np.float32)
    for i, cat in enumerate(texts_categories): cats[i][cat] = 1
    texts_categories = cats

    # keep labels with >= 1 examples
    regular_icd9_lookup = np.asarray(regular_icd9_lookup)
    rolled_icd9_lookup = np.asarray(rolled_icd9_lookup)

    keep = np.sum(regular_labels, 0) >= 1
    regular_labels = regular_labels[:, keep]
    regular_icd9_lookup = regular_icd9_lookup[keep]
    keep = np.sum(rolled_labels, 0) >= 1
    rolled_labels = rolled_labels[:, keep]
    rolled_icd9_lookup = rolled_icd9_lookup[keep]

    np.savez('icd9_lookup.npz',
             regular_icd9_lookup=regular_icd9_lookup,
             rolled_icd9_lookup=rolled_icd9_lookup)

    print('Texts shape:', texts.shape)
    print('Categories shape:', texts_categories.shape)
    print('Regular labels shape:', regular_labels.shape)
    print('Rolled labels shape:', rolled_labels.shape)

    # Shuffle
    if os.path.exists('shuffled_indices.npy'):
        indices = np.load('shuffled_indices.npy')
    else:
        indices = np.arange(texts.shape[0])
        np.random.shuffle(indices)
        np.save('shuffled_indices.npy', indices)
    texts = texts[indices]
    texts_categories = texts_categories[indices]
    regular_labels = regular_labels[indices]
    rolled_labels = rolled_labels[indices]

    np.savez('data.npz',
             x=texts, cats=texts_categories,
             reg_y=regular_labels, rol_y=rolled_labels)

    # Split
    # s1 = int(.64 * len(texts))
    # s2 = int(.8 * len(texts))
    # x_train, x_val, x_test = np.split(texts, [s1, s2])
    # cats_train, cats_val, cats_test = np.split(texts_categories, [s1, s2])
    # reg_y_train, reg_y_val, reg_y_test = np.split(regular_labels, [s1, s2])
    # rol_y_train, rol_y_val, rol_y_test = np.split(rolled_labels, [s1, s2])

    # np.savez('data.npz',
             # x_train=x_train, x_val=x_val, x_test=x_test,
             # cats_train=cats_train, cats_val=cats_val, cats_test=cats_test,
             # reg_y_train=reg_y_train, reg_y_val=reg_y_val, reg_y_test=reg_y_test,
             # rol_y_train=rol_y_train, rol_y_val=rol_y_val, rol_y_test=rol_y_test)


if __name__ == '__main__':
    main()
