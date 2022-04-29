from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import string

import numpy as np
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score


def read_diagnosis():
    diagnosis = {}
    icd9_lookup = []
    with open('diagnosis.csv', 'rb') as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            note_id = (row[1], row[2])
            icd9 = row[4]
            if icd9 not in icd9_lookup:
                icd9_lookup.append(icd9)
            diagnosis[note_id] = diagnosis.get(note_id, []) + [icd9_lookup.index(icd9)]
    return diagnosis, icd9_lookup


def read_notes(diagnosis):
    # Keras' `Tokenizer` removes all special chars and lowercases all
    # tokens.
    texts = []
    labels = []
    tt = string.maketrans(string.digits, 'd' * len(string.digits))
    with open('notes.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            # if row[6] != 'Discharge summary': continue
            key = (row[1], row[2])
            if key in diagnosis:
                text = row[-1].strip().translate(tt)
                if text:
                    texts.append(text)
                    labels.append(diagnosis[key])
    return texts, labels


def weighed_binary_crossentropy(target, output):
    epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon())
    output = tf.clip_by_value(output, epsilon, 1 - epsilon)
    output = tf.log(output / (1 - output))
    loss = tf.nn.weighted_cross_entropy_with_logits(
        targets=target,
        logits=output,
        pos_weight=33)
    return tf.reduce_mean(loss, axis=-1)


def evaluate_model(model, examples, targets):
    preds = np.around(model.predict(examples)).astype(np.int32)
    p = round(precision_score(targets, preds, average='micro') * 100, 2)
    r = round(recall_score(targets, preds, average='micro') * 100, 2)
    f1 = round(f1_score(targets, preds, average='micro') * 100, 2)
    return p, r, f1


class Metrics(Callback):
    def __init__(self, filename):
        super(Metrics, self).__init__()
        self.filename = filename
        self.ps = []
        self.rs = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs=None):
        p, r, f1 = evaluate_model(self.model,
                                  self.validation_data[:-2],
                                  self.validation_data[-2])
        self.ps.append(p)
        self.rs.append(r)
        self.f1s.append(f1)
        print('PRECISION: {} - RECALL: {} - F1: {}\n'.format(p, r, f1))

    def on_train_end(self, logs=None):
        with open(self.filename, 'w') as f:
            for p, r, f1 in zip(self.ps, self.rs, self.f1s):
                f.write('{} {} {}\n'.format(p, r, f1))
