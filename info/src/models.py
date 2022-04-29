from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from tensorflow.python.keras.layers import (
    Input,
    Embedding,
    Dense,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Conv1D,
    concatenate)
from tensorflow.python.keras.models import Model


class ModelInput(object):
    def __init__(self):
        self.x_train = None
        self.cats_train = None
        self.embeddings_matrix = None
        self.vocab_dim = None
        self.output_dim = None

# ===================================================================
# Pieces
# ===================================================================

def _bot_prelude(mi):
    embedding_input = Input(shape=(mi.x_train.shape[1],), dtype=np.int32)
    x = Embedding(mi.vocab_dim + 1, mi.embeddings_matrix.shape[1],
                  weights=[mi.embeddings_matrix], input_length=mi.x_train.shape[1],
                  trainable=False)(embedding_input)
    x = GlobalAveragePooling1D()(x)
    return x, embedding_input


def _cnn_prelude(mi):
    embedding_input = Input(shape=(mi.x_train.shape[1],), dtype=np.int32)
    x = Embedding(mi.vocab_dim + 1, mi.embeddings_matrix.shape[1],
                  weights=[mi.embeddings_matrix], input_length=mi.x_train.shape[1],
                  trainable=False)(embedding_input)
    x = Conv1D(250, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    return x, embedding_input


def _cnn_triple_conv_prelude(mi):
    embedding_input = Input(shape=(mi.x_train.shape[1],), dtype=np.int32)
    x = Embedding(mi.vocab_dim + 1, mi.embeddings_matrix.shape[1],
                  weights=[mi.embeddings_matrix], input_length=mi.x_train.shape[1],
                  trainable=False)(embedding_input)
    x1 = Conv1D(250, 2, activation='relu')(x)
    x1 = GlobalMaxPooling1D()(x1)
    x2 = Conv1D(250, 3, activation='relu')(x)
    x2 = GlobalMaxPooling1D()(x2)
    x3 = Conv1D(250, 4, activation='relu')(x)
    x3 = GlobalMaxPooling1D()(x3)
    x = concatenate([x1, x2, x3])
    return x, embedding_input


def _categories_v1(mi, x, num_dense):
    category_input = Input(shape=(mi.cats_train.shape[1],), dtype=np.float32)
    x = concatenate([x, category_input])
    for i in xrange(num_dense):
        x = Dense(64, activation='relu')(x)
    return x, category_input


def _output(mi, x, embedding_input, category_input=None):
    output = Dense(mi.output_dim, activation='sigmoid')(x)
    model = Model(inputs=([embedding_input, category_input]
                          if category_input is not None
                          else [embedding_input]),
                  outputs=[output])
    return model

# ===================================================================
# BoT
# ===================================================================

def bot_baseline(mi):
    x, embedding_input = _bot_prelude(mi)
    return _output(mi, x, embedding_input)


def bot_categories_v1_0(mi):
    x, embedding_input = _bot_prelude(mi)
    x, category_input = _categories_v1(mi, x, 0)
    return _output(mi, x, embedding_input, category_input)


def bot_categories_v1_1(mi):
    x, embedding_input = _bot_prelude(mi)
    x, category_input = _categories_v1(mi, x, 1)
    return _output(mi, x, embedding_input, category_input)


def bot_categories_v1_2(mi):
    x, embedding_input = _bot_prelude(mi)
    x, category_input = _categories_v1(mi, x, 2)
    return _output(mi, x, embedding_input, category_input)


def bot_categories_v1_3(mi):
    x, embedding_input = _bot_prelude(mi)
    x, category_input = _categories_v1(mi, x, 3)
    return _output(mi, x, embedding_input, category_input)

# ===================================================================
# CNN
# ===================================================================

def cnn_baseline(mi):
    x, embedding_input = _cnn_prelude(mi)
    return _output(mi, x, embedding_input)


def cnn_triple_conv(mi):
    x, embedding_input = _cnn_triple_conv_prelude(mi)
    return _output(mi, x, embedding_input)


def cnn_categories_v1_0(mi):
    x, embedding_input = _cnn_prelude(mi)
    x, category_input = _categories_v1(mi, x, 0)
    return _output(mi, x, embedding_input, category_input)


def cnn_categories_v1_1(mi):
    x, embedding_input = _cnn_prelude(mi)
    x, category_input = _categories_v1(mi, x, 1)
    return _output(mi, x, embedding_input, category_input)


def cnn_categories_v1_2(mi):
    x, embedding_input = _cnn_prelude(mi)
    x, category_input = _categories_v1(mi, x, 2)
    return _output(mi, x, embedding_input, category_input)


def cnn_categories_v1_3(mi):
    x, embedding_input = _cnn_prelude(mi)
    x, category_input = _categories_v1(mi, x, 3)
    return _output(mi, x, embedding_input, category_input)


MODELS = [
    bot_baseline,
    cnn_baseline,
    # bot_categories_v1_0,
    # bot_categories_v1_1,
    # bot_categories_v1_2,
    bot_categories_v1_3,
    # cnn_categories_v1_0,
    # cnn_categories_v1_1,
    # cnn_categories_v1_2,
    cnn_categories_v1_3,
    # cnn_triple_conv,
]
