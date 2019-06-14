import os
import re
import sys
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.utils import np_utils
import warnings
warnings.filterwarnings('ignore')
import os
import gc
from keras.engine.topology import Layer
def textrnn(sent_length, embeddings_weight,class_num):
    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)
    x = BatchNormalization()(embedding(content))
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.3))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.3))(x)
    x = Dropout(0.3)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.3)(Activation(activation="relu")(BatchNormalization()(Dense(100, kernel_regularizer=regularizers.l2(0.03))(conc))))
    output = Dense(class_num, activation="sigmoid")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

