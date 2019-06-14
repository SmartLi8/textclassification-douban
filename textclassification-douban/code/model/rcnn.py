# coding=utf-8

from keras import Input, Model
from keras import backend as K
from keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D


def RCNN(maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):

        input_current = Input((maxlen,))
        input_left = Input((maxlen,))
        input_right = Input((maxlen,))

        embedder = Embedding(max_features, embedding_dims, input_length=maxlen)
        embedding_current = embedder(input_current)
        embedding_left = embedder(input_left)
        embedding_right = embedder(input_right)

        x_left = SimpleRNN(128, return_sequences=True)(embedding_left)
        x_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedding_right)
        x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)
        x = Concatenate(axis=2)([x_left, embedding_current, x_right])

        x = Conv1D(64, kernel_size=1, activation='tanh')(x)
        x = GlobalMaxPooling1D()(x)

        output = Dense(class_num, activation=last_activation)(x)
        model = Model(inputs=[input_current, input_left, input_right], outputs=output)
        return model
