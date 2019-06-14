# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


def TextCNN(sent_length, embeddings_weight,
                 class_num=1,
                 last_activation='sigmoid'):

    content = Input(shape=(sent_length,), dtype='int32')
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=False)(content)

    convs = []
    for kernel_size in [3, 4, 5]:
        c = Conv1D(128, kernel_size, activation='relu')(embedding)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)

    output = Dense(class_num, activation=last_activation)(x)
    model = Model(inputs=content, outputs=output)
    model.summary()
    return model
