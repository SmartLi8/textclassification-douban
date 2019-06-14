# coding: utf-8

import tensorflow as tf
import os
import re
import sys  
import gc
import random
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from gensim.test.utils import datapath
from scipy import stats
import keras
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from keras.engine.topology import Layer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils.training_utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score
import warnings
from model.textrnn import  textrnn
from model.text_cnn import TextCNN
import pickle
#词向量

def w2v_pad(train,valid, maxlen_,victor_size):

    tokenizer = text.Tokenizer(num_words=args.num_words, lower=False,filters="")
    tokenizer.fit_on_texts(train+valid)

    train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(train), maxlen=maxlen_)
    valid_ = sequence.pad_sequences(tokenizer.texts_to_sequences(valid), maxlen=maxlen_)
    pickle.dump(tokenizer, open('../data/tokenizerclass.pkl', 'wb'))
    word_index = tokenizer.word_index
    
    count = 0
    nb_words = len(word_index)
    print('the num of word is : ',nb_words)
    word2vec_path = args.word2vec_path
    w2v_model = {}
    wv_from_text = KeyedVectors.load_word2vec_format(datapath(word2vec_path), binary=False)
    
    word2vec_model = {}
    with open("../data/wordembedding.txt",encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word2vec_model[word] = coefs


    #del w2v_model[359278]         
    num_words = min(args.num_words, nb_words) + 1
    w2v_count=0
    embedding_w2v_matrix = np.zeros((num_words, victor_size))
    for word, i in word_index.items():
        if i > args.num_words:
            continue
        try:
            embedding_w2v_vector= wv_from_text[word]
            w2v_count += 1
            embedding_w2v_matrix[i] = embedding_w2v_vector
        except KeyError as e:
            try:
                embedding_w2v_matrix[i] = word2vec_model[word]
            except Exception as e:
                unk_vec = np.random.random(victor_size) * 0.5
                unk_vec = unk_vec - unk_vec.mean()
                embedding_w2v_matrix[i] = unk_vec
    print(embedding_w2v_matrix.shape, train_.shape)
    return train_, valid_, embedding_w2v_matrix





def train(my_model):
    num_words = args.num_words
    maxlen=args.maxlen
    victor_size=args.embedding_vector
    victor_size=args.embedding_vector
    train_model = eval(my_model)
    name = my_model
    
    lb = LabelEncoder()
    x_train, x_test, y_train, y_test= [],[],[],[]
    with open('../data/x_train_2c') as f0:
        for line in f0:
            x_train.append(line.strip())
    with open('../data/y_train_2c', 'r') as f1:
        for line in f1:
            y_train.append(line.strip())
    with open('../data/x_test_2c', 'r') as f2:
        for line in f2:
            x_test.append(line.strip())
    with open('../data/y_test_2c', 'r') as f3:
        for line in f3:
            y_test.append(line.strip())
    y_train, y_test = np.array(y_train),np.array(y_test)
    print('the num of train is:',len(x_train))

    x_train,x_test,word_embedding = w2v_pad(x_train,x_test, maxlen,victor_size)
    train_label = lb.fit_transform(y_train)
    train_label = to_categorical(train_label)
    test_label = lb.transform(y_test)
    test_label = to_categorical(test_label)

    if not os.path.exists("../cache/"+name):
        os.mkdir("../cache/"+name)

    kf = KFold(n_splits=args.KFold, shuffle=True, random_state=520).split(x_train)
    train_model_pred = np.zeros((x_train.shape[0], args.classification))
    test_model_pred = np.zeros((x_test.shape[0], args.classification))

    for i, (train_fold, test_fold) in enumerate(kf):
        xtrain, xvalid, = x_train[train_fold, :], x_train[test_fold, :]
        ytrain, yvalid = train_label[train_fold], train_label[test_fold]

        print(i, 'fold')

        the_path = '../cache/' + name +'/' + str(i) +  name + '.hdf5'
        model = train_model(maxlen,word_embedding, args.classification)
        adam_optimizer = Adam(lr=1e-3, clipvalue=2.4)
        model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_acc', patience=4)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(the_path , monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        if not os.path.exists(the_path + '.hdf5'):
        #print("error")
            model.fit(xtrain, ytrain,
                    epochs=2,
                    batch_size=args.batch_size,
                    validation_data=(xvalid, yvalid),
                    verbose=1,
                    )
            model.get_layer('word_embedding').trainable = True
            adam_optimizer = Adam(lr=1e-3, clipvalue=1.5)
            model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
            model.fit(xtrain, ytrain,
                        epochs=5,
                        batch_size=args.batch_size,
                        validation_data=(xvalid, yvalid),
                        callbacks=[early_stopping, plateau, checkpoint],
                        verbose=1)
        model.load_weights(the_path)
        pre = model.predict(xvalid)
        print (name + ": valid's accuracy: %s" % f1_score(lb.inverse_transform(np.argmax(yvalid, 1)), 
                                                          lb.inverse_transform(np.argmax(pre, 1)).reshape(-1,1),
                                                          average='micro'))
    
        train_model_pred[test_fold, :] =  pre
        test_model_pred += model.predict(x_test)
        
        del model; gc.collect()
        K.clear_session()
    #线下测试
    print (name + ": offline test score: %s" % f1_score(lb.inverse_transform(np.argmax(train_label, 1)), 
                                                  lb.inverse_transform(np.argmax(train_model_pred, 1)).reshape(-1,1),
                                                  average='micro'))
    #
    test_model_pred /= args.KFold
    np.savez("../stacking/" + name + '.npz', train=train_model_pred, test=test_model_pred)

if __name__ == '__main__':

    import argparse
    warnings.filterwarnings('ignore')
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)
    parser=argparse.ArgumentParser()
    #parser.add_argument("--gpu",type=str, default='0')
    parser.add_argument("--maxlen",type=int,  default=70)
    parser.add_argument("--embedding_vector",type=int,  default=300)
    parser.add_argument("--num_words",type=int, default=100000)
    parser.add_argument("--batch_size",type=int, default=128)
    parser.add_argument("--classification",type=int, default=2)
    parser.add_argument("--project_path", type=str)
    parser.add_argument("--word2vec_path", type=str, default='/media/lty/dataset/dataset/word2vec_300d')
    parser.add_argument("--KFold",type=int)
    parser.add_argument("--model_name",type=str)
    args=parser.parse_args()


    train(args.model_name)




