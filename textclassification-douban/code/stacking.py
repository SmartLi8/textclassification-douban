import pickle
import glob
import pandas as pd
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from keras.utils import to_categorical
from keras.optimizers import *
from keras.callbacks import *
from sklearn.metrics import f1_score
def data_prepare():
    filepath = '../stacking/'
    files = os.listdir(filepath)
    x_train = []
    x_test = []
    for i,file in enumerate(files):
        npzfile = np.load(filepath + file)
        feature = npzfile['train']
        x_train = feature if i ==0 else np.concatenate((x_train, feature), axis=-1)
        feature = npzfile['test']
        x_test = feature if i ==0 else np.concatenate((x_test, feature), axis=-1)
    lb = LabelEncoder()
    y_train, y_test= [],[]
    with open('../data/y_train_2c', 'r') as f1:
        for line in f1:
            y_train.append(line.strip())
    with open('../data/y_test_2c', 'r') as f3:
        for line in f3:
            y_test.append(line.strip())
    y_train, y_test = np.array(y_train),np.array(y_test)
    train_label = lb.fit_transform(y_train)
    train_label = to_categorical(train_label)
    test_label = lb.transform(y_test)
    test_label = to_categorical(test_label)



    print('train_x shape: ', x_train.shape)
    print('train_y shape: ', train_label.shape)
    print('test_x shape: ', x_test.shape)

    return x_train,train_label,x_test,test_label, lb


def get_model(train_x):
    input_shape = Input(shape=(train_x.shape[1],), name='dialogs')
    x = Dense(256, activation='relu')(input_shape)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)
    res_model = Model(inputs=[input_shape], outputs=x)
    return res_model


# 第一次stacking
def stacking_first(train, train_y, test,test_label,lb):
    savepath = '../stacking/'
    os.makedirs(savepath, exist_ok=True)

    count_kflod = 0
    num_folds = 6
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10)
    predict = np.zeros((test.shape[0], 2))
    train_predict = np.zeros((train.shape[0], 2))
    scores = []
    f1s = []
    #lb = LabelEncoder()
    for train_index, test_index in kf.split(train):

        kfold_X_train = {}
        kfold_X_valid = {}

        y_train, y_test = train_y[train_index], train_y[test_index]

        kfold_X_train, kfold_X_valid = train[train_index], train[test_index]

        model_prefix = savepath + 'DNN' + str(count_kflod) + '.hdf5'


        res_model = get_model(train)
        res_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_acc', patience=4)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=3)
        checkpoint = ModelCheckpoint(model_prefix , monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        # res_model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,  class_weight=class_weight)
        res_model.fit(kfold_X_train, y_train, batch_size=BATCH_SIZE, epochs=5, verbose=1,
                      validation_data=(kfold_X_valid, y_test),
                      callbacks=[early_stopping, plateau, checkpoint])
        res_model.load_weights(model_prefix)
        pre = res_model.predict(kfold_X_valid)
        print ("valid's accuracy: %s" % f1_score(lb.inverse_transform(np.argmax(y_test, 1)), 
                                                          lb.inverse_transform(np.argmax(kfold_X_valid, 1)).reshape(-1,1),
                                                          average='micro'))
    
        train_predict[test_fold, :] =  pre
        predict += res_model.predict(x_test)
        
        del model; gc.collect()
        K.clear_session()
    #线下测试
    print ("offline train score: %s" % f1_score(lb.inverse_transform(np.argmax(train_y, 1)), 
                                                  lb.inverse_transform(np.argmax(train_predict, 1)).reshape(-1,1),
                                                  average='micro'))
    print ("online test score: %s" % f1_score(lb.inverse_transform(np.argmax(test_label, 1)), 
                                                  lb.inverse_transform(np.argmax(predict, 1)).reshape(-1,1),
                                                  average='micro'))  

 
    return predict


def save_result(predict, prefix):
    os.makedirs('../data/result', exist_ok=True)
    with open('../data/result/{}.pkl'.format(prefix), 'wb') as f:
        pickle.dump(predict, f)

    res = pd.DataFrame()
    test_id = pd.read_csv(config.TEST_X)

    label_stoi = pickle.load(open('../data/label_stoi.pkl', 'rb'))
    label_itos = {v: k for k, v in label_stoi.items()}

    results = np.argmax(predict, axis=-1)
    results = [label_itos[e] for e in results]
    res['id'] = test_id['id']
    res['class'] = results
    res.to_csv(prefix+'.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    BATCH_SIZE = 128
    train, train_y, test,y_test,lb  = data_prepare()
    stacking_first(train, train_y, test,y_test,lb)
    #save_result(predicts, prefix='stacking_first_op{}_{}_{}'.format(args.option, args.data_type, args.tfidf))

