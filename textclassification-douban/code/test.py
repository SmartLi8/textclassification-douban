# coding: utf-8
import re
import pickle
from data_process import seg_sentence, load_stopwordslist
from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np


instr =  input("请输入要分析的影评:\n")
SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')
#去除英文字符
LETTER_RE = re.compile(r'[a-zA-Z]+')
SHUZI = re.compile(r'[0-9]+')
if re.search(r'[a-zA-z]+://', instr) is None:

    instr = SPECIAL_SYMBOL_RE.sub('', instr).strip()
instr = LETTER_RE.sub('', instr)
instr = SHUZI.sub('', instr)

stop_path = './data/stop_words.txt'
stop_word = load_stopwordslist(stop_path)
t_word = []

word = seg_sentence(instr, stop_word)
tokenizer =  pickle.load(open('./data/tokenizerclass.pkl', 'rb'))

word = sequence.pad_sequences(tokenizer.texts_to_sequences([word]), maxlen=70)
model = load_model('./cache/bi_lstm_model/bi_lstm_model.hdf5')
predict = model.predict(word)
result = np.argmax(predict)
if result == 0:
    resu = '好评'
else:
    resu = '差评'

print('输入的影评句子是：%s.\n获得的情感分析结果为:%s'%(instr,resu))


