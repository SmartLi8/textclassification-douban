# coding: utf-8
import re
from langconv import *
from gensim.models import word2vec
import jieba
import numpy as np
def Traditional2Simplified(sentence):
	'''
  	将sentence中的繁体字转为简体字
  	'''
	sentence = Converter('zh-hans').convert(sentence)
	return sentence

def seg_sentence(sentence,stop_words):
    """
    对句子进行分词
    :param sentence:句子，String
    """
    sentence_seged = jieba.cut(sentence.strip())
    out_str = ""
    for word in sentence_seged:
        if word not in stop_words:
            if word != " ":
                out_str += word
                out_str += " "
    return out_str

def load_stopwordslist(filepath):
    """
    加载停用词
    :param filepath:停用词文件路径
    :return:
    """
    with open(filepath,"r",encoding="utf-8") as file:
        stop_words = [line.strip() for line in file]
        return stop_words
def partition_dataset(dataset,label, ratio):
    data_len = len(dataset)
    x_train_index = np.random.choice(data_len,round(data_len*ratio),replace = False)
    x_test_index = np.array(list(set(range(data_len)) - set(x_train_index)))
    x_train = dataset[x_train_index]
    y_train = label[x_train_index]
    x_test = dataset[x_test_index]
    y_test = label[x_test_index]
    return x_train, y_train , x_test, y_test


if __name__ == '__main__':
	with open('./data/review.txt') as f:
	    reviews = f.readlines()

	texts = []
	n_tradition = 0
	for review in reviews:
	    try:
	        review.encode('big5hkscs')
	        review = Traditional2Simplified(review)
	        texts.append(review)
	        n_tradition += 1
	    except:
	        texts.append(review)
	texts[0] = re.sub(r"\\ufeff", '', texts[0])

	t_review = []
	label = []
	#去除特殊符号
	SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')
	#去除英文字符
	LETTER_RE = re.compile(r'[a-zA-Z]+')
	SHUZI = re.compile(r'[0-9]+')
	for text in texts:
	    if re.search(r'[a-zA-z]+://', text) is None:
	        t_list = text.strip().split('##')
	        t_list[2] = SPECIAL_SYMBOL_RE.sub('', t_list[2]).strip()
	        t_list[2] = LETTER_RE.sub('', t_list[2]).strip()
	        t_list[2] = SHUZI.sub('', t_list[2]).strip()
	        if t_list[2]:
	            t_review.append(t_list[2])
	            label.append(t_list[1])

	print(len(t_review), len(label))
	#分词
	stop_path = './data/stop_words.txt'
	stop_word = load_stopwordslist(stop_path)
	t_word = []
	for line in t_review:
	    word = seg_sentence(line, stop_word)
	    t_word.append(word)
	#list_word = [line.strip().split(' ')  for line in t_word ]
	#model = word2vec.Word2Vec(sentences=list_word,size=300,window=4,min_count=5,workers=10, iter=10)
	#model.wv.save_word2vec_format(fname="./data/wordembedding_4gram.txt",fvocab=None)

	label2 = []
	for i in label:
	    if i == '1' or i =='2':
	        label2.append('差评')
	    elif i == '3':
	        label2.append('中评')
	    else:
	        label2.append('好评')
	t_word = np.array(t_word)
	label2 = np.array(label2)
	x_train, y_train, x_test, y_test =  partition_dataset(t_word, label2, 0.9)

	listwrit = ['./data/x_train2.0', './data/x_test2.0', './data/y_train2.0', './data/y_test2.0']
	f0 = open(listwrit[0], 'w')
	f2 = open(listwrit[1], 'w')
	f3 = open(listwrit[2], 'w')
	f5 = open(listwrit[3], 'w')
	f0.write('\n'.join(x_train))
	f2.write('\n'.join(x_test))
	f3.write('\n'.join(y_train))
	f5.write('\n'.join(y_test))
	f0.close()
	f2.close()
	f3.close()
	f5.close()