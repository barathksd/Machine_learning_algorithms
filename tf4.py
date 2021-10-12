# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:31:43 2019

@author: Lenovo
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import sklearn

data = keras.datasets.imdb
(x_tr,y_tr),(x_tst,y_tst) = data.load_data(num_words=12000)

word_index = data.get_word_index()
word_index = {key:val+3 for (key,val) in word_index.items()}
word_index['<PAD>']=0
word_index['<START>']=1
word_index['<UNK>']=2
word_index['<UNKNOWN>']=3
reverse_index = dict([(val,key) for (key,val) in word_index.items()])

def multi_hot(seq,dim):
    res = np.zeros((len(seq),dim))
    for i,words in enumerate(seq):    # has index and element of the array
        res[i] = words[:dim]
        
        if i==0:
            print(k for k in words)
    print(res[1])

multi_hot(x_tr,10)