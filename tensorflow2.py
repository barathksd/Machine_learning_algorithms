# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:48:35 2019

@author: Lenovo
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

data = keras.datasets.imdb
(x_tr,y_tr),(x_tst,y_tst) = data.load_data(num_words=12000)

word_index = data.get_word_index()
word_index = {key:val+3 for (key,val) in word_index.items()}
word_index['<PAD>']=0
word_index['<START>']=1
word_index['<UNK>']=2
word_index['<UNKNOWN>']=3
reverse_index = dict([(val,key) for (key,val) in word_index.items()])

k = [reverse_index[i] for i in x_tr[0]]
#r = ' '.join(k)
#print(r)

m = x_tr.shape[0]   #25000x1
mx = np.max([len(x_tr[i]) for i in range(m)])

print('preprocessing..')
xtr_pros  = keras.preprocessing.sequence.pad_sequences(x_tr,value=0,padding='post',maxlen=256)
xtst_pros = keras.preprocessing.sequence.pad_sequences(x_tst,value=0,padding='post',maxlen=256)

vocab_size = 12000

model = keras.Sequential([
    keras.layers.Embedding(12000,16),        
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
model.summary()
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

x_train = xtr_pros[10000:]
x_dev = xtr_pros[:10000]

y_train = y_tr[10000:]
y_dev = y_tr[:10000]

print('ready to fit')
history = model.fit(x_train,y_train,
                epochs=20,
                batch_size=512,
                validation_data=(x_dev,y_dev),
                verbose=1
)

print(model.evaluate(xtst_pros,y_tst))





