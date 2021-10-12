# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:30:21 2020

@author: Lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,Dense,Input

x = np.random.randn(128*10).reshape((128,10))
x[x<=0] = 0
x[x>0] = 1
y = np.random.randn(128)
y[y<=0] = 0
y[y>0] = 1
x[:,0] = y
x = x.reshape((128,1,10))
y = y.reshape(-1,1)

tf.reset_default_graph()
sess = tf.Session()
keras.backend.set_session(sess)

inputs = Input((1,10),name='inp')
lstm = LSTM(1, return_sequences=False)(inputs)
model = Model(outputs = lstm,inputs=inputs)
model.summary()
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1/16), metrics=['accuracy'])
model.fit(x[:30],y[:30],4,8,validation_data=(x[30:],y[30:]))