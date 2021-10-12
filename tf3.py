# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:01:33 2019

@author: Lenovo
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data = keras.datasets.boston_housing  
(x_tr,y_tr),(x_tst,y_tst) = data.load_data() 

temp = np.random.permutation(x_tr.shape[0])   #shuffle
xs_tr = x_tr[temp]
ys_tr = y_tr[temp]

xsc_tr = scaler.fit_transform(xs_tr)
xsc_tst = scaler.transform(x_tst)

def build_model():
    model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu,input_shape = (x_tr.shape[1],)),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
            ])
    
    optimizer = keras.optimizers.Adam(lr = 0.001)
    model.compile(loss='mse',
                   optimizer=optimizer,
                   metrics=['mae']
                   )
    return model

model = build_model()
model.summary()

class Printdot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch%100 == 0 :
            print('Epoch ',epoch)
ep = 500

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=25)
history = model.fit(xsc_tr,ys_tr,epochs=ep,validation_split=0.2,verbose=0,callbacks=[early_stop,Printdot()])



