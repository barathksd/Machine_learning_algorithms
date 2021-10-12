# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow import keras

data = keras.datasets.fashion_mnist

(x_tr,y_tr),(x_tst,y_tst) = data.load_data()
x_tr = x_tr/255
x_tst = x_tst/255

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(units = 100, activation = tf.nn.relu),
    keras.layers.Dense(units = 50, activation = tf.nn.relu),
    keras.layers.Dense(units = 25, activation = tf.nn.relu),
    keras.layers.Dense(units = 10, activation = tf.nn.softmax)
])

model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_tr,y_tr,epochs=60,batch_size=1000)

pred = model.predict(x_tst)
(tst_loss,tst_acc) = model.evaluate(x_tst,y_tst)
print('acc',tst_acc)






