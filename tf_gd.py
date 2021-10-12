# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:39:59 2019

@author: Lenovo
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def data():
    er = tf.random_normal([100,1],mean=0,stddev=0.2)
    print(er)
    x_d = tf.reshape(tf.range(1,21,0.2),[-1,1])
    y_t = tf.divide(tf.multiply(tf.constant(5,dtype=tf.float32),x_d),tf.add(x_d,2))
    y_d = tf.add(y_t,er)
    return x_d,y_d,y_t
    
def rmse(y, y_hat):
    return tf.sqrt(tf.reduce_mean(tf.square((y - y_hat))))

def forward(x, a,b):
    return tf.divide(tf.multiply(a,x),tf.sum(x,b))


lr = 0.4
i = 0
x_d,y_d,y_t = data()


while True:
    break 















