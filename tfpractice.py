# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:44:34 2019

@author: Lenovo
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from sklearn.datasets import load_diabetes   # linear regression
from sklearn.preprocessing import StandardScaler

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*y + y + 2
g = f*y
h = f*x

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    a = g.eval()
    b = h.eval()
    #print(a,b)     #tensorflow does not reuse the code in computation
    c = sess.run([g,h])     #computation is done once in the same graph
    #print(c)
    sess.close()
    
db = load_diabetes()
data = db['data']
target = db['target'].reshape(-1,1)
r,c = data.shape


def linreg():
    
    ndata = np.c_[np.ones((r, 1)), data]
    #print(ndata[1])
    x = tf.constant(ndata,dtype=tf.float32,name='x') # 442x11
    y = tf.constant(target,dtype=tf.float32,name='y') # 442x1
    xt = tf.transpose(x,name='xt') # 11x442
    wt = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xt,x)),xt),y)
    
    with tf.Session() as sess:  # tf can automatically run on GPU unlike np computation
        weights = wt.eval()
        print(weights)
        sess.close()
        
        

def sgd():   #normalize the data before sgd
    
    al = 0.1
    n_epoch = 10000
    sc = StandardScaler()
    data_sc = sc.fit_transform(data)
    ndata = np.c_[np.ones((r,1)),data_sc]
    print(ndata.shape,c+1,target.shape)
    
    tf.reset_default_graph()
    x = tf.constant(ndata, dtype=tf.float32,name='x') # 442x11
    y = tf.constant(target, dtype=tf.float32,name='y') # 442x1
  
    w = tf.Variable(tf.random_uniform([c+1,1],-10,10),name='w') # 11x1
    y_pred = tf.matmul(x, w, name="predictions")
    er = y_pred - y # 442x1
    mse = tf.reduce_mean(tf.square(er),name='mse')
    #grad = 2/r*(tf.matmul(tf.transpose(x),er))
    #train_op = tf.assign(w,w - al*grad)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=al)
    train_op = optimizer.minimize(mse)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epoch):
            if epoch % 3000 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
                save_path = saver.save(sess, "/tmp/my_model.ckpt")
            sess.run(train_op)
        save_path = saver.save(sess, "/tmp/my_model.ckpt")
        best_w = w.eval()
        print(best_w,'\n')
        print(y_pred.eval()[:5])
        print('\ntarget\n',y.eval()[:5])
        sess.close()

sgd()
