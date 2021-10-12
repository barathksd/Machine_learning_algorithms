# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:32:14 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
file1 = '/data/horse_tr.csv'
file2 = '/horse_tst.csv'
def load(file1,file2):
    data_train = pd.read_csv(file1,delimiter='\t',header=None)
    data_test = pd.read_csv(file2,delimiter='\t',header=None)
    x_train = np.array(data_train.filter(items=[i for i in range(data_train.shape[1]-1)]))
    b = np.ones((x_train.shape[0],1))
    xtr = np.mat(np.append(x_train,b,axis=1))    #299x22
    y_train = np.mat(data_train[data_train.shape[1]-1]).T   #299x1
    
    x_test = np.array(data_test.filter(items=[i for i in range(data_train.shape[1]-1)]))
    b = np.ones((x_test.shape[0],1))
    xtst = np.mat(np.append(x_test,b,axis=1))   #67x22
    y_test = np.mat(data_test[data_train.shape[1]-1]).T  #67x1
    return xtr,y_train,xtst,y_test
x_tr,y_tr,x_tst,y_tst = load(file1,file2)

w = np.mat(np.random.uniform(-0.2,0.2,x_tr.shape[1])).T  #22x1
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)


def sig(wt,x):
    return 1./(1+np.exp(-(x*wt)))

def train(w,x_tr,y_train):
    alp = 0.0075
    cyc = 250
    for i in range(cyc):
        h = sig(w,x_tr)  #299x1
        err = y_train - h  #299x1
        w = w + alp*x_tr.T*err
        print(np.sum(np.abs(err)))
    return w

wtr = train(w,x_tr,y_tr)

def test(w,x_test,y_test):
     h = sig(w,x_test)  #299x1
     h[h<0.5]=0
     h[h>=0.5]=1
     print('h\n',h.T)
     print('y\n',y_test.T)
     print('err test',np.sum(np.abs((y_test-h))))
     print('')
     
     
test(wtr,x_tst,y_tst)
test(wtr,x_tr,y_tr)