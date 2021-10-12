# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:30:30 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

categories = ['rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']
newsgroups_train = fetch_20newsgroups(subset='train',
categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test',
categories=categories, remove=('headers', 'footers', 'quotes'))

scaler = StandardScaler(with_mean=False,with_std=False)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)  #ax18242
y_train = np.mat(newsgroups_train.target)
X_test = vectorizer.transform(newsgroups_test.data)
y_test = np.mat(newsgroups_test.target)    

#x_tr = scaler.fit_transform(X_train)
#x_tst = scaler.transform(X_test)

classifier = Perceptron(max_iter=100, eta0=0.1,tol=1e-3)
classifier.fit(X_train, newsgroups_train.target )
predictions = classifier.predict(X_test)
print (classification_report(newsgroups_test.target, predictions))

     
    


