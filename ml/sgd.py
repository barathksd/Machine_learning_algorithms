# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 21:54:11 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data,data.target)
y_train=np.array([y_train]).T  #1D to 2D array
y_test=np.array([y_test]).T 

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

y_test=y_test[:,0]
regressor = SGDRegressor(loss='squared_loss',max_iter=10000,tol=0.21)
sc = cross_val_score(regressor, X_train, y_train.ravel(), cv=5)
print(sc.mean())


