# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 22:11:47 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df=pd.read_csv('C:\\Users\\Lenovo\\Desktop\\data\\wine.csv')
print(df.describe())
plt.scatter(df['Color intensity'],df['Quality'])
plt.show()
x = df[list(df.columns)[1:]]
y = df[list(df.columns)[0]]
X_train, X_test, y_train, y_test = train_test_split(x,y)

model = LinearRegression()
model.fit(X_train,y_train)
predict = model.predict(X_test)

y_test = np.array(y_test)
f = np.zeros(np.shape(y_test))
f[predict.round() == np.array(y_test)] = 1
print(f)
print('')
sc = cross_val_score(model,x,y,cv=5)
print(sc,sc.mean())