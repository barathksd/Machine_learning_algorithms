# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 22:35:33 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

file='C:/Users/Lenovo/Desktop/data/SMSSpamCollection'

f=pd.read_csv(file, delimiter='\t',header=None)
print(f.head())

X_train_raw, X_test_raw, y_train, y_test = train_test_split(f[1],f[0])

vect = TfidfVectorizer()
X_tr = vect.fit_transform(X_train_raw)
X_ts = vect.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_tr, y_train)
predictions = classifier.predict(X_ts)
for i in range(50):
    print(predictions[i],y_test.values[i],X_test_raw.values[i])
    print('')

confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
    