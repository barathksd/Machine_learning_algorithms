# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:23:21 2019

@author: Lenovo
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.cm as cm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


digits = datasets.load_digits()
counter = 1
for i in range(0, 4):
    plt.figure(counter, figsize=(3, 3))
    plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    counter += 1
plt.show()

x,y = digits.data,digits.target
x = x/8-1

x_tr,x_tst,y_tr,y_tst = train_test_split(x,y)

pipeline = Pipeline([
        ('clf',SVC(kernel='rbf', gamma=0.01, C=100))
    ])
print (x_tr.shape)
parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30),
    }
grid_search = GridSearchCV(pipeline, parameters, n_jobs=3,
verbose=1, scoring='accuracy')

grid_search.fit(x_tr,y_tr)
print ('Best score: %0.3f' % grid_search.best_score_)
print ('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(x_tst)
print (classification_report(y_tst, predictions))








