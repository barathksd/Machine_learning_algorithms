# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:21:15 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

data = pd.read_csv('C:/Users/Lenovo/Desktop/data/dt_rf_ads.csv',header=None,low_memory=False)
ex = set(data.columns.values)
rs = data[len(data.columns.values)-1]

ex.remove(len(data.columns.values)-1)
y = [1 if e == 'ad.' else 0 for e in rs]
X = data[list(ex)]
X.replace(to_replace=' *\?', value=-100, regex=True, inplace=True )
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline = Pipeline([
('clf', DecisionTreeClassifier(criterion='entropy'))
])
parameters = {
        
'clf__max_depth': (50, 150, 250),
'clf__min_samples_split': (1.0, 2, 3),
'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters,
verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)
print ('Best score: %0.3f' % grid_search.best_score_)
print ('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print (classification_report(y_test, predictions))





