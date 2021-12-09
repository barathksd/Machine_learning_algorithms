# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:50:24 2021

@author: Lenovo
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
columns = map(lambda x: (x.split(' (')[0]),iris['feature_names'])
df = pd.DataFrame(iris['data'],columns=columns)
df['Target'] = iris.target
# histogram

y,x = np.histogram(df[df['Target']==2]['sepal length'])
x = [(x[i]+x[i+1])/2 for i in range(len(x)-1)]
sns.barplot(x=x,y=y)
plt.show()
sns.kdeplot(df[df['Target']==2]['sepal length'], color='b', shade=True, Label='Iris_Virginica')
plt.show()

iris_setosa = df[df['Target']==0]
iris_virginica = df[df['Target']==2]  
# Plotting the KDE Plot
ax = sns.kdeplot(x=iris_setosa['sepal length'], 
            y=iris_setosa['sepal width'],
            color='r', shade=True, legend=True,
            cmap="Reds")
sns.kdeplot(x=iris_virginica['sepal length'], 
            y=iris_virginica['sepal width'], color='b',
            shade=True, legend=True,
            cmap="Blues",ax=ax)
plt.show()
