# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:40:16 2021

@author: Lenovo
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.optimize import curve_fit

df = sns.load_dataset("flights")
sns.lineplot(x=np.arange(df.shape[0]),y=df['passengers'])
plt.show()

def fn(x,a,b,c,d,e):
    return a + b*x + c*x*np.sin(e*x) + d*x*np.cos(e*x) 

xdata = df.index.to_numpy()
popt, pcov = curve_fit(fn, xdata, df['passengers'].values)

s = sns.lineplot(x=xdata, y=fn(xdata, *popt))
sns.lineplot(x=np.arange(df.shape[0]),y=df['passengers'])
plt.show()

