# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 23:01:34 2018

@author: Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file="C://Users//Lenovo//Desktop//data//reg0.txt"
def create(file):
    f=np.mat(pd.read_csv(file,delimiter="\t",usecols=[i for i in range(2)])) #298x21
    lab=np.mat(pd.read_csv(file,delimiter="\t",usecols=[2]))
  
    return f,lab

def lwlr(xs,x,y,k):
    m=np.shape(x)[0]
    w=np.mat(np.eye((m)))
    for i in range(m):
        diff=xs-x[i]
        w[i,i]=np.exp(diff*diff.T/(-2.0*k**2))
    xTx = x.T * (w * x)    
    if np.linalg.det(xTx)!=0:
        ws = xTx.I * (x.T * (w * y))
        return (xs*ws)[0]
    return -1

def lwlrTest(x,y,k):
    ys=[]
    m=np.shape(x)[0]
    for i in range(m):
        yexp=lwlr(x[i],x,y,k)
        ys.append(yexp[0,0])
        
    return np.mat(ys).T

def ridge(x,y,l):
    xTxnew = (x.T*x)+np.eye(np.shape(x)[1])*l   #2x2
    if np.linalg.det(xTxnew)!=0:
        wt = xTxnew.I*(x.T*y)
        return wt
    return 0

def ridgeTest(x,y,l):
    wt=ridge(x,y,l)
    return x*wt,wt
    
    
x,y=create(file)
k=0.01
#ys=lwlrTest(x,y,k)

# =============================================================================
# 
# fig=plt.figure()
# ax1=fig.add_subplot(111)
# x1=x[:,1]
# 
# ax1.plot(x1,y[:,0],'r*',x1,ys[:,0],'b*')
# 
# =============================================================================


