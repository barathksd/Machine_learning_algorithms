# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:10:47 2019

@author: Lenovo
"""

import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt

data = np.mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

def sim(a,b):
   
    
    l1 = 1.0/(1.0+la.norm(a-b))
    l2 = (1+np.corrcoef(a,b,rowvar = 0)[0][1])/2
    l3 = (a.T*b)/(la.norm(a)*la.norm(b))
    l3 = 0.5+0.5*l3
    
    return (l1+2*l2+l3)/4


def est(data,user,item,sim,xnew):
    simtot = 0
    siml = 0
    rt = 0
    for j in range(data.shape[1]):
        if data[user,j]!=0:
            z = np.nonzero(np.logical_and(data[:,item].A>0,data[:,j].A>0))[0]
            if len(z) == 0:
                siml = 0
            else:
                siml = sim(xnew[:,item],xnew[:,j])
                simtot += siml
            rt += siml*data[user,j]                   
    if simtot == 0: 
        return 0
    return rt/simtot

def rec(data,user,sim):
    U,sigma,VT=la.svd(data)
    sigma = np.eye(4)*sigma[:4]
    xnew = data.T*U[:,:4] * sigma
    unratedItems = np.nonzero(data[user,:].A==0)[1]
    if len(unratedItems) == 0: return 'you rated everything'
    Score = []
    for item in unratedItems:
        estScore = est(data,user,item,sim,xnew.T)
        Score.append((item,estScore))
    print('Score',Score)
    


file = 'C:/Users/Lenovo/Desktop/data/alps.jpg'
img = cv2.imread(file)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img)

u,sig,vt = la.svd(img)
img1 = u[:,:50]*(np.mat(np.eye(50)*sig[:50]))*vt[:50,:]


cv2.imwrite('C:/Users/Lenovo/Desktop/data/svd50_alps.jpg',img1)

print(rec(data,6,sim))





