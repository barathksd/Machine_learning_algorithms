# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 22:53:19 2018

@author: Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file='C:\\Users\\Lenovo\\Desktop\\data\\horse_train.txt'

def loadSimpData():
    datMat = np.matrix([[ 1. , 2.1],
                 [ 2. , 1.1],
                 [ 1.3, 1. ],
                 [ 1. , 1. ],
                 [ 2. , 1. ]])
    classLabels = np.mat([1.0, 1.0, -1.0, -1.0, 1.0]).T
    return datMat,classLabels

def create(file):
    f=pd.read_csv(file,delimiter="\t")
    f=np.array(f)
    lab=f[:,21]
    f=f[:,:21]
    return f,lab

def stump(f,dim,th,si):
    res=np.ones((np.shape(f)[0],1))
    if th<0:
        res[f[:,dim]<=th]=-1
    else:
        res[f[:,dim]>th]=-1
    return res

def buildStump(dataArr,classLabels,D):
    dataMatrix = dataArr; labelMat = classLabels
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = 1000
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stump(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print(weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def classify(d,l,n,w):
    
    return 0

#f,lab=create(file)
#w=np.mat(np.ones((298,1))/298)
#bestStump(f,lab,w)
datamat,classLabels=loadSimpData()
w=np.mat(np.ones((5,1))/5)
buildStump(datamat,classLabels,w)
