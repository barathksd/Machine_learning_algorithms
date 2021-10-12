# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:37:05 2021

@author: 81807
"""

import numpy as np
from common_tools import mongoconnect
from datetime import datetime
import datetime as dtlib


uri = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false'
videodb = mongoconnect(uri,'videoDB')
videodb['xylog'].delete_many({})

st = datetime(2021,5,7,10)
xlim = (-305,627)
ylim = (-890,1391)

def genxy(length=1,xlim=(-305,627),ylim=(-890,1391)):
    if length == 0:
        return None,None
    y = np.random.randint(ylim[0],ylim[1],length)
    x = np.random.randint(xlim[0],xlim[1],length)
    c = (y<0)*(x<0)
    x[c] = np.random.randint(0,xlim[1],np.sum(c))
    return x,y

def move(x,y,t,ct,length=1,xlim=(-305,627),ylim=(-890,1391)):
    c0 = t == ct
    s0 = np.sum(c0)
    if s0>0:
        x1 = x[c0]
        y1 = y[c0]
        x1 += np.random.randint(-10,10,s0)
        y1 += np.random.randint(-10,10,s0)
        
        c = (y1<0)*(x1<0) + (x1<xlim[0]) + (x1>xlim[1]) + (y1<ylim[0]) + (y1>ylim[1])
        t1 = t[c0]
        t1[~c] += dtlib.timedelta(seconds=1)
        t[c0] = t1
        s = np.sum(c)
        if s>0:
            t1 = t[c0]
            t1[c] += dtlib.timedelta(minutes=10)
            t[c0] = t1
            xtemp,ytemp = genxy(s,xlim,ylim)
            x1[c] = xtemp
            y1[c] = ytemp
        x[c0] = x1
        y[c0] = y1
    return x,y,t

def getcam(x,y,xlim=(-305,627),ylim=(-890,1391)):
    a = int((y-ylim[0])/((ylim[1]-ylim[0])/4))
    camName = 'Cam1'
    if y<0:
        m = xlim[1]/2  
        if x<m and x>=0:
            camName = 'Cam' + str(a*2+1)
        elif x>=m and x<=xlim[1]:
            camName = 'Cam' + str(a*2+2)
        else:
            camName = 'Cam0'
    else:
        m = (xlim[1]-xlim[0])/2
        if (x - xlim[0])<m and x>=xlim[0]:
            camName = 'Cam' + str(a*2+1)
        elif (x - xlim[0])>=m and x<=xlim[1]:
            camName = 'Cam' + str(a*2+2)
        else:
            camName = 'Cam0'
    return camName
        

names = [i['name'] for i in list(videodb['people'].find())]
length = len(names)
x,y = genxy(length,xlim,ylim)
t = np.array([st for i in range(length)])
data = {}

for i in range(3600):
    ct = st + dtlib.timedelta(seconds=i)
    l = [{'pName':names[j],'rx':int(x[j]),'ry':int(y[j]),'t':ct,'camName':getcam(x[j],y[j])} for j in np.where(t==ct)[0]]
    data[i] = l
    if len(l)>1:
        videodb['xylog'].insert_many(l)
    elif len(l) == 1:
        videodb['xylog'].insert_one(l[0])
    x,y,t = move(x,y,t,ct,length)




