# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:04:16 2021

@author: 81807
"""

import numpy as np
import cv2
import os
import sys
import pandas as pd
import pymongo 
from datetime import datetime
import datetime as dtlib
import time
import json

def mongoconnect(dburl,dbname,delete=False):
    client = pymongo.MongoClient(dburl)
    
    db = client[dbname]
    collnames = db.list_collection_names()
    if delete==False:
        return db
    
    for i in collnames:
        db.drop_collection(i)
    db = client[dbname]
    return db

uri = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false'
videodb = mongoconnect(uri,'videoDB')

n = 6
p = {i+1:chr(65+i) for i in range(n)}
cams = ['cam'+str(i) for i in range(3,5)]

roomx = (-305,637)
roomy = (-897,1391)

def getrxy(roomx,roomy,n=1):
    l = []
    for _ in range(n):
        x = -1
        y = -1
        while (x<0 and y<0):
            x = np.random.randint(roomx[0],roomx[1])
            y = np.random.randint(roomy[0],roomy[1])
        l.append([x,y])
    return l
    
def create(roomx,roomy,n,cams,p):
    l = np.array(getrxy(roomx,roomy,n))
    maxv = 5

    
    try:
        for i in range(5):
            pos = []
            t = datetime.now()
            for _ in range(60):
                v = np.random.randint(-maxv,maxv,n*2).reshape(n,2)
                l = l + v
                c = np.int32(np.random.rand(n) + 0.5)
                cs = [cams[i] for i in c]
                t = t + dtlib.timedelta(seconds=1)
                pos += [{'pid':i+1,'pName':p[i+1],'t':t,'rx':l[i][0],'ry':l[i][1],'camNo':cs[i]} for i in range(n)]
            videodb['xydata'].insert_many(pos)
                
                
            videodb['xydata'].delete_many({'rt':{'$lt':datetime.now()-dtlib.timedelta(hours=1)}})
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        
        
        
        
        