# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:25:29 2021

@author: 81807
"""

import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dtlib
import pandas as pd
import time
import json


def tfinit():
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution() 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
def nightsleep(wtime,stime,etime):
    #夜の9時過ぎたら朝まで一時停止する
    dt = datetime.now(dtlib.timezone(dtlib.timedelta(hours=+9),'JST'))
    year,month,day = dt.year,dt.month,dt.day
    if dt.hour > etime:
        dt2 = datetime(year,month,day+1,stime,15,0)
        time.sleep((dt2-datetime.now()).seconds)
    
    #或いは、5分（300秒）待つ
    else:
        time.sleep(wtime)


def utc():
    return time.mktime(datetime.now().timetuple())

def datefromutc(utc):
    return datetime.fromtimestamp(utc)

def bytetostr(byte):
    return byte.decode("utf-8") 

def strtobyte(string):
    return str.encode(string)


def mongoconnect(dburl,dbname,delete=False):
    import pymongo
    client = pymongo.MongoClient(dburl)
    
    db = client[dbname]
    collnames = db.list_collection_names()
    if delete==False:
        return db
    
    for i in collnames:
        db.drop_collection(i)
    db = client[dbname]
    return db


def execsql(sqlcmd,database='activestreamhcx',cmd='selectall'):
    import pymysql
    connection = pymysql.connect(host='10.200.0.52',
                                     port=3306,
                                     user='docker',
                                     password='docker',
                                     database=database,
                                     cursorclass=pymysql.cursors.DictCursor)

    with connection.cursor() as cursor:
        #sqlcmd = "SELECT * from aicamera_log"
        result = None
        cursor.execute(sqlcmd)
        if cmd == 'select_1':
            result = cursor.fetchone()
            
        elif cmd == 'selectall':
            result = cursor.fetchall()
            
    #connection.commit()
    #connection.close()
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    return result

def write_csv(wpath,data):
    import csv
    with open(wpath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def write_json(wpath,data):
    with open(wpath, 'w') as outfile:
        json.dump(data, outfile)
    
    
def read_json(rpath):
    with open(rpath) as json_file:
        data = json.load(json_file)
        return data

def append_json(rpath,name,val,wpath=None):
    if wpath == None:
        wpath = rpath
    data = read_json(rpath)
    data[name] = val
    write_json(wpath,data)
    return data

def load_csv(file):
    return pd.read_csv(file,delimiter=',',names=['f','x','y','w','h']).drop([0],axis=0)

def save_csv(file,df):
    df.to_csv(file,index=False)
    
def curvefit(y,doplot = False,method = None):
    from sklearn.linear_model import LinearRegression
    #y = [i[0] for i in df[(df[1]==319) & (df[0]>700)][5]]
    
    if method is None:
        l = np.arange(1,len(y))
        return np.sum(l*[y[i+1]-y[i] for i in range(len(y)-1)])/np.sum(l)
    
    elif method == 'sine':
        if len(y)<=3:
            print('cant perform sine fit')
            return 
        vy = np.array([y[i+1]-y[i] for i in range(len(y)-1)])
        t = np.arange(vy.shape[0])
        gmean = np.mean(vy)
        gamp = (vy[np.argsort(vy)[-5]] - vy[np.argsort(vy)[5]])/2
        gphase = 0

        merror = 512
        p = []
        for gfreq in np.arange(0,1.5,0.03):
            optimize_func = lambda x: (x[0]*np.sin(x[1]*t+x[2]) + x[3] - vy)**2
            est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [gamp, gfreq, gphase, gmean])[0]

            vp = est_amp*np.sin(est_freq*t+est_phase) + est_mean
            er = np.sum(np.square(vp-vy))
            if er < merror:
                merror = er
                p = (est_amp, est_freq, est_phase, est_mean)

        vp = p[0]*np.sin(p[1]*t+p[2]) + p[3]

        if doplot == True:
            plt.plot(t,vy)
            plt.plot(t,vp)

        t = len(y)-1
        return p[0]*np.sin(p[1]*t+p[2]) + p[3]
    
    elif method == 'acc':
        ay = np.array([y[i+2]+y[i]-2*y[i+1] for i in range(len(y)-2)])
        t = np.arange(ay.shape[0]).reshape(-1,1)
        reg = LinearRegression()
        reg.fit(t,ay.reshape(-1,1))
        
        return reg.predict(np.array([len(ay)]).reshape(-1,1))[0]
    
def angle(pos):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(pos[:,0].reshape(-1,1), pos[:,1].reshape(-1,1))
    return float(reg.coef_)

def pair_angle(pos1,pos2):
    d1 = pos1[1] - pos1[0]
    d2 = pos2[1] - pos2[0]
    dot = np.dot((d1,d2))/(np.sum(np.square(d1)) + np.sum(np.square(d2)))
    angle = np.arccos(dot)*180/np.pi  # angle in degrees
    
    return dot,angle

def eudist(pos1,pos2):
    (x1,y1) = pos1
    (x2,y2) = pos2
    return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    