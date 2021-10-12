# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:06:10 2021

@author: Lenovo
"""

from flask import Flask
import base64
import pandas as pd
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = '082d96e286cb2cd8d64fc70462466f7d'

base = 'c:/users/lenovo/desktop/data/code/fl_ply/'
imgfile = base + '背景図.jpg'
bgimg = base64.b64encode(open(imgfile, 'rb').read())
bgimg = bgimg.decode()

file = base + 'xylog.csv'
df = pd.read_csv(file)
df = df[['camName','pName','rx','ry','t']]
names = list(map(str.lower,df['pName'].unique()))

def is_validname(name):
    name = name.lower()
    if name == 'all':
        return True
    return name in names

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

def write_json(wpath,data):
    with open(wpath, 'w') as outfile:
        json.dump(data, outfile)
