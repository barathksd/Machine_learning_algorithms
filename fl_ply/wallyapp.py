# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:02:17 2021

@author: 81807
"""

from __future__ import print_function
from __init__ import app, df, names, bgimg, base
from flask import render_template, request, Response, url_for, jsonify, flash, redirect, send_file, session, make_response
import numpy as np
import cv2
import pandas as pd
import os
import sys
import datetime as dtlib
from datetime import datetime
import json
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px
import plotly
from search import SearchForm
import traceback

c = 5
#Debug(app)
videodb = None
dp = 'test'
st = '10:00'

def getdata(df,name,sdate,stime,etime):
    
    t1 = f'{sdate}T{stime}:00.000Z'
    if etime != stime:
        t2 = f'{sdate}T{etime}:00.000Z'
    else:
        t2 = f'{sdate}T{etime}:59.000Z'
        
    if name != 'all':
        return df[(df['pName'] == name) & (df['t']>t1) & (df['t']<t2)] 
    else:
        return df[(df['t']>t1) & (df['t']<t2)] 

def getgraph(data):
    fig = None
    try:
        fig = px.scatter(data,x='ry', y='rx',color='pName',labels={'x':'ry','y':'rx'},title='map',hover_data=['pName','t'])

        fig.update_xaxes(range=[-860, 1400])
        fig.update_yaxes(range=[647, -315])
        fig.update_layout(
                        images= [dict(
                            source='data:image/png;base64,{}'.format(bgimg),
                            xref="x", yref="y",
                            x=-890, y=-315,
                            sizex=2260, sizey=962,
                            sizing="stretch",
                            layer="below")])
        #plot(fig)
        fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        with open(base+'ex.txt','w') as txtfile:
            txtfile.write(str(e))       
    return fig
    

@app.route("/live",methods=['GET','POST'])
def live():
    form = SearchForm()
    fig = None
    return render_template('live.html',form = form, fig = fig)

@app.route("/livedata",methods=['GET','POST'])
def livedata():
    global st
    posts = None
    sdate = str(dtlib.date(2021,5,7))
    m = str(int(st[-2:])+2)
    if len(m) == 1:
        m = '0' + m 
    et = st[:3] + m
    data = getdata(df, 'all', sdate, st, et)
    
    with open(base+'livedata.txt','w') as txtfile:
        txtfile.write(st+' '+et+' '+str(datetime.now()) + ' '+str(len(data))) 
    fig = getgraph(data)
    st = et
    return fig

@app.route("/",methods=['GET','POST'])
def start():
    form = SearchForm()
    data = None
    error = None
    posts = None
    fig = None
    if request.method == 'POST':
        base = 'C:\\Users\\Lenovo\\Desktop\\data\\code\\fl_ply\\'
        if form.validate_on_submit():
            name = form.name.data
            data = request.form

            error = None
            sdate = str(dtlib.date(2021,5,7))
            stime = form.stime.data
            etime = form.etime.data
            df2 = getdata(df,name,sdate,stime,etime)

            fig = getgraph(df2)
        
        else:
            error = 'Name not found'
            
    return render_template('index.html',posts=posts,form = form,data = data,error=error,fig=fig)

@app.route("/increment",methods=['GET','POST'])
def increment():
    global c
    if request.method == 'POST':
        c = c+5
        return redirect(url_for('home',c=c))
    return '<html></html>'

@app.route("/names",methods=['GET','POST'])
def getnames():
    namesdict = [{'name':i} for i in names]
    return jsonify(namesdict)

@app.route("/sendimage",methods=['GET','POST'])
def sendimg():
    base = 'c:/users/lenovo/desktop'
    with open(base+'img.txt','w') as txtfile:
        txtfile.write('in send img')
    img = Image.fromarray(np.uint8(np.zeros((128,128))))
    imgio = BytesIO()
    img.save(imgio,'JPEG')
    imgio.seek(0)
    return send_file(imgio, mimetype='image/jpeg')


    
    