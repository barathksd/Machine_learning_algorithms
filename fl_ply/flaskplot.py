# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:27:45 2021

@author: Lenovo
"""
from flask import Flask,render_template,Response,url_for
#import cv2
import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import datetime as dtlib
from datetime import datetime
import base64

base = 'c:/users/lenovo/desktop/data/code/fl_ply/'
imgfile = base + '背景図.jpg'
bgimg = base64.b64encode(open(imgfile, 'rb').read())
bgimg = bgimg.decode()
df = pd.read_csv(base+'data4_14_16.csv')
df['personID'] = df['personID'].apply(str)
df['acttime'] = df['UTC'].apply(lambda x:datetime.fromtimestamp(x)+dtlib.timedelta(hours=9))+df['UTCMs'].apply(lambda x:dtlib.timedelta(milliseconds=x))
df = df.sort_values(by=['UTC','UTCMs'])
bsize = 1*60
df['utctime'] = pd.Series(np.int32(df['UTC']/bsize)*bsize).apply(lambda x:datetime.fromtimestamp(x)+dtlib.timedelta(hours=9))
df['utctime'] = pd.to_datetime(df['utctime']).apply(str)
app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def start():

    fig = px.scatter(df,x='ry', y='rx',color='personID',animation_frame='utctime',labels={'x':'ry','y':'rx'},title='map',hover_data=['personID','personName','source','acttime'],width=60)
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
    data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',data=data)
    

if __name__ == '__main__':
    app.run(debug=True)