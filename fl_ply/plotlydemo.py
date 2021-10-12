# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:15:08 2021

@author: Lenovo
"""

import plotly
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import json
import cv2
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()

def barplot():
    d = np.random.randint(0,100,(10,4))
    df = pd.DataFrame(d,columns=['a','b','c','d'])
    df.iplot()
    
def lineplot():
    #df = px.data.stocks()
    #fig = px.scatter(df,x='date',y='GOOG',labels={'x':'Date','y':'Price'})
    #fig.show()
    import cv2
    import pandas as pd
    
    img = cv2.imread('c:/users/lenovo/desktop/5-Quora.jpg')
    img = cv2.resize(img,(2000,900),cv2.INTER_AREA)
    #img.resize((942,2288))
    df = pd.read_csv('c:/users/lenovo/desktop/data4_14_16.csv')
    fig = go.Figure()
    fig.add_trace(
        go.Image(z=img,opacity=0.5,x0=-800,y0=-305)
    )
    #px.scatter(df,x='ry', y='rx',color='personID',labels={'x':'ry','y':'rx'},title='map')
    # Add trace
    fig.add_trace(
        go.Scatter(x=df.ry, y=df.rx,mode='markers',marker=dict(color=df.personID),text=df.personID)
    )

    # Set templates
    fig.update_layout(template="plotly_white")
    
    fig.show()
    return fig
    
def create_plot():

    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe

    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def tdmap():
    import datetime as dtlib
    from datetime import datetime
    import base64
    base = 'c:/users/lenovo/desktop/data/code/fl_ply/'
    imgfile = base + '背景図.jpg'
    bgimg = base64.b64encode(open(imgfile, 'rb').read())
    bgimg = bgimg.decode()
    df = pd.read_csv(base + 'data4_14_16.csv')
    df['personID'] = df['personID'].apply(str)
    df['acttime'] = df['UTC'].apply(lambda x:datetime.fromtimestamp(x)+dtlib.timedelta(hours=9))+df['UTCMs'].apply(lambda x:dtlib.timedelta(milliseconds=x))
    df.sort_values(by=['UTC','UTCMs'])
    bsize = 1*60
    df['utctime'] = pd.Series(np.int32(df['UTC']/bsize)*bsize).apply(lambda x:datetime.fromtimestamp(x)+dtlib.timedelta(hours=9))
    df['utctime'] = pd.to_datetime(df['utctime']).apply(str)
    fig = px.scatter(df,x='ry', y='rx',color='personID',labels={'x':'ry','y':'rx'},title='map',hover_data=['personID','personName','source','acttime'])
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
    plot(fig)
    '''
    fig = go.Figure()
    fig.add_trace(
        go.Image(z=img,opacity=0.5,x0=-800,y0=-305)
    )
    # Add trace
    fig.add_trace(
        go.Scatter(x=df.ry, y=df.rx,mode='markers',marker=dict(color=df.personID,showscale=True),text=df.personName,showlegend=True)
    )
    
    # Set templates
    fig.update_layout(template="plotly_white")
    plot(fig)
    '''

def tdmap2():
    import datetime as dtlib
    from datetime import datetime
    import base64
    base = 'c:/users/lenovo/desktop/data/code/fl_ply/'
    imgfile = base + '背景図.jpg'
    bgimg = base64.b64encode(open(imgfile, 'rb').read())
    bgimg = bgimg.decode()
    df = pd.read_csv(base + 'xylog.csv')
    df = df[['camName','pName','rx','ry','t']]
    
    sdate = str(dtlib.date(2021,5,7))
    stime = '10:10'
    etime = '10:12'
    t1 = f'{sdate}T{stime}:00.000Z'
    t2 = f'{sdate}T{etime}:00.000Z'
    df2 = df[(df['t']>t1) & (df['t']<t2)]
    print(df2)
    fig = px.scatter(df2,x='ry', y='rx',color='pName',animation_frame='t',labels={'x':'ry','y':'rx'},title='map',hover_data=['pName','t'])
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
    plot(fig)