# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:36:32 2021

@author: Lenovo
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from io import StringIO, BytesIO
from openpyxl import load_workbook
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns

def get_df(xlfile, sheet_name='Sheet1'):
    wb = load_workbook(filename = xlfile)
    ws = wb[sheet_name]
    
    data = ws.values
    cols = next(data)[1:]
    data = list(data)
    idx = [r[0] for r in data]
    data = (islice(r, 1, None) for r in data)
    df = pd.DataFrame(data, index=None, columns=cols)  
    df['date'] = idx
    return df

#df = get_df('C:/users/lenovo/downloads/factory1.xlsx','Sheet1')

df = get_df('C:/users/lenovo/downloads/factory.xlsx','Sheet1')
df = df[['AI200372_洗浄機蒸気流量', 'AI200223_蒸気流量', 'PI300196_ｶﾞｽ流量', 'PI300197_ｶﾞｽ流量','PI300200', 'PI300240', 'PI300243', 'date']]

d = {'A223':'工程-4','A372':'清掃装置','P196':'Boiler','P197':'工程-1','P200':'厨房用','P240':'食堂空調','P243':'空調-9'}
l = list(df.columns[:-1])
for s in l:
    sc = s
    s = s.split('_')
    k = s[0][0]+s[0][-3:]
    
    d[k] = d[k] +('_'+s[1] if len(s)>1 else '')
    df.rename(columns={sc:k},inplace=True)

np.set_printoptions(precision=3,suppress=True)

df['month'] = df['date'].apply(lambda x: str(x.year)+'/'+str(x.month))
df['day'] = df['date'].apply(lambda x: str(x.year)+'/'+str(x.month)+'/'+str(x.date().day))
df['hour'] = df['date'].apply(lambda x: str(x.year)+'/'+str(x.month)+'/'+str(x.day_name())+'/'+str(x.hour))
df['weekday'] = df['date'].apply(lambda x: str(x.year)+'/'+str(x.month)+'/'+str(x.day_name()))
df['weekofyear'] = df['date'].apply(lambda x: str(x.year)+'/'+str(x.month)+'/'+str(x.weekofyear))
dfc = df.copy()

df = dfc.copy()
limit = 600

#df = df[df['month'].between('2020/3','2020/6')]
#df = df[(df['month'] == '2019/5') | (df['month'] == '2020/5') | (df['month'] == '2021/5')]
#df = df[(df['month'] == '2019/6') | (df['month'] == '2020/6')]

def plotall(df,folder,limit=1200):
    for i in range(len(d)):
        s = sns.barplot(data=df.head(limit),y=df.columns[i],x='date')
        plt.savefig('C:/users/lenovo/pictures/plots/{}/all/{}.jpg'.format(folder,df.columns[i]))
        plt.show()
        
    for group in ['month','day','hour','weekday','weekofyear']:
        df2 = df.groupby(by=group).sum()
        df2[group] = df2.index
        df2['year'] = df2[group].apply(lambda x: x.split('/')[0])
        m = ''
        if df2.shape[0]>limit:
            df2 = df2.head(limit)
            m = '_r'
            
        for i in range(len(d)):
            s = sns.barplot(x=df2[group],y=df2[df.columns[i]],hue=df2['year'],palette=sns.color_palette())
            plt.savefig('C:/users/lenovo/pictures/plots/{}/{}/{}.jpg'.format(folder,group,df.columns[i]+m))
            plt.show()













