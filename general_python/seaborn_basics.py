# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:10:21 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse

def run():
    for i in range(5):
        with open('C:\\Users\\Lenovo\\Desktop\\run'+str(i)+'.txt','w') as txtfile:
            txtfile.write(str(0)+str(1)+' '+str(os.getpid()))
        time.sleep(1)

    
def sinplot():
    x = np.linspace(0,10,101)
    for i in range(5):
        plt.plot(x,np.sin(x+i)*(x+i))

    sns.set_style('whitegrid')
    sns.set_palette(sns.color_palette('BrBG',5))
    sns.despine()
    plt.show()

#sinplot()

def barandjoint():
    n = 250
    sns.distplot(np.random.randn(n),bins=20,color='#F75394')
    
    df = pd.DataFrame(np.concatenate((np.random.randn(n).reshape(-1,1),np.random.rand(n).reshape(-1,1)),1))
    df.set_axis(['x','y'],axis=1,inplace=True)
    df['z'] = np.arange(n)/100
    df['sp'] = np.concatenate((np.array(['a1' for i in range (100)]),np.array(['a2' for i in range (100)]),np.array(['a3' for i in range (50)])))
    #df = df.drop('x',axis=1)
    sns.jointplot(x='x',y='z',data=df,color='green',kind='scatter')
    sns.pairplot(df,hue='sp',diag_kind="kde",kind="scatter",palette="husl")
    plt.show()

#barandjoint()

def boxandswarm():
    df = sns.load_dataset('iris')
    sns.stripplot(x='species',y='petal_length',data=df,jitter=True)
    plt.show()
    sns.swarmplot(x='species',y='petal_length',data=df)
    plt.show()
    sns.violinplot(x="species", y="petal_length", data=df)
    plt.show()
    df = sns.load_dataset('tips')
    sns.violinplot(x='day',y='total_bill',hue='sex',data=df)
    plt.show()
    sns.barplot(x='day',y='total_bill',hue='sex',data=df)
    plt.show()

#boxandswarm()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=1)
    parser.add_argument('--model-dir', type=str, default=2)
    parser.add_argument('--train', type=str, default=3)

    args = parser.parse_args()
    print(args)















