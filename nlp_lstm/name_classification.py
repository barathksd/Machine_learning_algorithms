# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:56:36 2021

@author: Lenovo
"""

import os
import unicodedata
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=3)

#OHE = OneHotEncoder(categories=[np.array([i for i in all_letters])],handle_unknown='ignore')
#OHE.fit([[i] for i in all_letters])

file_path = 'C:/users/lenovo/desktop/data/code/datasets/nlp_data/names'
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
MLB = MultiLabelBinarizer(classes=list(all_letters))
MLB.fit(list(all_letters))

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def turnToTensor(name,mlb):
    return torch.FloatTensor(mlb.transform(list(name)))

def load_data(file_path):
    names = {}
    for i in os.listdir(file_path):
        with open(os.path.join(file_path,i),'rb') as f:
            n = f.read().strip().decode('utf-8').split('\n')
        n = list(map(unicodeToAscii,n))
        names[i.split('.')[0]] = list(set(n))
        
    return names

names = load_data(file_path)
categories_dict = {index:i for index,i in enumerate(names.keys())}
all_categories = list(names.keys())
MLB2 = MultiLabelBinarizer(classes=all_categories)
MLB2.fit([[i] for i in all_categories])

def train_test(names):
    train,test = [],[]
    
    for key,val in names.items():
        
        a = np.array(val)
        np.random.shuffle(a)
        train.extend([(i,key) for i in a[:min(int(0.8*len(a)),1000)]])
        test.extend([(i,key) for i in a[min(int(0.8*len(a)),1000):]])
        
    train = np.array(train)
    np.random.shuffle(train)
    test = np.array(test)
    return train,test

training_data,test_data = train_test(names)

class LSTMnamer(nn.Module):
    
    def __init__(self,in_size,hidden_size,out_size):
        super(LSTMnamer,self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(in_size,hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.out = nn.LogSoftmax(dim=1)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))

    def forward(self,x):
        y,self.hidden_cell = self.lstm(x.view(len(x),1,-1))
        y = self.linear(y.view(len(x),-1))
        y = self.out(y)
        return y[-1]
    
    def init_hidden(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))
        

n_hidden = 64
n_categories = len(names.keys())
model = LSTMnamer(n_letters,n_hidden,n_categories)
opt = {}
opt['loss'] = nn.NLLLoss()  # negative log likelihood loss
opt['optimizer'] = torch.optim.Adam(model.parameters(), lr=0.002)
#opt['optimizer'] = torch.optim.SGD(model.parameters(), lr=0.1)
opt['epochs'] = 20

def train(opt,model,training_data):

    loss_function = opt['loss']
    optimizer = opt['optimizer']
    epochs = opt['epochs']
    for epoch in range(epochs):
        for name,category in training_data:
            optimizer.zero_grad()
            model.init_hidden()
            name = turnToTensor(name,MLB)
            category = torch.tensor([all_categories.index(category)],dtype=torch.long)
            
            out = model(name)
            l = loss_function(out.view(1,-1), category)
            #print(l,out.view(1,-1), category)
            l.backward()
            optimizer.step()

        if epoch%1 == 0:
            print(f'epoch: {epoch} loss: {l.item():10.8f}')


train(opt,model,training_data)

def get_output(out):
    return out.topk(1).indices.item()

def test(model,test_data):
    ypred = []
    ytrue = []
    for name,category in test_data:
        with torch.no_grad():
            name = turnToTensor(name, MLB)
            out = get_output(model(name))
            ypred.append(out)
            ytrue.append(all_categories.index(category))
    
    confusion_matrix(ytrue,ypred)
    cm = confusion_matrix(ytrue,ypred)
    cm = (cm/(np.sum(cm,axis=1)+0.01).reshape(-1,1))
    return ytrue,ypred,cm

ytrue,ypred,cm = test(model,test_data)

def plot_results(cm):
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    
    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # sphinx_gallery_thumbnail_number = 2
    plt.show()

    # seaborn plot
    df = pd.DataFrame(np.round(cm,decimals=2), index = [i for i in all_categories],
                      columns = [i for i in all_categories])
    plt.figure(figsize = (10,7))
    sns.heatmap(df, annot=True)
    plt.show()

plot_results(cm)