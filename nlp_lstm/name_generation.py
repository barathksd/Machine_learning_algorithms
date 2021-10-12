# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:28:21 2021

@author: Lenovo
"""

import os
import unicodedata
import string
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

np.set_printoptions(precision=3)

#OHE = OneHotEncoder(categories=[np.array([i for i in all_letters])],handle_unknown='ignore')
#OHE.fit([[i] for i in all_letters])

file_path = 'C:/users/lenovo/desktop/data/code/datasets/nlp_data/names'
all_letters = list(string.ascii_letters[:26]+" .,;'")+['<EOS>','<S>']
n_letters = len(all_letters)
MLB = MultiLabelBinarizer(classes=all_letters)
MLB.fit(all_letters)

def unicodeToAscii(s):
    s = s.strip()
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
            n = f.read().strip().decode('utf-8').lower().split('\n')
        n = list(map(unicodeToAscii,n))
        names[i.split('.')[0]] = list(set(n))
        
    return names

    
def get_data(names,max_val=1000):
    a = {}
    for category,ns in names.items():
        ns = np.array(ns)
        np.random.shuffle(ns)
        ns = ns[:max_val]
        l = []
        for name in ns:
            l.append(([['<S>'],name[0]],name[1]))
            if len(name)>=2:
                l.extend([(name[:i],name[i]) for i in range(2,len(name))])
            l.append((name,'<EOS>'))
        a[category] = l
    return a

names = load_data(file_path)
categories_dict = {index:i for index,i in enumerate(names.keys())}
all_categories = list(names.keys())
n_categories = len(all_categories)
MLB2 = MultiLabelBinarizer(classes=all_categories)
MLB2.fit([[i] for i in all_categories])
training_data = get_data(names)
#list(map(lambda x: (x[0],len(x[1])),training_data.items()))

def get_sample(training_data):
    c = random.choice(all_categories)
    return random.choice(training_data[c]),c

class LSTMnamer(nn.Module):
    
    def __init__(self,in_size,hidden_size,out_size):
        super(LSTMnamer,self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(in_size,hidden_size)
        #self.lstm2 = nn.LSTM(hidden_size,hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.out = nn.LogSoftmax(dim=1)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))

    def forward(self,x):
        y,self.hidden_cell = self.lstm(x.view(len(x),1,-1),self.hidden_cell)
        #y,_ = self.lstm2(y.view(len(x),1,-1))
        y = self.linear(y.view(len(x),-1))
        y = self.out(y)
        return y[-1]
    
    def init_hidden(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))

n_hidden = 128
n_categories = len(names.keys())
model = LSTMnamer(n_letters+n_categories,n_hidden,len(all_letters)-1)
opt = {}
opt['loss'] = nn.NLLLoss()  # negative log likelihood loss
opt['optimizer'] = torch.optim.Adam(model.parameters(), lr=0.002)
opt['epochs'] = 100000

def train(opt,model,training_data):

    loss_function = opt['loss']
    optimizer = opt['optimizer']
    epochs = opt['epochs']
    loss_list = []
    for epoch in range(epochs):
        name, category = get_sample(training_data)
        category = turnToTensor([[category]], MLB2)
        inp,target = turnToTensor(name[0], MLB),torch.LongTensor([all_letters.index(name[1])])
        
        optimizer.zero_grad()
        model.init_hidden()

        out = model(torch.cat((torch.tile(category,(len(inp),1)),inp),axis=1))
        l = loss_function(out.view(1,-1), target)
        loss_list.append(l.item())
        #print(l,out.view(1,-1), category)
        l.backward()
        optimizer.step()

        if epoch%2000 == 0:
            print(f'epoch: {epoch} loss: {np.mean(loss_list):10.8f}')
            loss_list = []


#train(opt,model,training_data)

def get_output(out):
    return np.random.choice(np.arange(len(out)),p=torch.exp(out).detach().numpy())

def generate(model,category=None):
    if category is None:
        category = random.choice(all_categories)
    ct = turnToTensor([[category]], MLB2)
    name = ''
    inp = turnToTensor([['<S>']], MLB)
    i = 0
    model.init_hidden()
    while True:
        with torch.no_grad():
            out = model(torch.cat((torch.tile(ct,(len(inp),1)),inp),axis=1))
        out = get_output(out)
        if all_letters[out] == '<EOS>':
            break
        name += all_letters[out]
        inp = turnToTensor(all_letters[out], MLB)
        i += 1
        if i>20:
            break
    return name,category

for _ in range(10):
    print(generate(model,'Japanese'))

