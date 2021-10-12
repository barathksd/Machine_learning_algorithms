# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:11:23 2021

@author: Lenovo
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set_style('darkgrid')

flight_data = sns.load_dataset("flights")
flight_data.head()
flight_data['date'] = flight_data.transpose().apply(lambda x: str(x['year'])+'/'+x['month'])

for i in ['date','month']:
    s = sns.lineplot(x=flight_data[i],y=flight_data['passengers'])
    s.set_title('Total passengers over time')
    plt.show()

n = flight_data.shape[0]

all_data = flight_data['passengers'].values
train_data = all_data[:int(0.8*n)]
test_data = all_data[int(0.8*n):]

standard = StandardScaler()
train_data = standard.fit_transform(train_data.reshape(-1,1))
test_data = standard.transform(test_data.reshape(-1,1))

train_data = torch.FloatTensor(train_data).view(-1)
train_window = 12

def getallseq(input_data,seq_length):
    return [(input_data[i:i+seq_length],input_data[i+seq_length:i+seq_length+1]) for i in range(len(input_data)-seq_length)]
    
train_seq = getallseq(train_data, train_window)


class LSTMts(nn.Module):
    
    def __init__(self, in_size,hidden_dim=60,out_size=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_size,hidden_dim)
        self.out = nn.Linear(hidden_dim,out_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_dim),
                            torch.zeros(1,1,self.hidden_dim))
        
    def forward(self,data):
        # LSTM has 3 inputs, previous state output/current input, previous hidden_state, previous cell state  
        # input_shape is (seq, batch, feature) by default
        y,self.hidden_cell = self.lstm(data.view(len(data),1,-1),self.hidden_cell) 
        # shape of y = (seq, batch, hidden_dim)   
        # shape of hidden_cell[0] = hidden_cell[1] = (1,1,hidden_dim)
        y = self.out(y.view(len(data),-1))
        return y[-1]
    
    def init_hidden(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_dim),
                            torch.zeros(1,1,self.hidden_dim))

model = LSTMts(in_size=1,hidden_dim=60,out_size=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model_dict = {'model':model,'opt':optimizer,'loss':loss_fn}

def train(input_data,model_dict,epochs):
    model = model_dict['model']
    optimizer = model_dict['opt']
    loss_fn = model_dict['loss']
    for i in range(epochs):
        for seq, labels in input_data:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_dim),
                        torch.zeros(1, 1, model.hidden_dim))
            out = model(seq)
            loss = loss_fn(out,labels)
            loss.backward()
            optimizer.step()
        if i%10 == 1:
            print(f'epoch: {i:3} loss: {loss.item():10.8f}')
    
    print(f'epoch: {i:3} loss: {loss.item():10.10f}')

train(train_seq,model_dict,epochs=60)

def pred(input_data,model,num,seq_length):
    data = input_data[-1*seq_length:]
    for i in range(num):
        seq = torch.FloatTensor(input_data[-seq_length:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_dim),
                        torch.zeros(1, 1, model.hidden_dim))
            p = model(seq)
        input_data.append(p.item())
    return input_data

predictions = pred(train_data.tolist()[:50],model,94,train_window)
predictions = standard.inverse_transform(np.array(predictions))

sns.lineplot(x=np.arange(flight_data.shape[0]),y=flight_data['passengers'])
sns.lineplot(x=np.arange(predictions.shape[0]),y=predictions)
plt.show()


