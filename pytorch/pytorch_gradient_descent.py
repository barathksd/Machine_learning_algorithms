# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:40:32 2019

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import seaborn as sns
import pandas as pd

def data():
    er = torch.randn(100,1)*0.2
    x_d = Variable(torch.arange(1,21,0.2), requires_grad=False).view(-1,1)
    y_t = 5*x_d/(x_d+2)   # a=5, b=2 
    y_d = er+y_t
    return x_d,y_d,y_t

x_d,y_d,y_t = data()
n = x_d.shape[0]



def rmse(y, y_hat):    
    return torch.sqrt(torch.mean((y - y_hat).pow(2).sum()))

# print(rmse(y_t,y_d))

def forward(x, a,b):
    return a*x/(x+b)

alpha = 0.4
a1 =  Variable(torch.FloatTensor([1]), requires_grad=True)
b1 =  Variable(torch.FloatTensor([1]), requires_grad=True)
exp_loss = rmse(y_t,y_d)
loss_history = []
coeff = []


opt = torch.optim.Adam([a1,b1],lr=alpha,betas=(0.9,0.999))
it = 10
i=1
while True:
    y_hat = forward(x_d, a1,b1)
    opt.zero_grad()
    loss = rmse(y_d,y_hat)
        
    loss_history.append(loss.data) 
    coeff.append(zip(a1.data,b1.data))
    
    loss.backward()
    opt.step()
    if(loss.detach().numpy()<=exp_loss.detach().numpy()*0.992 or i>8192):
        break
    i += 1

print("loss = %s" % loss.data)
print('expected_loss ',exp_loss)
print("a1,b1 = %s %s" % (a1.data[0], b1.data[0]))
y_hat = forward(x_d,a1,b1)

plt.figure()
plt.scatter(x_d,y_d)
plt.plot(np.array(x_d),np.array(y_t),c='r')
plt.plot(np.array(x_d),np.array(y_hat.detach()),c='y')



sns.set(style="darkgrid")

x_d = np.array(x_d).reshape(-1)
x_d = [i for i in x_d]
y_d = np.array(y_d).reshape(-1)
y_d = [i for i in y_d]
y_h = np.array(y_hat.detach()).reshape(-1)
y_h = [i for i in y_h]

k = pd.DataFrame([x_d,y_d],index=['a','b']).T
g = sns.jointplot('a', 'b', data=k, kind="scatter",
                  xlim=(0.5, 21), ylim=(0, 6), color="m", height=7)









    