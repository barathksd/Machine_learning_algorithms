import numpy as np
import argparse
import cv2
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import dot, tile, linalg
from numpy.linalg import inv, multi_dot

sns.set_style('darkgrid')
np.set_printoptions(suppress=True)


t = np.arange(0,10,0.1)
m = 1
k = 1
g = 10
v = -m*g*(1-np.exp(-k*t/m))/k
plt.plot(t,v)
plt.show()


x0 = 500
x = x0 - m*g*(t+m*(np.exp(-k*t/m)-1)/k)/k
plt.plot(t,x)
plt.show()

z = x + np.random.randn(t.shape[0])*8
plt.plot(t,z)
plt.show()
#z = np.round(np.vstack((z,np.gradient(z))),3).T


dt = 0.1
R = np.array([[64,0],[0,130]])
A = np.array([[1,dt],[0,1-k*dt/m]])
B = np.array([[1,0],[0,-g*dt]])
Q = np.array([[10,0],[0,1]])
u = np.array([0,1])
pt = np.array([[0,0],[0,0]])
H = np.array([[1,0],[0,0]])

p = []
xp = np.array([x0,0])

for i,_ in enumerate(t):
    xp = np.matmul(A,xp) + np.matmul(B,u)
    p.append(xp[0])

xp = np.array([x0-50,0])
xk = []

for i,_ in enumerate(t):
    if i>2:
        Q = np.array([[0.03,0],[0,1]])
    else:
        Q = np.array([[10,0],[0,1]])
        
    zt = np.array([z[i],0])
    #print('zt\n',zt)
    
    xp = np.matmul(A,xp) + np.matmul(B,u)
    pt = multi_dot((A,pt,A.T)) + Q
    K = multi_dot((pt,H.T,inv(multi_dot((H,pt,H.T))+R)))
    xp = xp + np.matmul(K,zt-np.matmul(H,xp))
    pt = pt - multi_dot((K,H,pt))
    xk.append(xp[0])
    '''
    print('xp\n',xp)
    print('k\n',K)
    print('pt\n',pt)
    print('')
    '''
    #xp,pt = kf_predict(xp, pt, A, Q, B, u)
    #xp,pt,K,IM,IS,LH = kf_update(xp, pt, zt, H, R)
    
plt.plot(t,x,'b')
plt.plot(t,xk,'r')
plt.plot(t,z,'g--')
#plt.plot(t,p)
plt.show()






