# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:19:03 2019

@author: Lenovo
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


print(np.zeros(2))
x = np.arange(1,20,0.2)
print(x.shape)
y = x + np.sin(x)

plt.figure()
plt.plot(x,y,x,x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.arange(-1, 1, 0.1),np.arange(-1,1,0.1))
z = (x.ravel()**2-y.ravel()**2).reshape(20,20)

ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
plt.figure()
plt.contourf(x,y,z, levels=[0.1,0.6],
    colors=['#C0C0C0','#808080'], extend='both')