# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:29:48 2021

@author: 81807
"""


import numpy as np
import sys
import os
from multiprocessing import Pool,Process,Lock,Queue
import multiprocessing as mp
import time

def sq(x):
    
    print('input: %d' % x)
    #time.sleep(1)
    retValue = x * x
    print('double: %d' % (retValue))
    return(retValue)

def sq_wrapper(args):
    return sq(*args)

    
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hellof', name)

def foo(q):
    q.put('helloq')
    
def sqrun(x,q,name):
    print('name ',name)
    for _ in range(1000):
        l = []
        for i in range(x):
            l.append(i*i/np.sqrt(i+1))
    q.put([len(l),name,os.getpid()])

if __name__ == '__main__':

    s = time.time()
    with Pool(4) as p:
        l = list(p.map(sq,range(16)))
        print(l)
    print(time.time()-s)
    s = time.time()
    with Pool(8) as p:
        print(p.map(sq,range(16)))
    print(time.time()-s)
    
    s = time.time()
    q = Queue()
    q2 = Queue()
    p = Process(name='p1',target=sqrun, args=(100,q,'p1'))
    p2 = Process(name='p2',target=sqrun, args=(50,q,'p2'))
    print('going to start')
    p.start()
    print(q.get())
    p2.start()
    print(q.get())
    p.join()
    p2.join()
    print(time.time()-s)