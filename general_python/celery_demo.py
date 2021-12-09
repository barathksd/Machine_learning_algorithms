# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:55:43 2021

@author: 81807
"""

from celery import Celery


app = Celery('celery_demo',broker='pyamqp://guest@localhost:5672//',backend='redis://localhost:6379')

@app.task
def add(x,y):
    return x+y