# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:22:35 2021

@author: Lenovo
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('C:\\Users\\Lenovo\\Desktop\\data\\test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def add(x,y):
    
    return x+y

def subtract(x,y):
    
    return x-y

def multiply(x,y):
    
    return x*y

def divide(x,y):
    
    try:
        x/y
    except ZeroDivisionError:
        logger.exception('division by 0')
    else:
        return x/y
    
n1 = 3
n2 = 0

a = add(n1, n2)
logger.debug(f'{n1} + {n2} = {a}')
s = subtract(n1, n2)
logger.debug(f'{n1} - {n2} = {s}')
m = multiply(n1, n2)
logger.debug(f'{n1} * {n2} = {m}')
d = divide(n1, n2)
logger.debug(f'{n1} / {n2} = {d}')
