# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 14:31:47 2020

@author: Lenovo
"""

#import os
#import multiprocessing as mp
#from multiprocessing import Process

import numpy as np


class Person:
    
    def __init__(self,name,age=None,gender=None):
        self.name = name
        self.age = age
        self.gender = gender
    
    @classmethod
    def getname(cls,name):
        return cls(name)
    
a = Person('a')

class Cat:
    def __init__(self):
        self.__name = 'ミケ'

    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        self.__name = name
    
c = Cat()








