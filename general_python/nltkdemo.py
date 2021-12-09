# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:46:57 2021

@author: 81807
"""

import nltk
from nltk.stem.porter import PorterStemmer
import regex
#nltk.download('punkt')

pstemmer = PorterStemmer()

sentence = "At eight o'clock on Thursday morning \
... Arthur didn't feel very good."

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

print(list(map(lambda x: pstemmer.stem(x),tokenize(sentence))))


