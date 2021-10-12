# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22:19:13 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


onehot_encoder = DictVectorizer()
inst= [{'city': 'New York'},{'city': 'sans'},{'city': 'London'}]
dt=onehot_encoder.fit_transform(inst).toarray()
print(dt)

cp=['This corpus contains eight unique words: UNC played Duke' \
    'in, basketball, lost, the, and game The corpus unique words '\
    'comprise its vocabulary The bag-of-words '\
    'model uses a feature vector with an element for each of the words in the corpus '\
    'vocabulary to represent each document. Our corpus has eight unique words, so each '\
    'document will be represented by a vector with eight elements The number of elements '\
    'that comprise a feature vector is called the vectors dimension A dictionary maps the '\
    'vocabulary to indices in the feature vector ',

    'In the most basic bag-of-words representation, each element in the feature vector is a '\
    'binary value that represents whether or not the corresponding word appeared in the '\
    'document For example the frst word in the frst document is UNC The frst word in '\
    'the dictionary is UNC so the frst element in the vector is equal to one. The last word '\
    'in the dictionary is game The frst document does not contain the word game, so '\
    'the eighth element in its vector is set to 0 The CountVectorizer class can produce '\
    'a bag-of-words representation from a string or file By default, CountVectorizer '\
    'converts the characters in the documents to lowercase and tokenizes the documents '\
    'Tokenization is the process of splitting a string into tokens or meaningful sequences '\
    'of characters Tokens frequently are words but they may also be shorter sequences '\
    'including punctuation characters and affxes The CountVectorizer class tokenizes '\
    'using a regular expression that splits strings on whitespace and extracts sequences of '\
    'characters that are two or more characters in length',
    
    'PCA is most useful when the variance in a data set is distributed unevenly across the '\
    'dimensions. Consider a three-dimensional data set with a spherical convex hull. PCA '\
    'cannot be used effectively with this data set because there is equal variance in each '\
    'dimension; none of the dimensions can be discarded without losing a signifcant '\
    'amount of information. '
    
    ]

vectorizer = CountVectorizer(stop_words='english')

a = vectorizer.fit_transform(cp).todense()
b = vectorizer.vocabulary_
b1 = [i for i in b.values()]
b2 = [i for i in b.keys()]
b1.sort()
b2.sort()
print(a)
print(b1)
print(b2)
