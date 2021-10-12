# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 01:48:30 2021

@author: Lenovo
"""

import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import random

#nltk.download('punkt')
#nltk.download('stopwords')
ps = PorterStemmer()

st = stopwords.words('english')

# stemming converts words to its stem. It need not be a dictionary word, 
# removes prefix and affix based on few rules

# lemmatization — will be a dictionary word. reduces to a root synonym.

folder_path = 'c:/users/lenovo/desktop/data/code/datasets/nlp_data/stories'

# tf(t,d) = count of t in d / number of words in d
# df(t) = occurrence of t in documents
# idf(t) = log(N/(df + 1)) for document set N
# tf-idf(t, d) = tf(t, d) * log(N/(df + 1))

target_name = 'index.html'
dataset = []
for path, subdir, files in os.walk(folder_path):
    
    if target_name in files:
        with open(os.path.join(path,target_name),'r') as f:
            file = f.read().strip()
        file_names,file_titles = [],[]
        file_titles = re.findall('<BR><TD> (.*)\n', file)
        file_names = re.findall('><A HREF="(.*)">',file)[-len(file_titles):]
        if len(file_titles)>0 and len(file_names)>0:
            file_names = [os.path.join(path,i) for i in file_names]
            dataset.extend(zip(file_names,file_titles))

words = {}
docs = {}
for file_path,title in dataset:
    with open(file_path,'r',encoding='utf8', errors='ignore') as f:
        file = f.read().strip()
    file = file + f' {title}'
    file = re.sub("[!\"#$%&()*+-./:;<=>?@[\]^_`\'{|}~\n\\\]",' ',file.lower())
    word_tokens = word_tokenize(file)
    filtered_sentence = [ps.stem(w) for w in word_tokens if not w in st and len(w)>1]
    docs[title] = len(filtered_sentence)
    for w in filtered_sentence:
        if w in words.keys():
            if title in words[w].keys():
                words[w][title] += 1
            else:
                words[w][title] = 1
        else:
            words[w] = {}
            words[w][title] = 1

word_list = list(words.keys())
doc_list = list(set([title for _,title in dataset]))
N = len(doc_list)

def get_random(l):
    return random.choice(l)

word = get_random(word_list)
doc = get_random(list(words[word].keys()))

def tf_idf(word,doc,words,docs):
    
    tf = 0
    if doc in words[word].keys():
        tf = words[word][doc]/docs[doc]
        
    df = len(words[word].keys())
    idf = np.log(N/(df+1))
    
    return tf*idf

tf_idf(word,doc,words,docs)



        