# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 13:56:42 2021

@author: Lenovo
"""

import os
import re
import pickle
import random
import unicodedata
import string
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SOS = 0
EOS = 1

#OHE = OneHotEncoder(categories=[np.array([i for i in all_letters])],handle_unknown='ignore')
#OHE.fit([[i] for i in all_letters])



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Wordlist:
    
    def __init__(self,name):
        self.name = name
        self.word_count = {}
        self.word_index = {}
        self.index_word = {0:'<SOS>',1:'<EOS>'}
        self.n_index = 2
        
    def add_sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word(word)
    
    def add_word(self,word):
        if word not in self.word_index:
            self.word_index[word] = self.n_index
            self.index_word[self.n_index] = word
            self.n_index += 1
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1


def load_data(file_path,lang1='English',lang2='French',max_length=10,limit=25000):
    
    with open(file_path,'rb') as f:
        n = f.read().strip().decode('utf-8').split('\n')[:limit]
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in n]
    pairs = [[p[0],p[1]] for p in pairs if len(p[0].split(' '))<max_length and len(p[1].split(' '))<max_length]
    wl1,wl2 = Wordlist(lang1),Wordlist(lang2)
    for pair in pairs:
        wl1.add_sentence(pair[0])
        wl2.add_sentence(pair[1])
    return pairs,wl1,wl2

def indexSentence(sentence,wl):
    return [wl.word_index[word] for word in sentence.split(' ')]

def tensorSentence(sentence,wl):
    l = indexSentence(sentence,wl)
    l.append(EOS)
    #a = np.zeros((len(l),size))
    #a[np.arange(len(l)),l] = 1
    return torch.LongTensor(l).view(-1,1)

def tensorPair(pair,wl1,wl2):
    return (tensorSentence(pair[0],wl1),tensorSentence(pair[1],wl2))


class LangDataset(Dataset):
    
    def __init__(self, pairs, wl1, wl2):
        super().__init__()
        self.lang1 = wl1.name
        self.lang2 = wl2.name

        self.pairs = pairs
        self.wl1 =  wl1
        self.wl2 = wl2
        self.pairs = [tensorPair(self.pairs[i], self.wl1, self.wl2) for i in range(len(self.pairs))]
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return random.choice(self.pairs)
        
        
        
# print(random.choice(pairs))


file_path = os.path.join('C:/users/lenovo/desktop/data/code/datasets/nlp_data/eng-fra.txt')
lang1='English'
lang2='French'
max_length=10
limit=100000

def save_data(file_path,lang1,lang2,max_length,limit):
    pairs,wl1,wl2 = load_data(file_path,lang1,lang2,max_length,limit)
    
    d = {'pairs':pairs,'wl1':wl1,'wl2':wl2}
    
    with open(r'C:\Users\Lenovo\Desktop\data\code\datasets\nlp_data\eng_fra_processed','wb') as f:
        pickle.dump(d,f)




