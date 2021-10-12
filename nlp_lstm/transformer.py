# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:58:33 2021

@author: Lenovo
"""

import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


MAX_LENGTH = 64

    
def position_encoder(d_model, max_length=MAX_LENGTH):

    '''
    Positional Encoding
    PE(pos,2i) = sin(pos/(10000^(2*i/d_model)))
    PE(pos,2i+1) = cos(pos/(10000^(2*i/d_model)))        
    '''
    return torch.FloatTensor([[np.sin(pos/(np.power(10000,i/d_model))) 
                                  if i%2 == 0 else np.cos(pos/(np.power(10000,(i-1)/d_model))) 
                                  for i in range(d_model)] 
                                 for pos in range(max_length)]).unsqueeze(0)

def plot_array(ar):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(ar)
    fig.colorbar(cax)
    plt.show()

def attention(q, k, v, d_k, mask, dropout=0.1):
    # (bs,head,sl,d_k) x (bs,head,d_k,sl) = (bs,head,sl,sl)
    scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(d_k)
    
    if mask:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout:
        scores = dropout(scores)
    
    # (bs,head,sl,sl) x (bs,head,sl,d_k) = (bs,head,sl,d_k)
    return torch.matmul(scores, v)

def mask_lt(size):
    
    # returns a lower triangular matrix with ones
    return torch.FloatTensor(np.triu(np.ones((1, size, size)),
k=1).astype('uint8') == 0)
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, nhead, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.h = nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
        return output
        
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(nhead, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
    def forward(self, x, e_outputs, src_mask, trg_mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            return x
# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class Encoder(nn.Module):
    
    def __init__(self, d_model, nhead, vocab_size, n_encoders, max_length=MAX_LENGTH):
        super(Encoder,self).__init__()
        '''
        Input -> Embedding layer -> Positional encoding -> Multi-head attention ->
        add & batch norm -> Feed forward layer -> add & batch norm -> Linear -> 
        Softmax
        '''
        # 1. Embedding
        # d_model = number of expected features in the encoder/decoder inputs
        # nhead = number of heads in multi-head attention layer
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.max_length = max_length
        self.n_encoders = n_encoders
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = position_encoder(d_model,max_length)
        self.encoding_layers = get_clones(EncoderLayer(d_model, nhead),n_encoders)
        self.norm = Norm(d_model)
        
        
    def forward(self, x, mask):
        
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:,:len(x)].clone().detach().requires_grad_(False)
        for i in range(self.n_encoders):
            x = self.encoding_layers[i](x,mask)
        return self.norm(x)
        
class Decoder(nn.Module):
    
    def __init__(self, d_model, nhead, vocab_size, n_decoders, max_length=MAX_LENGTH):
        super(Decoder,self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.vocab_size = vocab_size
        self.n_decoders = n_decoders
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = position_encoder(d_model, max_length)
        self.decoding_layers = get_clones(DecoderLayer(d_model, nhead), n_decoders)
        self.norm = Norm(d_model)
        

    def forward(self, x, e_outputs, en_mask, de_mask):
        
        x = self.embedding(x)
        x = x * np.sqrt(self.d_model)
        x = x + self.pe[:,:len(x)].clone().detach().requires_grad_(False)
        for i in range(self.n_decoders):
            x = self.decoding_layers[i](x, e_outputs, en_mask, de_mask)
            
        return self.norm(x)       


class Transformer(nn.Module):
    
    def __init__(self, d_model, nhead, src_vocab_size, target_vocab_size, n_encoders=1, n_decoders=1, encoder=None, decoder=None):
        super(Transformer,self).__init__()
        
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(d_model, nhead, src_vocab_size, n_encoders)

        if decoder:
            self.decoder = nn.Embedding(target_vocab_size, d_model)
        else:
            self.decoder = Decoder(d_model, nhead, target_vocab_size, n_decoders)
        
        self.out = nn.Linear(d_model,target_vocab_size)
        self._reset_parameters()
        
    def forward(self, src, target, src_mask, target_mask):
        enc_outputs = self.encoder(src,src_mask)
        target = self.decoder(target,enc_outputs,src_mask, target_mask)
        return self.out(target)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
d_model = 100
nhead = 10
src_vocab_size = 5000
target_vocab_size = 6000
n_encoders = 1
n_decoders = 1
max_length = 64

model = Transformer(d_model, nhead, src_vocab_size, target_vocab_size)

opt = {}
opt['optimizer'] = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
opt['loss'] = nn.CrossEntropyLoss()
opt['epochs'] = 100

def train(opt,model,data):
    
    optimizer = opt['optimizer']
    loss = opt['loss']
    epochs = opt['epochs']
    
    for epoch in epochs:
        None












