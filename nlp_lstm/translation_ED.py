# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:09:45 2021

@author: Lenovo
"""


import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from translation_dataprep import dataset, tensorSentence

np.set_printoptions(precision=3)

SOS = 0
EOS = 1
MAX_LENGTH = dataset.max_length

# b = oneHot(,wl1.n_index)

class Encoder(nn.Module):
    
    def __init__(self, in_size, hidden_size):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(in_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.hidden_cell = torch.zeros(1, 1, self.hidden_size)
    
    def forward(self,x):
        y = self.embedding(x).view(len(x),1,-1)
        y,self.hidden_cell = self.gru(y,self.hidden_cell)
        return y
        
    def init_hidden(self):
        self.hidden_cell = torch.zeros(1, 1, self.hidden_size)
        
'''
The initial input token for the decoder is the start-of-string <SOS> token, 
and the hidden states are the context vectors based on the weighted sum of the encoder’s activations using attention weights).
'''

class Decoder(nn.Module):

    def __init__(self, hidden_size, out_size,max_length,dropout=0.1):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(out_size,hidden_size)
        self.attn = nn.Linear(hidden_size*2,max_length)
        self.attn_combine = nn.Linear(hidden_size*2,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,out_size)
        
        self.hidden_cell = torch.zeros(1, 1, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,encoder_outputs):
        y = self.embedding(x).view(1,1,-1)
        y = self.dropout(y)
        attn_weights = F.softmax(self.attn(torch.cat((y[0],self.hidden_cell[0]),1)),dim=1)
        y = torch.cat((y[0],torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))[0]),1)
        y = self.attn_combine(y).unsqueeze(0)
        y = F.relu(y)
        y,self.hidden_cell = self.gru(y,self.hidden_cell)
        
        y = F.log_softmax(self.out(y.view(1,-1)),dim=1)
    
        return y, attn_weights
        
    def init_hidden(self):
        self.hidden_cell = torch.zeros(1, 1, self.hidden_size)



hidden_size = 128
encoder = Encoder(dataset.wl1.n_index, hidden_size)
decoder = Decoder(hidden_size, dataset.wl2.n_index, dataset.max_length)
opt = {}
opt['encoder_optimizer'] = torch.optim.Adam(encoder.parameters(), lr=0.01)
opt['decoder_optimizer'] = torch.optim.Adam(decoder.parameters(), lr=0.01)
opt['loss'] = nn.NLLLoss()
opt['iterations'] = 30000
teacher_forcing = 0.5

def train(opt,encoder,decoder,dataset):
    
    iterations = opt['iterations']
    encoder_optimizer = opt['encoder_optimizer']
    decoder_optimizer = opt['decoder_optimizer']
    loss_function = opt['loss']
    iterations = opt['iterations']
    #training_pairs = [tensorPair(random.choice(pairs), wl1, wl2) for _ in range(iterations)]
    training_data = iter(dataset)
    for i in range(iterations):
        inp,target = next(training_data)
        target_length = target.shape[0]
        encoder.init_hidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        y = encoder(inp).view(len(inp),-1)
        encoder_outputs = torch.zeros((dataset.max_length,hidden_size))
        encoder_outputs[:len(y)] = y
        y = None
        decoder_input = torch.tensor([[SOS]])
        decoder.hidden_cell = encoder.hidden_cell
        #print(decoder.hidden_cell)
        loss = 0
        if np.random.rand() > teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_attention = decoder(
                    decoder_input, encoder_outputs)
                loss += loss_function(decoder_output, target[di])
                decoder_input = target[di]
                
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_attention = decoder(
                    decoder_input, encoder_outputs)
                loss += loss_function(decoder_output, target[di])
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # Teacher forcing
                if decoder_input.item() == EOS:
                    break
                
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        if i%1000 == 0:
            print(loss.item() / target_length)
    
train(opt,encoder,decoder,dataset)
    
def evaluate(sentence,encoder,decoder,max_length=10):
    inp = tensorSentence(sentence, dataset.wl1)
    
    with torch.no_grad():
        y = encoder(inp).view(len(inp),-1)
        encoder.init_hidden()
        
        y = encoder(inp).view(len(inp),-1)
        encoder_outputs = torch.zeros((MAX_LENGTH,hidden_size))
        encoder_outputs[:len(y)] = y
        y = None
        decoder_input = torch.tensor([[SOS]])
        decoder.hidden_cell = encoder.hidden_cell
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_attention = decoder(
                decoder_input, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dataset.wl2.index_word[topi.item()])

            decoder_input = topi.squeeze().detach()
    
    return decoded_words, decoder_attentions

