# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:30:19 2021

@author: Lenovo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
n = 0
# For each words-list (sentence) and tags-list in each tuple of training_data
for words, tags in training_data:
    for word in words:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = n  # Assign each word with a unique index
            n += 1
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index
ix_to_tag = {v:k for k,v in tag_to_ix.items()}

EMBEDDING_DIM = 10
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.out_layer = nn.Linear(hidden_dim,tagset_size)
        
    def forward(self, sentence):
        embeds = self.embedding_layer(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.out_layer(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
        
        
model = LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,n,len(tag_to_ix))
opt = {}
opt['loss'] = nn.NLLLoss()  # negative log likelihood loss
opt['optimizer'] = torch.optim.SGD(model.parameters(), lr=0.1)
opt['epochs'] = 100

'''
out = [1,0,0]
out = torch.FloatTensor(out)
target = torch.tensor([0])
nn.NLLLoss()(out.view(1,-1),target)
Out[98]: tensor(-1.)

NLLloss = - out[target]
'''


def test(sentence,model,word_to_ix,ix_to_tag):
    with torch.no_grad():
        inputs = prepare_sequence(sentence, word_to_ix)
        tag_scores = model(inputs)
        val,ind = tag_scores.max(dim=1)
        l = []
        for i in np.uint8(ind):
            l.append(ix_to_tag[i])
        return l
    
    
l = test(training_data[0][0],model,word_to_ix,ix_to_tag)
    
def train(opt,model,training_data):
    
    loss_function = opt['loss']
    optimizer = opt['optimizer']
    epochs = opt['epochs']
    for epoch in range(epochs):
        for sentence,tags in training_data:
            inputs = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(inputs)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            
        if epoch %10 == 0: print(epoch,loss.item())


train(opt,model,training_data)

l = test(training_data[0][0],model,word_to_ix,ix_to_tag)


