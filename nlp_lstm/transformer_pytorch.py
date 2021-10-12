# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 01:49:35 2021

@author: Lenovo
"""

import sys
if 'C:/users/lenovo/desktop/data/code/nlp_lstm' not in sys.path:
    sys.path.append('C:/users/lenovo/desktop/data/code/nlp_lstm')
import pickle
import numpy as np
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, TransformerDecoderLayer, TransformerDecoder
from torch.utils.data import Dataset, dataset
from translation_dataprep import LangDataset, tensorSentence, Wordlist

'''
rom torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def data_process(raw_text_iter: dataset.IterableDataset) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
'''  
  
class PositionEncoder(nn.Module):

    def __init__(self, d_model: int, max_length: int, dropout: float = 0.1):
        super(PositionEncoder,self).__init__()   
        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.FloatTensor([[np.sin(pos/(np.power(10000,i/d_model))) 
                                  if i%2 == 0 else np.cos(pos/(np.power(10000,(i-1)/d_model))) 
                                  for i in range(d_model)] 
                                 for pos in range(max_length)]).view(max_length,1,d_model)
        
        # buffers = ‘fixed tensors / non-learnable parameters / stuff that does not require gradient’
        # parameters = ‘learnable parameters, requires gradient’

        #self.register_buffer('pe', self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    
class MyEncoder(nn.Module):

    def __init__(self, d_model: int, src_vocab_size: int, max_length: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionEncoder(d_model, max_length, dropout)
        
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, encoder_norm)
        
        self.init_weights()
        self._reset_parameters()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask)
        
        return memory
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class MyDecoder(nn.Module):
    
    def __init__(self, d_model: int,target_vocab_size: int, max_length: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoder = PositionEncoder(d_model, max_length, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        self.out = nn.Linear(d_model,target_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        self._reset_parameters()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, target, memory: Tensor, target_mask) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        target = self.embedding(target) * math.sqrt(self.d_model)
        target = self.pos_encoder(target)
        output = self.transformer_decoder(target,memory,target_mask)
        output = self.out(output)
        return self.softmax(output)
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


with open(r'C:\Users\Lenovo\Desktop\data\code\datasets\nlp_data\eng_fra_processed','rb') as f:
    d = pickle.load(f)

dataset = LangDataset(d['pairs'], d['wl1'], d['wl2'])

src_vocab_size = dataset.wl1.n_index  # size of vocabulary
target_vocab_size = dataset.wl2.n_index
max_length = 10 # max sentence length
d_model = 256  # embedding dimension
d_hid = 256  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
SOS = 0
EOS = 1

encoder = MyEncoder(d_model, src_vocab_size, max_length, nhead, d_hid, nlayers, dropout)
decoder = MyDecoder(d_model, target_vocab_size, max_length, nhead, d_hid, nlayers, dropout)
opt = {}
opt['loss'] = nn.CrossEntropyLoss()
opt['encoder_optim'] = torch.optim.SGD(encoder.parameters(), lr=0.001)
opt['decoder_optim'] = torch.optim.SGD(decoder.parameters(), lr=0.001)
opt['iters'] = 1000
opt['teacher_forcing'] = 0

def train(opt,encoder,decoder,dataset):
    loss_fn = opt['loss']
    encoder_optim = opt['encoder_optim']
    decoder_optim = opt['decoder_optim']
    iters = opt['iters']
    teacher_forcing = opt['teacher_forcing']
    
    training_data = iter(dataset)
    total_loss = 0
    print_every = 50
    try:
        for i in range(iters):
            inp,target = next(training_data)
            target_length = target.size(0)
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            l = 0
            
            memory = encoder(inp,None)
            decoder_input = torch.tensor([[SOS]])
            
            if np.random.rand() > teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for j in range(target_length):
                    target_mask = generate_square_subsequent_mask(decoder_input.shape[0])
                    out = decoder(decoder_input,memory,target_mask)
                    
                    l += loss_fn(out[-1].view(-1,out.size(-1)),target[j])
                    val, index = out[ -1].data.topk(1)

                    decoder_input = torch.cat((decoder_input,target[j].view(-1,1)),dim=0)
                    
            else:
                for j in range(target_length):
                    target_mask = generate_square_subsequent_mask(decoder_input.shape[0])
                    out = decoder(decoder_input,memory,target_mask)

                    l += loss_fn(out[-1].view(-1,out.size(-1)),target[j])
                    val, index = out[ -1].data.topk(1)
              
                    decoder_input = torch.cat((decoder_input,index),dim=0)
                    
            l.backward()
            encoder_optim.step()
            decoder_optim.step()

            
            total_loss += l.item()
            if (i+1)%print_every==0:
                print('loss ',total_loss/print_every)
                print(evaluate(encoder,decoder,'the cat is missing',dataset.wl1,dataset.wl2,max_length))

                total_loss = 0
                
    except KeyboardInterrupt:
        return encoder, decoder
    
    
    return encoder, decoder

def evaluate(encoder,decoder,sentence,wl1,wl2,max_length):
    
    with torch.no_grad():
        inp = tensorSentence(sentence, wl1)
        memory = encoder(inp,None)
        decoder_input = torch.tensor([[SOS]])
        
        for i in range(max_length):
            target_mask = generate_square_subsequent_mask(decoder_input.shape[0])
            out = decoder(decoder_input,memory,target_mask)
            
            val, index = out[-1].data.topk(1)
            if index.item() == EOS:
                break                
            decoder_input = torch.cat((decoder_input,index),dim=0)
        decoder_input = decoder_input.clone().detach().numpy().reshape(-1,)
    l = [wl2.index_word[i] for i in decoder_input]
    
    return ' '.join(l)


print(evaluate(encoder,decoder,'the cat is missing',dataset.wl1,dataset.wl2,max_length))
encoder, decoder = train(opt,encoder,decoder,dataset)
print(evaluate(encoder,decoder,'the cat is missing',dataset.wl1,dataset.wl2,max_length))

