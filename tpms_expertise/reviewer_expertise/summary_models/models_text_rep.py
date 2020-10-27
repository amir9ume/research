import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
from torchtext.utils import download_from_url
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import operator


class LSTMWithAttentionAutoEncoder(nn.Module):
    def __init__(self,
        emb_dim,
    
        src_vocab_size,
        src_hidden_dim,
        batch_size,
        pad_token_src,
        bidirectional=False,
        batch_first=True,
        nlayers=1,
        dropout=0.,
        device='cpu'
    ):
        super(LSTMWithAttentionAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.src_vocab_size = src_vocab_size
        self.src_hidden_dim = src_hidden_dim
        self.batch_size = batch_size
        self.device= device

        self.pad_token_src = pad_token_src
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.directions=2
        else:
            self.directions=1
        self.nlayers = nlayers
        self.dropout = dropout
        self.batch_first = batch_first
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.emb_dim, self.pad_token_src)
        
        # if self.bidirectional:
        #     src_hidden_dim = src_hidden_dim // 2
        
        self.encoder = nn.LSTM(self.emb_dim, src_hidden_dim, 
                num_layers=self.nlayers, dropout=self.dropout,
                bidirectional=self.bidirectional, batch_first=self.batch_first)
        
        self.input_dim=50
        self.output_dim= 25
        self.projection_layer= Projection_layer(self.input_dim, self.output_dim)

        self.init_weights()

    
    def init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
    

    def get_state(self, input):
        #print(input.size(0), input.size(1))
        batch_size = input.size(0) \
            if self.batch_first else input.size(1)
        
        h0_encoder = Variable(torch.zeros(self.nlayers * self.directions,batch_size,self.src_hidden_dim ))
        c0_encoder = Variable(torch.zeros(self.nlayers * self.directions,batch_size,self.src_hidden_dim))
    
        return h0_encoder.to(self.device), c0_encoder.to(self.device)

    def forward(self, input):
        src_embedding = self.src_embedding(input)
        # trg_embedding = self.trg_embedding(input)
        h0, c0 = self.get_state(input)
        #self.encoder.flatten_parameters()
        encoded, (h_,c_) = self.encoder(src_embedding, (h0, c0))
        if self.bidirectional:
            h_t = torch.cat((h_[-1], h_[-2]), 1)
            c_t = torch.cat((c_[-1], c_[-2]), 1)
        else:
            h_t = h_[-1]
            c_t = c_[-1]
        
        #for now doing this to remove thinking about the 512 tokens
    #    encoded_sum= torch.sum(encoded,dim=1)
        z= self.projection_layer(h_t)
        return z


class Projection_layer(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,    
        device='cpu'
    ):
        super(Projection_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.non_linearity= nn.ReLU()
        self.out= nn.Linear(input_dim,output_dim)

    def forward(self, input):
        h_= self.non_linearity(input)
        z= self.out(h_)
        return z

