import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
import os

import pandas as pd 

#load data
import pickle
import numpy as np
from scipy.stats import entropy
import os

torch.manual_seed(1)

class Attention_Module(nn.Module):
    def __init__(self,num_topics,attention_matrix_size):
        super(Attention_Module,self).__init__()
        self.num_topics= num_topics
        self.attention_matrix_size= attention_matrix_size
        self.W_Q= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_K= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_V = nn.Linear(self.num_topics,self.attention_matrix_size)
        self.padding=True
       # self.dropout=dropout

   # def init_weights(self):

    def element_wise_mul(self,input1, input2):
        feature_list = []
        for feature_1, feature_2 in zip(input1, input2):
            #feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
            feature = feature_1 * feature_2
            feature_list.append(feature.unsqueeze(0))
        output = torch.cat(feature_list, 0)
        return torch.sum(output, 2)


    def forward(self, submitter_emb, reviewer_emb):
        Q= self.W_Q(submitter_emb)
        K= self.W_K(reviewer_emb)
     #   K= self.dropout(K)
        normalizing_factor= (math.sqrt(self.attention_matrix_size))
        e= (torch.bmm(Q.unsqueeze(dim=1),K.permute(0,2,1)) /normalizing_factor )
        ww= self.element_wise_mul(e,reviewer_emb.permute(0,2,1))
        return ww

class Multi_Head_Attention(nn.Module):
    def __init__(self,number_attention_heads,num_topics,attention_matrix_size):
        super(Multi_Head_Attention,self).__init__()
        self.number_attention_heads= number_attention_heads
        self.num_topics= num_topics
        self.attention_matrix_size= attention_matrix_size
        #as the output from this multi-head should always be a fixed final size
        self.w_out= nn.Linear(20,1)

        self.heads= nn.ModuleList()
        for i in range(self.number_attention_heads):
            self.heads.append(Attention_Module(self.num_topics, self.attention_matrix_size))
        
    def forward(self, submitter_emb, reviewer_emb):
        x= []
        for i in range(self.number_attention_heads):
            x.append(self.heads[i](submitter_emb,reviewer_emb))
        z= torch.stack(x,dim=1)
        o= torch.sum(z,dim=1)
        return o


class Match_LR(nn.Module):
    def __init__(self,batch_size,submitter_emb_dim,reviewer_emb_dim,
                 n_classes,attn_over_docs=True):
        super(Match_LR, self).__init__()
        self.attn_over_docs=attn_over_docs
        self.padding=False
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim= reviewer_emb_dim
        
        self.weights_add = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True)#.cuda()
        self.weights_diff = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True)#.cuda()
        self.weights_multi = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True)#.cuda()
        
        self.fc2 = nn.Linear(self.reviewer_emb_dim, self.reviewer_emb_dim)
        self.output = nn.Linear(128, 1)
        self.combined = nn.Linear(self.submitter_emb_dim, 128)   
        
        self.dropout= nn.Dropout(p=0.2)

        if self.attn_over_docs:
            self.padding=True
          #  self.attention_module= Attention_Module(25,20)#num_topics,attention_matrix_size
            self.attention_module=Multi_Head_Attention(6,25,20)
            self.init_weights()
        
    def init_weights(self):
        initrange = 4.0
        #self.weights.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.fill_(0)
        self.weights_add.data.uniform_(-initrange, initrange)
        self.weights_diff.data.uniform_(-initrange, initrange)
        self.weights_multi.data.uniform_(-initrange, initrange)

    
    def forward(self, submitter_emb, reviewer_emb):
        if self.attn_over_docs==True:
            reviewer_emb= self.attention_module(submitter_emb, reviewer_emb)
       
        add = submitter_emb + self.fc2(reviewer_emb)
        diff = submitter_emb - self.fc2(reviewer_emb)
        multi = submitter_emb * (self.fc2(reviewer_emb))     
        
        #commenting out multi term for the moment
      #  combo = self.combined(nn.Tanh()(self.weights_add * add) + nn.Tanh()(self.weights_diff * diff) + nn.Tanh()(self.weights_multi * multi))
        combo = self.combined(nn.Tanh()(self.weights_add * add) + nn.Tanh()(self.weights_diff * diff)) #+ nn.Tanh()(self.weights_multi * multi))
        op = 3*torch.sigmoid(self.output(combo))
        return op.view(-1)
    
            
class Regression_Attention_Over_docs(nn.Module):
    def __init__(self,
                 batch_size, n_classes,attn_flag=True, test_flag=True):
        super(Regression_Attention_Over_docs, self).__init__()
        
        self.padding=True
        self.num_topics= 25
        self.attention_matrix_size= 20

#        self.n_classes = n_classes
        self.batch_size = batch_size        
        
        self.attn_flag= attn_flag
        self.test_flag= test_flag

        self.W_Q= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_K= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_V = nn.Linear(self.num_topics,self.attention_matrix_size)
        
        self.batch_norm= torch.nn.BatchNorm1d(self.attention_matrix_size)
        self.w_submitted= nn.Linear(self.num_topics,self.attention_matrix_size)
        self.w_out= nn.Linear(self.attention_matrix_size*2,1)
 
    def init_weights(self):
        initrange = 0.1
        # self.W_K.weight.data.uniform_(-initrange, initrange)
        # self.W_Q.weight.data.uniform_(-initrange, initrange)
        # self.W_V.weight.data.uniform_(-initrange, initrange)
        
    #    self.w_out.weight.data.uniform_(0,3)
        self.w_out.bias.data.fill_(1)

    def scale_sigmoid(self,x):
        return 3*torch.sigmoid(x)

    #forward is actually defining the equation
    def forward(self, submitter_emb, reviewer_emb):    
        if self.attn_flag:
            Q= self.W_Q(submitter_emb)
        #    Q=self.batch_norm(Q)
            K= self.W_K(reviewer_emb)
            V= self.W_V(reviewer_emb)
            

            #matrix multiplication QxK.T , same as in the paper
            normalizing_factor= (math.sqrt(self.attention_matrix_size))
            e= torch.bmm(Q.unsqueeze(dim=1),K.permute(0,2,1))/normalizing_factor
            
            if self.test_flag:
                ww=torch.bmm(e, reviewer_emb)
                x= self.W_V(ww).squeeze(dim=1)
            else:
                softm= torch.nn.Softmax(dim=2)
                a=softm(e)
                attn_output= torch.bmm(a,V)
                x= torch.sum(attn_output,dim=1)
            #    x=self.batch_norm(x)
            combine= torch.cat((x,self.w_submitted(submitter_emb)),dim=1 )
        else:
            x= torch.mean(reviewer_emb,dim=1)
            combine= torch.cat((self.W_V(x),self.w_submitted(submitter_emb)),dim=1 )
        
                
        out= self.w_out(combine)
        out= self.scale_sigmoid(out)
        return out.squeeze(dim=1)


class Regression_Simple(nn.Module):
    def __init__(self,submitter_emb_dim, reviewer_emb_dim,
                 batch_size, n_classes,mode):

        super(Regression_Simple, self).__init__()
          
        self.num_topics= 25
        self.batch_size = batch_size
        self.W= nn.Linear(1,1)        
        self.mode=mode
        self.padding=False

    def bdot(self,a, b):
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

    def forward(self, submitter_emb, reviewer_emb):
        x= self.bdot (submitter_emb, reviewer_emb)
        y= self.W(x.unsqueeze(dim=1))
        return (y.squeeze(dim=1))