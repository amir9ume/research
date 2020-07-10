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
#%matplotlib inline
# import sys
# sys.path.insert(1, '../')
# import utilities

torch.manual_seed(1)



class Match_LR(nn.Module):
    def __init__(self,
                 submitter_emb_dim,
                 reviewer_emb_dim,
                 batch_size,
                 n_classes,):
        super(Match_LR, self).__init__()
        self.padding=False
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim = reviewer_emb_dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.weights_add = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True)#.cuda()
        self.weights_diff = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True)#.cuda()
        self.weights_multi = Variable(torch.Tensor(submitter_emb_dim), requires_grad=True)#.cuda()
        
        self.fc2 = nn.Linear(self.reviewer_emb_dim, self.reviewer_emb_dim)
        self.output = nn.Linear(128, 1)
        self.combined = nn.Linear(self.submitter_emb_dim, 128)
    
        self.init_weights()
        
    def init_weights(self):
        initrange = 4.0
        #self.weights.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.fill_(0)
        self.weights_add.data.uniform_(-initrange, initrange)
        self.weights_diff.data.uniform_(-initrange, initrange)
        self.weights_multi.data.uniform_(-initrange, initrange)
        
    def forward(self, submitter_emb, reviewer_emb):
        #submitter_f = self.fc_submitter(submitter_emb)
        #reviewer_f = self.fc_reviewer(reviewer_emb)
        add = submitter_emb + self.fc2(reviewer_emb)
        diff = submitter_emb - self.fc2(reviewer_emb)
        multi = submitter_emb * (self.fc2(reviewer_emb))
         
        combo = self.combined(nn.Tanh()(self.weights_add * add) + nn.Tanh()(self.weights_diff * diff) + nn.Tanh()(self.weights_multi * multi))
        combo= torch.sum(combo,dim=1)
        op = 3*torch.sigmoid(self.output(combo))
        return op.view(-1)
    


class Regression_Attention_Over_docs(nn.Module):
    def __init__(self,submitter_emb_dim, reviewer_emb_dim,
                 batch_size, n_classes,):
        super(Regression_Attention_Over_docs, self).__init__()
        
        self.padding=True
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim = reviewer_emb_dim
        
        #you need self.num_topics for now. You can call it in semantic embedding space  in future
        self.num_topics= 25
        self.attention_matrix_size= 20

        self.n_classes = n_classes
        self.batch_size = batch_size        
        
        self.W_Q= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_K= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_V = nn.Linear(self.num_topics,self.attention_matrix_size)
        
        self.batch_norm= torch.nn.BatchNorm1d(self.attention_matrix_size)

        self.w_submitted= nn.Linear(self.num_topics,self.attention_matrix_size)
        #for now treat it as a regression problem , instead of a class prediction problem
        self.w_out= nn.Linear(self.attention_matrix_size*2,1)
    #    self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.W_K.weight.data.uniform_(-initrange, initrange)
        # self.W_Q.weight.data.uniform_(-initrange, initrange)
        # self.W_V.weight.data.uniform_(-initrange, initrange)
        
    #    self.w_out.weight.data.uniform_(0,3)
        self.w_out.bias.data.fill_(1)


    #this tanh activation could be squashing everything. And hence no gradient flow beyond it
    def mapping_to_target_range( self, x, target_min=-0.5, target_max=3.41 ) :
        x02 = torch.tanh(x) + 1 # x in range(0,2)
        scale = ( target_max-target_min )/2.
        return  x02 * scale + target_min

    def scaling_different(self, x):
        x -= x.min()
        x /= x.max()
        return x * 3

    def scale_sigmoid(self,x):
        return 3*torch.sigmoid(x)

    #forward is actually defining the equation
    def forward(self, submitter_emb, reviewer_emb):    

        Q= self.W_Q(submitter_emb)
        Q=self.batch_norm(Q)
        K= self.W_K(reviewer_emb)
        V= self.W_V(reviewer_emb)
        

        #matrix multiplication QxK.T , same as in the paper
        normalizing_factor= (math.sqrt(self.attention_matrix_size))
        e= torch.bmm(Q.unsqueeze(dim=1),K.permute(0,2,1))/normalizing_factor
        softm= torch.nn.Softmax(dim=2)
        a=softm(e)
        attn_output= torch.bmm(a,V)
        x= torch.sum(attn_output,dim=1)
        x=self.batch_norm(x)
        combine= torch.cat((x,self.w_submitted(submitter_emb)),dim=1 )        
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
        #do the dot product thing you were doing before
        # if self.mode=="mean":
        #     rev= torch.mean(reviewer_emb,dim=1)
        # elif self.mode=="max":
        #     rev= torch.max(reviewer_emb,dim=1)
            
        x= self.bdot (submitter_emb, reviewer_emb)

        y= self.W(x.unsqueeze(dim=1))
        return (y.squeeze(dim=1))