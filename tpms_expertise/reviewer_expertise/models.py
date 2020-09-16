import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
import os

import pandas as pd


import pickle
import numpy as np
from scipy.stats import entropy
import os

torch.manual_seed(1)


class Attention_Module(nn.Module):
    def __init__(self, num_topics, attention_matrix_size, dropout):
        super(Attention_Module, self).__init__()
        self.num_topics = num_topics
        self.attention_matrix_size = attention_matrix_size
        self.W_Q = nn.Linear(self.num_topics, self.attention_matrix_size)
        self.W_K = nn.Linear(self.num_topics, self.attention_matrix_size)
        self.W_V = nn.Linear(self.num_topics, self.attention_matrix_size)
        self.padding = True
        self.dropout = dropout 

    
    def negative_attention(self,attention_scores):
        mod_e= torch.softmax(attention_scores,dim=2)
        neg_attention_scores= (1 - mod_e) * -1
        return torch.stack((attention_scores,neg_attention_scores))


    def forward(self, submitter_emb, reviewer_emb):
        Q = self.W_Q(submitter_emb)
        K = self.W_K(reviewer_emb)
        normalizing_factor = (math.sqrt(self.attention_matrix_size))
        e = (torch.bmm(Q.unsqueeze(dim=1), K.permute(0, 2, 1)) / normalizing_factor)
        # e=F.softmax(e)
        # total_e= self.negative_attention(e)
        # return total_e
        return torch.softmax(e,dim=2) 

#some of the attention scores should go to zero. but they are not going currently
class Multi_Head_Attention(nn.Module):
    def __init__(self, number_attention_heads, num_topics, attention_matrix_size, dropout):
        super(Multi_Head_Attention, self).__init__()
        self.number_attention_heads = number_attention_heads
        self.num_topics = num_topics
        self.attention_matrix_size = attention_matrix_size
        self.dropout = dropout
        # as the output from this multi-head should always be a fixed final size
        self.w_out = nn.Linear(
            self.number_attention_heads*self.num_topics, self.num_topics)

        self.heads = nn.ModuleList()
        for i in range(self.number_attention_heads):
            self.heads.append(Attention_Module(
                self.num_topics, self.attention_matrix_size, dropout))

    def forward(self, submitter_emb, reviewer_emb):
        x = []
        for i in range(self.number_attention_heads):
            x.append(self.heads[i](submitter_emb, reviewer_emb))
        z = torch.cat(x, dim=1)
        o = self.w_out(z)
        o = self.dropout(o)
        return o

class Rank_Net(nn.Module):
    def __init__(self, batch_size, paper_representation_dim, attention_matrix_size,
                device ):
        super(Rank_Net, self).__init__()
        
        self.padding = True
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=0.2)
        self.attention_matrix_size= attention_matrix_size
        self.attention_module= Attention_Module(25,self.attention_matrix_size,self.dropout)

    
    def get_rank_loss(self, attention_scores, actual_relevance):    
        actual_relevance= F.log_softmax(actual_relevance,dim=1)
        #make sure softmax has not been already taken for these attention scores. or does it matter at all?
        attention_scores= F.softmax(attention_scores,dim=2)
        listnet_score_div= F.kl_div(actual_relevance, attention_scores)
        return listnet_score_div

    #either write a function for dot product. or write your data in a way you can get actual relevance scores
    def forward(self, submitter_emb, reviewer_emb):
        attention_scores = self.attention_module(
            submitter_emb, reviewer_emb)
        return attention_scores

    



class Match_LR(nn.Module):
    def __init__(self, batch_size, paper_representation_dim, attention_matrix_size,
                 n_classes, device, attn_over_docs=True):
        super(Match_LR, self).__init__()
        self.attn_over_docs = attn_over_docs
        self.padding = False
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.submitter_emb_dim = paper_representation_dim
        self.attention_matrix_size = attention_matrix_size

        self.weights_add = Variable(torch.Tensor(
            self.submitter_emb_dim), requires_grad=True).to(device) 
        self.weights_diff = Variable(torch.Tensor(
            self.submitter_emb_dim), requires_grad=True).to(device) 
        self.weights_multi = Variable(torch.Tensor(
            self.submitter_emb_dim), requires_grad=True).to(device)  

        self.fc2 = nn.Linear(self.submitter_emb_dim, self.submitter_emb_dim)
        self.output = nn.Linear(128, 1)
        self.combined = nn.Linear(self.submitter_emb_dim, 128)

        self.number_heads = 1
        self.dropout = nn.Dropout(p=0.2)

        if self.attn_over_docs:
            self.padding = True
            if self.number_heads == 1:
                self.attention_module= Attention_Module(25,self.attention_matrix_size,self.dropout)#num_topics,attention_matrix_size
            else:
                self.attention_module = Multi_Head_Attention(
                    self.number_heads, 25, self.attention_matrix_size, self.dropout)
            self.init_weights()

    def init_weights(self):
        initrange = 4.0
        self.fc2.bias.data.fill_(0)
        self.weights_add.data.uniform_(-initrange, initrange)
        self.weights_diff.data.uniform_(-initrange, initrange)
        self.weights_multi.data.uniform_(-initrange, initrange)
    

    def get_reviewer_weighted(self, submitter_emb, reviewer_emb):        
        total_e = self.attention_module(
            submitter_emb, reviewer_emb)
    #    weighted_reviewer_emb= torch.sum(torch.matmul(total_e, reviewer_emb),dim=0).squeeze(dim=1)
        weighted_reviewer_emb= torch.matmul(total_e, reviewer_emb).squeeze(dim=1)
        return weighted_reviewer_emb
    
    #you get attention scores for a data sample. not out of thin air.remember that
    

    def forward(self, submitter_emb, reviewer_emb):
        if self.attn_over_docs == True:
            reviewer_emb = self.get_reviewer_weighted(
                submitter_emb, reviewer_emb)
        add = submitter_emb + self.fc2(reviewer_emb)
        diff = submitter_emb - self.fc2(reviewer_emb)
        multi = submitter_emb * (self.fc2(reviewer_emb))
        combo = self.combined(
            nn.Tanh()(self.weights_add * add) + nn.Tanh()(self.weights_diff * diff))
        op = 3*torch.sigmoid(self.output(combo))
        return op.view(-1)


    # def get_distance_attn_reviewer_sub(self, submitter_emb, reviewer_emb):
    #     mean_rep = torch.mean(reviewer_emb, dim=1)  # .squeeze()
    #     attn_rep = self.attention_module(
    #         submitter_emb, reviewer_emb)  # .squeeze()
    #     distance_paper_from_mean = F.kl_div(F.log_softmax(
    #         submitter_emb), F.softmax(mean_rep, dim=1), reduction='sum')
    #     distance_paper_from_attn = F.kl_div(F.log_softmax(
    #         submitter_emb), F.softmax(attn_rep, dim=1), reduction='sum')

    #     return distance_paper_from_mean, distance_paper_from_attn
    
    # def get_distance_from_max_3 (self, submitter_emb, reviewer_emb):
    #     x=torch.argsort(submitter_emb)[0][-3:]
    #     mean_rep = torch.mean(reviewer_emb, dim=1)  # .squeeze()
    #     attn_rep = self.attention_module(
    #         submitter_emb, reviewer_emb)  # .squeeze()

    #     distance_paper_from_mean = F.kl_div(F.log_softmax(
    #         submitter_emb[0][x]), F.softmax(mean_rep[0][x], dim=0), reduction='sum')
    #     distance_paper_from_attn = F.kl_div(F.log_softmax(
    #         submitter_emb[0][x]), F.softmax(attn_rep[0][x], dim=0), reduction='sum')


    #     return distance_paper_from_mean, distance_paper_from_attn

    #tells where the model is focussing
    def get_attention_scores(self, submitter_emb, reviewer_emb):
        attn_scores = self.attention_module(
            submitter_emb, reviewer_emb)

        #just calculate entropy of the attention scores above. no need for this kl divergence at the moment
        entropy_attn_scores= None

        return entropy_attn_scores

    

class MRRLoss(nn.Module):
# """ Mean Reciprocal Rank Loss """
    def __init__(self):
        super(MRRLoss, self).__init__()

    def forward(self, u, v):
        # u=torch.reshape(u,(-1,))
        # v=torch.reshape(v,(-1,))
        #cosine distance between all pair of embedding in u and v batches.
        # cos = nn.CosineSimilarity(dim=0)
        # distances=cos(u,v)
        cos = nn.CosineSimilarity(dim=2)
        distances=cos(u.unsqueeze(dim=1),v)
        # by construction the diagonal contains the correct elements
        correct_elements = torch.diag(cos(u.unsqueeze(dim=1),v),0).unsqueeze(-1)

        #maybe you can replace distances with the attention score?
        # number of elements ranked wrong.
        return torch.sum(distances < correct_elements)

def get_rank_loss(submitter_emb,reviewer_emb):
    rank_loss= MRRLoss()
    return rank_loss(submitter_emb,reviewer_emb)
