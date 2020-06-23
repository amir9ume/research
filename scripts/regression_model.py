import torch
import torch.nn as nn
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
import sys
sys.path.insert(1, '../')
import utilities


r= os.getcwd()
print(os.getcwd())
data_path = '../../../../dat/scratch/rectified2/'


reviewer_representation = pickle.load( open( data_path+ 'dict_reviewer_lda_vectors.pickle', "rb" ))
paper_representation = pickle.load( open( data_path + 'dict_paper_lda_vectors.pickle', "rb" ))


#df= pd.read_csv('../../neurips19_anon/anon_bids_file')

bds_path='~/arcopy/neurips19_anon/anon_bids_file'
df= utilities.get_bids_data_filtered(bds_path)


size= len(df.index)
print('data size is ', size)
#df= df[:int(0.1 * size)]
print(df.sample(4))

def prepare_data(submitter, reviewer, df, gpu_flag=False):
    train_data_sub = []
    train_data_rev = []
    submit = submitter.keys()
    submitter_ids = []
    reviewer_ids = []
    rev = reviewer.keys()
    labels = []
    for i in range(len(df)):
        pid_curr= str(df.iloc[i]['pid'])
        rev_curr=    str(df.iloc[i]['anon_id']) 
        if pid_curr  in submit and rev_curr in reviewer:
            train_data_sub.append(torch.tensor(submitter[pid_curr],requires_grad=True))#.cuda()
            train_data_rev.append(torch.tensor(reviewer[rev_curr], requires_grad=True))#.cuda()
            idx = int(df.iloc[i]['bid'])
            temp = torch.LongTensor([0, 0, 0, 0])#.cuda()
            for i in range(4):
                if i == idx:
                    temp[i] = 1
            labels.append(temp)
            submitter_ids.append(df.iloc[i]['pid'])
            reviewer_ids.append(df.iloc[i]['anon_id'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids

def get_batch_eval(paper_emb, rev_emb, trg_value, idx, batch_size):
    paper_lines = Variable(torch.stack(paper_emb[idx:idx+batch_size]).squeeze(), requires_grad=True)
    review_emb = Variable(torch.stack(rev_emb[idx:idx+batch_size]).squeeze(), requires_grad=True)
    trg = torch.stack(trg_value[idx:idx+batch_size]).squeeze()
    return paper_lines, review_emb, trg    



class Match_Classify(nn.Module):
    def __init__(self,submitter_emb_dim, reviewer_emb_dim,
                 batch_size, n_classes,):
        super(Match_Classify, self).__init__()
        
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim = reviewer_emb_dim
        
        self.n_classes = n_classes
        self.batch_size = batch_size        
        
        self.combined = nn.Linear(self.submitter_emb_dim, 25)
        self.out= nn.Linear(25, n_classes)

        
        #forward is actually defining the equation
        #U_i, P_j combination
    def forward(self, submitter_emb, reviewer_emb):
        combine= self.combined((submitter_emb + reviewer_emb).float())      
        
        #remember to bound your softmax between [0 to 3]. pure softmax just puts it between [0 to 1]
        op = F.softmax(self.out(combine))
        return op


#get your data ready here
#wonder what the shapes of submitter hid and reviewer hid can be::
data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(paper_representation, reviewer_representation, df)
train_ratio = int(0.8*len(data_sub))
test_ratio = len(data_sub) - train_ratio

train_sub = data_sub[:train_ratio]
test_sub = data_sub[train_ratio:]

train_rev = data_rev[:train_ratio]
test_rev = data_rev[train_ratio:]

y_train = data_y[:train_ratio]
y_test = data_y[train_ratio:]

batch_size=25
model = Match_Classify(25,25,batch_size,4) 
  
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 
  

epochs=100

losses= []
for e_num in range(epochs):
    loss_ep = 0
    for i in range(0, len(y_train), batch_size):
        tr_sub, tr_rev, y = get_batch_eval(train_sub, train_rev, y_train, i, batch_size) 
        optimizer.zero_grad()
        prediction = model(tr_sub, tr_rev).float()
       # loss = criterion(prediction, y.argmax(dim=1))
        loss = criterion(prediction, y.float())
        loss_ep += loss.item()
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()
    losses.append(loss_ep/batch_size)
    print("Epoch:", e_num, " Loss:", losses[-1])


#code for evaluation part goes here


new_var = Variable(torch.Tensor([[4.0]])) 
pred = model(new_var) 
print("predict (after training)", 4, model(new_var).item())