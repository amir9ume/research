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
import sys
sys.path.insert(1, '../')
import utilities

torch.manual_seed(1)


r= os.getcwd()
print(os.getcwd())
data_path = '../data_info/loaded_pickles_nips19/'


reviewer_representation = pickle.load( open( data_path+ 'dict_all_reviewer_lda_vectors.pickle', "rb" ))
paper_representation = pickle.load( open( data_path + 'dict_paper_lda_vectors.pickle', "rb" ))


#df= pd.read_csv('../../neurips19_anon/anon_bids_file')

bds_path='~/arcopy/neurips19_anon/anon_bids_file'
df= utilities.get_bids_data_filtered(bds_path)
# df= pd.read_csv('~/arcopy/neurips19_anon/fake_small_data')
df= utilities.get_equal_sized_data(df)
size= len(df.index)

#df= df[:int(0.01 * size)]
print('data size is ', len(df.index))
print(df.sample(3))

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
    paper_lines = Variable(torch.stack(paper_emb[idx:idx+batch_size]), requires_grad=True)#.permute(1,0)
    rev_papers=rev_emb[idx:idx+batch_size]
    reviewer_papers_padded = pad_sequence(rev_papers, batch_first=True, padding_value=0) # padding as different reviewers can have different number of papers
    trg = torch.stack(trg_value[idx:idx+batch_size]).squeeze()
    return paper_lines.float(), reviewer_papers_padded.float(), trg    




class Match_Regression(nn.Module):
    def __init__(self,submitter_emb_dim, reviewer_emb_dim,
                 batch_size, n_classes,):
        super(Match_Regression, self).__init__()
        
        self.submitter_emb_dim = submitter_emb_dim
        self.reviewer_emb_dim = reviewer_emb_dim
        
        #you need self.num_topics for now. You can call it in semantic embedding space  in future
        self.num_topics= 25
        self.attention_matrix_size= 10

        self.n_classes = n_classes
        self.batch_size = batch_size        
        
        self.W_Q= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_K= nn.Linear(self.num_topics,self.attention_matrix_size )
        self.W_V = nn.Linear(self.num_topics,self.attention_matrix_size)
        
        self.w_submitted= nn.Linear(self.num_topics,self.attention_matrix_size)
        #for now treat it as a regression problem , instead of a class prediction problem
        self.w_out= nn.Linear(self.attention_matrix_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W_K.weight.data.uniform_(-initrange, initrange)
        self.W_Q.weight.data.uniform_(-initrange, initrange)
        self.W_V.weight.data.uniform_(-initrange, initrange)
        
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

    #forward is actually defining the equation
    def forward(self, submitter_emb, reviewer_emb):    

        Q= self.W_Q(submitter_emb)
        K= self.W_K(reviewer_emb)
        V= self.W_V(reviewer_emb)

        #matrix multiplication QxK.T , same as in the paper
        normalizing_factor= (math.sqrt(self.attention_matrix_size))
        s= torch.bmm(Q.unsqueeze(dim=1),K.permute(0,2,1))/normalizing_factor

        softm= torch.nn.Softmax(dim=2)
        z=softm(s)
        attn_output= torch.bmm(z,V)

        combine= torch.cat((attn_output,self.w_submitted(submitter_emb).unsqueeze(dim=1)),dim=1 )
        aggregate= torch.sum(combine, dim=1) #kind of aggregating information over the concatenated dimension
        out= self.w_out(aggregate)
        out= self.mapping_to_target_range(out)
        return out.squeeze(dim=1)

#wonder what the shapes of submitter hid and reviewer hid can be::
data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(paper_representation, reviewer_representation, df)
train_length = int(0.7*len(data_sub))
val_length= int(0*len(data_sub))
test_length = len(data_sub) - train_length - val_length 

train_sub = data_sub[:train_length]
val_sub= data_sub[train_length: (train_length+val_length)]
test_sub = data_sub[train_length+val_length:]

train_rev = data_rev[:train_length]
val_rev= data_rev[train_length: (train_length+val_length)]
test_rev = data_rev[train_length+val_length:]

y_train = data_y[:train_length]
y_val= data_y[train_length:(train_length+val_length)]
y_test = data_y[train_length+val_length:]



batch_size=32
model = Match_Regression(25,25,batch_size,4) 
  
criterion = torch.nn.MSELoss(size_average = False) 
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.001,momentum=0.9) 
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

epochs=80
#TRAINING MODULE
losses= []
for e_num in range(epochs):
    loss_ep = 0
    correct=0
    wrong=0
    for i in range(0, len(y_train), batch_size):
        mini_batch_submitted_paper, mini_batch_reviewer_paper, y = get_batch_eval(train_sub, train_rev, y_train, i, batch_size) 
        if len(y.shape)>1:
            optimizer.zero_grad()
            prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            loss = criterion(prediction, y.argmax(dim=1).float())
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()

            class_label = torch.round(prediction).squeeze(dim=0)#.argmax(dim=1)
            trg_label = y.argmax(dim=1)
            correct = correct + torch.sum(class_label==trg_label).item()

            #checking gradients
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name , '===========\gradient', param.grad)
            #     #    print(name, '=========\weight',param)
                    
            losses.append(loss_ep/batch_size)
    print("Epoch:", e_num, " Loss:", losses[-1], ": Train Accuracy:", correct/len(y_train))

    # #Validate after each epoch
    # model.eval()
    # with torch.no_grad():
    #     correct=0
    #     wrong=0
    #     loss_val=0
    #     for i in range(0, len(y_val)):
    #         prediction = model(val_sub[i].unsqueeze(dim=1).float(), val_rev[i].T.float()).float()
    #         loss = criterion(prediction, y_val[i].float())
    #         loss_val += loss.item()
        
    #         class_label = prediction.argmax(dim=1)
    #         trg_label = y_val[i].argmax()
    #         if class_label == trg_label:
    #             correct += 1
    #         else:
    #             wrong += 1

    #     print("Validation Loss:", loss_val/len(y_val), ": Validation Accuracy:", correct/len(y_val))
    #     model.train()


#code for test set evaluation part 
with torch.no_grad():
    model.eval()
    correct=0
    wrong=0
    loss_test=0
    for i in range(0, len(y_test)):
        prediction = model(test_sub[i].unsqueeze(dim=1).float(), test_rev[i].T.float()).float()
        loss = criterion(prediction, y_test[i].float())
        loss_test += loss.item()
    
        class_label = torch.round(prediction).squeeze(dim=0)
        trg_label = y_test[i].argmax()
        if class_label == trg_label:
            correct += 1
        else:
            wrong += 1

    print("Test Loss:", loss_test/len(y_test), ": Test Accuracy:", correct/len(y_test))




