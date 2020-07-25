import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


import pandas as pd 
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay
#load data
import pickle
import numpy as np
from scipy.stats import entropy
import os
#%matplotlib inline
import sys
sys.path.insert(1, '../')

import utilities

from reviewer_expertise.pytorchtools import EarlyStopping


torch.manual_seed(1)

from reviewer_expertise.models import Match_LR,Regression_Attention_Over_docs,Regression_Simple
from reviewer_expertise.utilities_model import format_time, make_plot_training
r= os.getcwd()
print(os.getcwd())
data_path = '../data_info/loaded_pickles_nips19/'

flag_early_stopping=False

rep='LDA'
#rep='BOW'
model_name="Regression_Attention_Over_docs"
#model_name="Match_LR"
#model_name= "Regression_Simple"

if rep=='BOW':
    folder='reviewer_expertise/summary_models/'
    reviewer_representation= pickle.load( open( folder+ 'ac_reviewer_bag_words_tensor_nips19', "rb" ))
    paper_representation= pickle.load(open(folder+'submitted_paper_bag_words_tensor_nips19','rb'))
else:
    reviewer_representation = pickle.load( open( data_path+ 'dict_all_reviewer_lda_vectors.pickle', "rb" ))
    paper_representation = pickle.load( open( data_path + 'dict_paper_lda_vectors.pickle', "rb" ))


#df= pd.read_csv('../../neurips19_anon/anon_bids_file')

bds_path='~/arcopy/neurips19_anon/anon_bids_file'
bds_path= '~/arcopy/workingAmir/data_info/loaded_pickles_nips19/bids_ac_anon_nips19'
df= pd.read_csv(bds_path)
#df= utilities.get_equal_sized_data(df)
df=df.sample(frac=1)
size= len(df.index)

print('data size is ', len(df.index))
print(df.sample(3))

def prepare_data_bow(submitter,reviewer,df,gpu_flag=False):
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
            train_data_sub.append(submitter[pid_curr])#.cuda()
            train_data_rev.append(torch.mean(torch.stack(reviewer[rev_curr]),dim=0))#.cuda()
            idx = int(df.iloc[i]['bid'])
            temp = torch.LongTensor([0, 0, 0, 0])#.cuda()
            for i in range(4):
                if i == idx:
                    temp[i] = 1
            labels.append(temp)
            submitter_ids.append(df.iloc[i]['pid'])
            reviewer_ids.append(df.iloc[i]['anon_id'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids



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

def get_batch_eval(paper_emb, rev_emb, trg_value, idx, batch_size,padding):
    paper_lines = Variable(torch.stack(paper_emb[idx:idx+batch_size]), requires_grad=True)#.permute(1,0)
    rev_papers=rev_emb[idx:idx+batch_size]
    if padding ==True:
        reviewer_papers = pad_sequence(rev_papers, batch_first=True, padding_value=0) # padding as different reviewers can have different number of papers
    elif (model_name=="Regression_Simple"  or model_name=="Match_LR")and rep !="BOW":
        reviewer_papers = torch.stack([torch.mean(r,dim=0) for r in rev_papers])
    elif rep=="BOW":
        reviewer_papers = torch.stack(rev_papers)
    trg = torch.stack(trg_value[idx:idx+batch_size]).squeeze()
    
    return paper_lines.float(), reviewer_papers.float(), trg    


if rep=="BOW":
    data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data_bow(paper_representation, reviewer_representation, df)
else:
    data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(paper_representation, reviewer_representation, df)

train_length = int(0.7*len(data_sub))
val_length= int(.15*len(data_sub))
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


batch_size=16
patience=5

flag_attn= True

if model_name=="Match_LR": #Match LR is Meghana's model
    model= Match_LR(25,25,batch_size,4)
elif model_name=="Regression_Attention_Over_docs":
    model=Regression_Attention_Over_docs(batch_size,4,flag_attn,True)
elif model_name=="Regression_Simple":
    model= Regression_Simple(25,25,batch_size,4,"mean")



criterion = torch.nn.MSELoss(reduction='sum') 
 
if model_name in ["Regression_Attention_Over_docs"]:
   # learning_rate = 1e-5
   # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001,momentum=0.9)
    epochs=30
else:
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001,momentum=0.9)
    epochs=80



#TRAINING MODULE
losses= []
training_stats = []
print('Model Name ',model_name, ' -- representation used --', rep)

PATH=rep+'_'+model_name+'_flag_attn_'+str(flag_attn)+'_epochs_'+str(epochs)

for e_num in range(epochs):
    loss_ep = 0
    correct=0
    wrong=0
    # Measure how long the training epoch takes.
    t0 = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model.train()
    for i in range(0, len(y_train), batch_size):
        mini_batch_submitted_paper, mini_batch_reviewer_paper, y = get_batch_eval(train_sub, train_rev, y_train, i, batch_size,model.padding) 
        if len(y.shape)>1:
            optimizer.zero_grad()
            if model_name=="Match_LR":
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            else:    
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            loss = criterion(prediction, y.argmax(dim=1).float())
            loss_ep += loss.item()
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()

            class_label = torch.round(prediction).squeeze(dim=0)#.argmax(dim=1)
            trg_label = y.argmax(dim=1)
            correct = correct + torch.sum(class_label==trg_label).item()

                    
    losses.append(loss_ep/len(y_train))
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("Epoch:", e_num, " Loss:", losses[-1], ": Train Accuracy:", correct/len(y_train), "  Training epcoh took: {:}".format(training_time))

    # #Validate after each epoch
    model.eval()
    with torch.no_grad():
        correct=0
        wrong=0
        loss_val=0
        for i in range(0, len(y_val)):
            if model_name=="Regression_Simple" and rep!="BOW":
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), torch.mean(val_rev[i].unsqueeze(dim=0),dim=1).float()).float()
            elif rep=="BOW":
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), val_rev[i].unsqueeze(dim=0).float()).float()
            elif rep=="LDA" and model_name=="Match_LR":
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), torch.mean(val_rev[i].unsqueeze(dim=0),dim=1).float()).float()[0]
            else:        
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), val_rev[i].unsqueeze(dim=0).float()).float()
            loss = criterion(prediction, y_val[i].argmax(dim=0).float())
            loss_val += loss.item()
            
            class_label = torch.round(prediction).squeeze(dim=0)#.argmax(dim=1)
            trg_label = y_val[i].argmax(dim=0)
            if rep=="LDA" and model_name=="Match_LR":
                if class_label==trg_label:
                    correct= correct+1
            else:      
                correct = correct + torch.sum(class_label==trg_label).item()
            
        print("Validation Loss:", loss_val/len(y_val), ": Validation Accuracy:", correct/len(y_val))
        print("========================================================")
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if flag_early_stopping:
            early_stopping(loss_val/len(y_val), model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load('checkpoint.pt'))
            
        training_stats.append(
        {
            'epoch': e_num ,
            'Training Loss': losses[-1],
            'Valid. Loss': loss_val/len(y_val),
            'Valid. Accur.': correct/len(y_val),
            'Training Time': training_time
            }
    )

PATH=rep+'_'+model_name+'_flag_attn_'+str(flag_attn)+'_epochs_'+str(epochs)
torch.save(model.state_dict(), PATH)

model = Regression_Attention_Over_docs(batch_size,4,flag_attn,True)
model.load_state_dict(torch.load(PATH))


# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

print(df_stats)
make_plot_training(df_stats,epochs)


# #Test after each epoch
model.eval()
y_true= np.zeros(len(y_test))
y_pred= np.zeros(len(y_test))
with torch.no_grad():
    correct=0
    wrong=0
    loss_test=0
    for i in range(0, len(y_test)):
        if model_name=="Regression_Simple" and rep!="BOW":
                prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(test_rev[i].unsqueeze(dim=0),dim=1).float()).float()
        elif rep=="BOW":
            prediction = model(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(dim=0).float()).float()
        elif rep=="LDA" and model_name=="Match_LR":
            prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(test_rev[i].unsqueeze(dim=0),dim=1).float()).float()[0]
        else:        
            prediction = model(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(dim=0).float()).float()        
        loss = criterion(prediction, y_test[i].argmax(dim=0).float())
        loss_test += loss.item()
        
        class_label = torch.round(prediction).squeeze(dim=0)#.argmax(dim=1)
        trg_label = y_test[i].argmax(dim=0)
        
        y_true[i]=class_label
        y_pred[i]= trg_label
        
        if class_label==trg_label:
            correct= correct+1
        else:
            correct = correct + torch.sum(class_label==trg_label).item()
        
    print("Test Loss:", loss_test/len(y_test), ": Test Accuracy:", correct/len(y_test))
    print("========================================================")
    cm=confusion_matrix(y_true,y_pred)
    print(cm)





