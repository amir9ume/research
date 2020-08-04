import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import datetime

from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import entropy

import os
#%matplotlib inline
import sys
sys.path.insert(1, '../')
import utilities

from reviewer_expertise.pytorchtools import EarlyStopping
from reviewer_expertise.models import Match_LR,Regression_Attention_Over_docs,Regression_Simple
from reviewer_expertise.utilities_model import get_train_test_data_from_hidden_representations,format_time, make_plot_training,prepare_data_bow,prepare_data,get_batch_eval, calculate_entropy_element, get_reviewer_entropy

torch.manual_seed(1)
r= os.getcwd()
print(os.getcwd())
data_path = '../data_info/loaded_pickles_nips19/'

flag_early_stopping=False

rep='LDA'
#rep='BOW'
model_name="Regression_Attention_Over_docs"
#model_name="Match_LR"
#model_name= "Regression_Simple"


train_sub,val_sub,test_sub, train_rev,val_rev,test_rev,y_train,y_val,y_test= get_train_test_data_from_hidden_representations(rep,data_path)

#saving test data, for later evaluation
torch.save(test_sub, 'test_sub.pt')
torch.save(test_rev, 'test_rev.pt')
torch.save(y_test, 'y_test.pt')

batch_size=16
patience=5

Attention_over_docs=True

if model_name=="Match_LR": #Match LR is Meghana's model
    model= Match_LR(batch_size,25,25,4,Attention_over_docs)
elif model_name=="Regression_Attention_Over_docs":
    model=Regression_Attention_Over_docs(batch_size,4,Attention_over_docs,True)
elif model_name=="Regression_Simple":
    model= Regression_Simple(25,25,batch_size,4,"mean")



criterion = torch.nn.MSELoss(reduction='sum') 
 
if model_name in ["Regression_Attention_Over_docs"]:
  #  learning_rate = 1e-5
  #  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001,momentum=0.9)
    epochs=3
else:
   # learning_rate = 1e-5
   # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001,momentum=0.9)
    epochs=30



#TRAINING MODULE
losses= []
training_stats = []
print('Model Name ',model_name, ' -- representation used --', rep)
use_pre_trained_attention=False
if use_pre_trained_attention:
    path_pre_trained=""
    #pass pre-trained weights to a particular layer. see how to do this.
    torch.load()

for e_num in range(epochs):
    loss_ep = 0
    correct=0
    wrong=0
    # Measure how long the training epoch takes.
    t0 = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model.train()
    for i in range(0, len(y_train), batch_size):
        mini_batch_submitted_paper, mini_batch_reviewer_paper, y = get_batch_eval(model_name,train_sub, train_rev, y_train, i, batch_size,model.padding) 
        if len(y.shape)>1:
            optimizer.zero_grad()
            if model_name=="Match_LR":
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            else:    
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            loss = criterion(prediction, y.argmax(dim=1).float())
            loss_ep += loss.item()
            loss.backward()         
            optimizer.step()
            class_label = torch.round(prediction).squeeze(dim=0)
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
        rev_entropy_val=0
        for i in range(0, len(y_val)):
            if model_name=="Regression_Simple" and rep!="BOW":
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), torch.mean(val_rev[i].unsqueeze(dim=0),dim=1).float()).float()
            elif rep=="BOW":
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), val_rev[i].unsqueeze(dim=0).float()).float()
            elif rep=="LDA" and model_name=="Match_LR" and model.padding==False:
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), torch.mean(val_rev[i].unsqueeze(dim=0),dim=1).float()).float()[0]
            else:        
                prediction = model(val_sub[i].unsqueeze(dim=0).float(), val_rev[i].unsqueeze(dim=0).float()).float()
            loss = criterion(prediction, y_val[i].argmax(dim=0).float())
            loss_val += loss.item()
            
            rev_entropy= get_reviewer_entropy(val_rev[i].unsqueeze(dim=0).float())
            rev_entropy_val+= rev_entropy

            class_label = torch.round(prediction).squeeze(dim=0)
            trg_label = y_val[i].argmax(dim=0)
            if rep=="LDA" and model_name=="Match_LR":
                if class_label==trg_label:
                    correct= correct+1
            else:      
                correct = correct + torch.sum(class_label==trg_label).item()
           
        print("Validation Loss:", loss_val/len(y_val), ": Validation Accuracy:", correct/len(y_val))
        print('reviewer entropy ', rev_entropy_val/len(y_val))
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

        #print entropy of the reviewer
        
        
        training_stats.append(
        {
            'epoch': e_num ,
            'Training Loss': losses[-1],
            'Valid. Loss': loss_val/len(y_val),
            'Valid. Accur.': correct/len(y_val),
            'Training Time': training_time
            }
    )

PATH=rep+'-'+model_name+'-flag_attn-'+str(Attention_over_docs)+'-epochs-'+str(epochs)+'-batch_size-'+str(batch_size)
torch.save(model.state_dict(), PATH)

# model = Match_LR(batch_size,25,25,4,Attention_over_docs)
# model.load_state_dict(torch.load(PATH))


# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

print(df_stats)
make_plot_training(df_stats,epochs)



