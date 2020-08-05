import os
import torch

import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle

from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import entropy

from reviewer_expertise.models import Match_LR,Regression_Attention_Over_docs,Regression_Simple
from reviewer_expertise.utilities_model import get_train_test_data_from_hidden_representations,format_time, make_plot_training,prepare_data_bow,prepare_data,get_batch_eval, calculate_entropy_element, get_reviewer_entropy
from reviewer_expertise.utilities_model import get_test_data_from_hidden_representations_with_ids,Average
folder="./model_training/"
#model_path= "LDA-Match_LR-flag_attn-True-epochs-30-batch_size-16"
model_path="LDA-Match_LR-flag_attn-True-epochs-30-batch_size-16"
#model_path="LDA-Match_LR-flag_attn-True-epochs-3-batch_size-16"
params=model_path.split('-')
if not os.path.exists(model_path):
    print('model not trained yet')
else:
    rep=params[0]
    model_name= params[1]
    Attention_over_docs=bool(params[3])
    batch_size=int(params[-1])

    data_path = '../data_info/loaded_pickles_nips19/'
    """
    test_sub= torch.load('test_sub.pt')
    test_rev=torch.load('test_rev.pt')
    y_test=torch.load('y_test.pt')
    """

    train_sub,val_sub,test_sub, train_rev,val_rev,test_rev,y_train,y_val,y_test, reviewer_ids, submitter_ids= get_test_data_from_hidden_representations_with_ids(rep,data_path) 
    
    model = Match_LR(batch_size,25,25,4,Attention_over_docs)
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.MSELoss(reduction='sum')

    model.eval()
    y_true= np.zeros(len(y_test))
    y_pred= np.zeros(len(y_test))

    rev_dict={}

    with torch.no_grad():
        correct=0
        wrong=0
        loss_test=0
        for i in range(0, len(y_test)):
            if model_name=="Regression_Simple" and rep!="BOW":
                    prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(test_rev[i].unsqueeze(dim=0),dim=1).float()).float()
            elif rep=="BOW":
                prediction = model(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(dim=0).float()).float()
            elif rep=="LDA" and model_name=="Match_LR" and model.padding==False:
                prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(test_rev[i].unsqueeze(dim=0),dim=1).float()).float()[0]
            else:        
                prediction = model(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(dim=0).float()).float()        
            loss = criterion(prediction, y_test[i].argmax(dim=0).float())
            loss_test += loss.item()
            
            rev_id= reviewer_ids[i]
            submission_id= submitter_ids[i]
            if rev_id not in rev_dict:
                rev_dict[rev_id]= [loss.item()]
            else:
                rev_dict[rev_id].append(loss.item())

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

for rev_id in rev_dict:
    avg= Average(rev_dict[rev_id])
    rev_dict[rev_id]= avg

#maybe store this dict as pickle
with open("rev_dict_for_plot", "wb") as output_file:
    pickle.dump(rev_dict, output_file)
print('evalute done')



