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
from datetime import datetime
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import entropy

import os
# import optuna
# from sklearn.externals import joblib
import sys

from reviewer_expertise.pytorchtools import EarlyStopping
from reviewer_expertise.models import Match_LR,Regression_Attention_Over_docs,Regression_Simple
from reviewer_expertise.utilities_model import get_train_test_data_from_hidden_representations,format_time, make_plot_training,prepare_data_bow,prepare_data,get_batch_eval, calculate_entropy_element, get_reviewer_entropy

import argparse
parser=argparse.ArgumentParser()
#this arhgument shall store , location of the data folder
parser.add_argument('--path',type=str,help='path to the $slurm tmpdir is passed')
parser.add_argument('--save_model',type=str,help='path for saving the results')
args=parser.parse_args()
print('slurm tmpdir loc: ', args.path)

torch.manual_seed(1)
r= os.getcwd()
print(os.getcwd())

#data_path = '../../workingAmir/data_info/loaded_pickles_nips19/'
data_path=args.path
curr_time_is=str(datetime.today().strftime('%Y-%m-%d'))
#saved_models_folder=curr_time_is+'/'
saved_models_folder= args.save_model
os.mkdir(saved_models_folder)


flag_early_stopping=False

rep='LDA'
#rep='BOW'
#model_name="Regression_Attention_Over_docs"
model_name="Match_LR"
#model_name= "Regression_Simple"

device="cuda" if torch.cuda.is_available() else "cpu"
train_sub,val_sub,test_sub, train_rev,val_rev,test_rev,y_train,y_val,y_test= get_train_test_data_from_hidden_representations(rep,data_path,device)

# save_folder=''
# #saving test data, for later evaluation
# torch.save(test_sub, save_folder+'test_sub.pt')
# torch.save(test_rev, save_folder'test_rev.pt')
# torch.save(y_test, save_folder'y_test.pt')


patience=5

Attention_over_docs=True
kl_div_flag=True

#trial.suggest_categorical('batch_size',[ 32,64])
#trial.suggest_int('epochs',25,55) . trial.suggest_loguniform('lr', 1e-5, 1e-2)
#def train_expertise_model(trial):
cfg = { 'device' : "cuda" if torch.cuda.is_available() else "cpu",
        'batch_size' :64,
        'epochs' : 2,
        'seed' : 0,
        'log_interval' : 100,
        'save_model' : False,
        'weight_mse': 1 ,
        'lr' :0.009801 ,          
        'momentum': 0.97403,
        'optimizer': torch.optim.Adam}

epochs= cfg['epochs']
batch_size=cfg['batch_size']

if model_name=="Match_LR": #Match LR is Meghana's model
    model= Match_LR(batch_size,25,25,4,Attention_over_docs)
elif model_name=="Regression_Attention_Over_docs":    
    model=Regression_Attention_Over_docs(batch_size,4,Attention_over_docs,True)
elif model_name=="Regression_Simple":
    model= Regression_Simple(25,25,batch_size,4,"mean")


criterion = torch.nn.MSELoss(reduction='sum') 
optimizer=cfg['optimizer'](model.parameters(), lr=cfg['lr'])


device=cfg['device']
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

#TRAINING MODULE
losses= []
training_stats = []
print('Model Name ',model_name, ' -- representation used --', rep, 'Attention over docs: ',str(Attention_over_docs))
print('KL divergence flag: ',kl_div_flag)

for e_num in range(epochs):
    loss_ep = 0
    correct=0
    wrong=0
    # Measure how long the training epoch takes.
    t0 = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model.train()
    for i in range(0, len(y_train), batch_size):
        mini_batch_submitted_paper, mini_batch_reviewer_paper, y = get_batch_eval(model_name,train_sub, train_rev, y_train, i, batch_size,model.padding,rep) 
        if len(y.shape)>1:
            optimizer.zero_grad()
            if model_name=="Match_LR":
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            else:    
                prediction = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
            weighted_averages_reviewers= model.get_reviewer_weighted(mini_batch_submitted_paper,mini_batch_reviewer_paper)
            weighted_averages_reviewers= F.softmax(weighted_averages_reviewers,dim=1)

            loss1 = criterion(prediction, y.argmax(dim=1).float())
            loss_ep += loss1.item()
            
            if kl_div_flag:
                loss2=F.kl_div(F.log_softmax(mini_batch_submitted_paper,dim=1), weighted_averages_reviewers,reduction= 'sum')
            else:
                loss2= torch.zeros_like(loss1)
            weight_mse= cfg['weight_mse']
            loss= weight_mse* loss1 + loss2    
            loss_ep= weight_mse* loss_ep + loss2.item()

        #    d1, d2= model.get_distance_attn_reviewer_sub(mini_batch_submitted_paper, mini_batch_reviewer_paper)

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
            #model.load_state_dict(torch.load('checkpoint.pt'))

        training_stats.append(
        {
            'epoch': e_num ,
            'Training Loss': losses[-1],
            'Valid. Loss': loss_val/len(y_val),
            'Valid. Accur.': correct/len(y_val),
            'Training Time': training_time
            }
    )

PATH=rep+'-'+model_name+'-flag_attn-'+str(Attention_over_docs)+'-epochs-'+str(epochs)+'-batch_size-'+str(batch_size)+"-KL-"+str(kl_div_flag)
    
torch.save(model.state_dict(), saved_models_folder+'/'+PATH)

    # #test set
    # model.eval()
    # with torch.no_grad():
    #     correct=0
    #     wrong=0
    #     loss_test=0
    #     rev_entropy_val=0
    #     for i in range(0, len(y_test)):
    #         if model_name=="Regression_Simple" and rep!="BOW":
    #             prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(test_rev[i].unsqueeze(dim=0),dim=1).float()).float()
    #         elif rep=="BOW":
    #             prediction = model(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(dim=0).float()).float()
    #         elif rep=="LDA" and model_name=="Match_LR" and model.padding==False:
    #             prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(test_rev[i].unsqueeze(dim=0),dim=1).float()).float()[0]
    #         else:        
    #             prediction = model(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(dim=0).float()).float()
    #         loss = criterion(prediction, y_test[i].argmax(dim=0).float())
    #         loss_val += loss.item()
            
    #         rev_entropy= get_reviewer_entropy(test_rev[i].unsqueeze(dim=0).float())
           
    #         class_label = torch.round(prediction).squeeze(dim=0)
    #         trg_label = y_test[i].argmax(dim=0)
    #         if rep=="LDA" and model_name=="Match_LR":
    #             if class_label==trg_label:
    #                 correct= correct+1
    #         else:      
    #             correct = correct + torch.sum(class_label==trg_label).item()


    # return correct/len(y_test)
    
# if __name__ == '__main__':

#     sampler = optuna.samplers.TPESampler()        
#     study = optuna.create_study(sampler=sampler, direction='maximize')
    
#     study.optimize(func=train_expertise_model, n_trials=1)
#     joblib.dump(study, saved_models_folder+'/'+'optuna_5_no_trial')

#     df = study.trials_dataframe()
#     print(df.head(4))

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial)