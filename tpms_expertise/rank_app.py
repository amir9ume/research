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

import os
import optuna
from sklearn.externals import joblib
import sys

from reviewer_expertise.models import Match_LR, Rank_Net
from reviewer_expertise.other_models import Regression_Attention_Over_docs, Regression_Simple , get_actual_relevance_using_cosine, get_synthetic_relevance_values 
from reviewer_expertise.utilities_model import get_train_test_data_from_hidden_representations, format_time, make_plot_training, prepare_data_bow, prepare_data, get_batch_eval, calculate_entropy_element, get_reviewer_entropy, pad_sequence_my

import argparse
parser = argparse.ArgumentParser()
# this arhgument shall store , location of the data folder
parser.add_argument('--path', type=str,
                    help='path to the $slurm tmpdir is passed')
parser.add_argument('--save_model', type=str,
                    help='path for saving the results')
parser.add_argument('--epochs', type=int, help='number of epochs')
args = parser.parse_args()
print('slurm tmpdir loc: ', args.path)

torch.manual_seed(1)
r = os.getcwd()
print(os.getcwd())

#data_path = '../../workingAmir/data_info/loaded_pickles_nips19/'
data_path = args.path
curr_time_is = str(datetime.today().strftime('%Y-%m-%d'))
# saved_models_folder=curr_time_is+'/'
saved_models_folder = args.save_model
if os.path.exists(saved_models_folder):
    pass
else:
    os.mkdir(saved_models_folder)


rep = 'LDA'
model_name = "Rank_Net"

device = "cuda" if torch.cuda.is_available() else "cpu"
train_sub, val_sub, test_sub, train_rev, val_rev, test_rev, y_train, y_val, y_test = get_train_test_data_from_hidden_representations(
    rep, data_path, device, 0.5)

# save_folder=''
# #saving test data, for later evaluation
# torch.save(test_sub, save_folder+'test_sub.pt')
# torch.save(test_rev, save_folder'test_rev.pt')
# torch.save(y_test, save_folder'y_test.pt')

Attention_over_docs = True
kl_div_flag = False

#trial.suggest_categorical('batch_size',[ 32,64])
#trial.suggest_int('epochs',25,55) . trial.suggest_loguniform('lr', 1e-5, 1e-2)
#trial.suggest_int('hidden_size',5,50)


def get_rank_loss(self, attention_scores, actual_relevance):    
        actual_relevance= F.log_softmax(actual_relevance,dim=1)
        #make sure softmax has not been already taken for these attention scores. or does it matter at all?
        attention_scores= F.softmax(attention_scores,dim=2)
        listnet_score_div= F.kl_div(actual_relevance, attention_scores)
        return listnet_score_div


def train_expertise_model(trial):
    cfg = {'device': "cuda" if torch.cuda.is_available() else "cpu",
           'batch_size': 32,
           'epochs': args.epochs,
           'seed': 0,
           'log_interval': 100,
           'save_model': False,
           'weight_mse': 1,
           'lr': 0.009801,
           'hidden_size':24,
           'momentum': 0.9,
           'optimizer': torch.optim.SGD}
    
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    device = cfg['device']

    if model_name == "Rank_Net":  # Match LR is Meghana's model
        model = Rank_Net(batch_size, 25, cfg['hidden_size'], device)
    
    criterion = torch.nn.KLDivLoss(reduction='sum').to(device)
    optimizer = cfg['optimizer'](model.parameters(), lr=cfg['lr'])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)


    
    model.to(device)
    # TRAINING MODULE
    losses = []
    training_stats = []
    print('Model Name ', model_name, ' -- representation used --',
          rep, 'Attention over docs: ', str(Attention_over_docs))
    print('KL divergence flag: ', kl_div_flag)

    for e_num in range(epochs):
        loss_ep = 0
        correct = 0
        wrong = 0
        # Measure how long the training epoch takes.
        t0 = time.time()

        model.train()
        for i in range(0, len(y_train), batch_size):
            mini_batch_submitted_paper, mini_batch_reviewer_paper, y = get_batch_eval(
                model_name, train_sub, train_rev, y_train, i, batch_size, model.padding, rep)
            if len(y.shape) > 1:
                optimizer.zero_grad()
                
                attention_scores = model(mini_batch_submitted_paper, mini_batch_reviewer_paper).float()
                actual_relevance= F.log_softmax(get_actual_relevance_using_cosine(mini_batch_submitted_paper,mini_batch_reviewer_paper),dim=1).unsqueeze(dim=1)
                attention_scores= F.softmax(attention_scores,dim=2)

                # listNet_div= model.get_rank_loss(attention_scores,actual_relevance)
                loss= criterion(actual_relevance, attention_scores)
                loss_ep += loss.item()

                loss.backward()
                optimizer.step()

        losses.append(loss_ep/len(y_train))
        training_time = format_time(time.time() - t0)
        print("Epoch:", e_num, " Loss:",
              losses[-1], ": Training epcoh took: {:}".format(training_time))

        # #Validate after each epoch
        model.eval()
        with torch.no_grad():
            correct = 0
            wrong = 0
            loss_val = 0
            for i in range(0, len(y_val)):
                attention_scores = model(val_sub[i].unsqueeze(dim=0).float(), val_rev[i].unsqueeze(
                        dim=0).float()).float()

                attention_scores= F.softmax(attention_scores,dim=2) 

                actual_relevance= F.log_softmax(get_actual_relevance_using_cosine(val_sub[i].unsqueeze(dim=0).float(),val_rev[i].unsqueeze(
                        dim=0).float()),dim=1) #.unsqueeze(dim=1)       
                loss = criterion(actual_relevance, attention_scores)
                loss_val += loss.item()

            print("Validation Loss:", loss_val/len(y_val))

            print("========================================================")

            training_stats.append(
                {
                    'epoch': e_num,
                    'Training Loss': losses[-1],
                    'Valid. Loss': loss_val/len(y_val),

                    'Training Time': training_time
                }
            )

    PATH = rep+'-'+model_name+'-flag_attn-'+str(Attention_over_docs)+'-epochs-'+str(
        epochs)+'-batch_size-'+str(batch_size)+"-KL-"+str(kl_div_flag)

    torch.save(model.state_dict(), saved_models_folder+'/'+PATH)

    # test set
    model.eval()
    with torch.no_grad():
        correct = 0
        wrong = 0
        loss_test = 0
        # rev_entropy_val=0
        for i in range(0, len(y_test)):
            attention_scores = model(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(
                        dim=0).float()).float()

            attention_scores= F.softmax(attention_scores,dim=2) 

            actual_relevance= F.log_softmax(get_actual_relevance_using_cosine(test_sub[i].unsqueeze(dim=0).float(),test_rev[i].unsqueeze(
                    dim=0).float()),dim=1) #.unsqueeze(dim=1)       
            loss = criterion(actual_relevance, attention_scores)
            loss_test += loss.item()

            

    return loss_test/len(y_test)


if __name__ == '__main__':

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='maximize')

    study.optimize(func=train_expertise_model, n_trials=1)
    joblib.dump(study, saved_models_folder+'/'+'optuna_7_trial')

    df = study.trials_dataframe()
    print(df)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial)
