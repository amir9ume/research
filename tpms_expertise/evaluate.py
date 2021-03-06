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
from datetime import datetime
import pickle

from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import entropy

from reviewer_expertise.models import Match_LR, Regression_Attention_Over_docs, Regression_Simple
from reviewer_expertise.utilities_model import get_train_test_data_from_hidden_representations, format_time, make_plot_training, prepare_data_bow, prepare_data, get_batch_eval, calculate_entropy_element, get_reviewer_entropy
from reviewer_expertise.utilities_model import get_test_data_from_hidden_representations_with_ids, Average, pad_sequence_my
import argparse

folder = "./model_training/"


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to the $slurm tmpdir is passed')
parser.add_argument('--saved_models', type=str,
                    help='folder path to the saved models ')
parser.add_argument('--model_name', type=str, help='name of the saved model')

args = parser.parse_args()

curr_time_is = str(datetime.today().strftime('%Y-%m-%d'))
saved_evaluation = 'sav_eval_'+curr_time_is+'_old/'
os.mkdir(saved_evaluation)


# saved_models="../../workingAmir/tpms_expertise/saved_models_today/"
saved_models = args.saved_models
# model_path="LDA-Match_LR-flag_attn-True-epochs-2-batch_size-64-KL-True"
model_path = args.model_name

params = model_path.split('-')
if not os.path.exists(saved_models+model_path):
    print('model not trained yet')
else:
    rep = params[0]
    model_name = params[1]
    Attention_over_docs = bool(params[3])
    batch_size = int(params[-3])
    KL_flag = str(params[-1])
   # data_path = '../../workingAmir/data_info/loaded_pickles_nips19/'
    data_path = args.data_path
    """
    test_sub= torch.load('test_sub.pt')
    test_rev=torch.load('test_rev.pt')
    y_test=torch.load('y_test.pt')
    """

    device = 'cpu'
    # device='cuda:0'
    train_sub, val_sub, test_sub, train_rev, val_rev, test_rev, y_train, y_val, y_test, reviewer_ids, submitter_ids = get_test_data_from_hidden_representations_with_ids(
        rep, data_path, device)

    print('Model Name ', model_name, ' -- representation used --', rep,
          'Attention over docs: ', str(Attention_over_docs), ' KL used :', KL_flag)
    model = Match_LR(batch_size, 25, 25, 4, device, Attention_over_docs)
    model.load_state_dict(torch.load(saved_models+model_path))
    criterion = torch.nn.MSELoss(reduction='sum')

    model.to(device)
    model.eval()
    y_true = np.zeros(len(y_test))
    y_pred = np.zeros(len(y_test))

    rev_dict = {}
    reviewer_entropies = {}
    reviewer_distances_mean = {}
    reviewer_distances_attn = {}
    reviewer_mses = {}
    with torch.no_grad():
        correct = 0
        wrong = 0
        loss_test = 0
        for i in range(0, len(y_test)):
            if model_name == "Regression_Simple" and rep != "BOW":
                prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(
                    test_rev[i].unsqueeze(dim=0), dim=1).float()).float()
            elif rep == "BOW":
                prediction = model(test_sub[i].unsqueeze(
                    dim=0).float(), test_rev[i].unsqueeze(dim=0).float()).float()
            elif rep == "LDA" and model_name == "Match_LR" and model.padding == False:
                prediction = model(test_sub[i].unsqueeze(dim=0).float(), torch.mean(
                    test_rev[i].unsqueeze(dim=0), dim=1).float()).float()[0]
            else:
             #   test_rev[i]=pad_sequence_my(test_rev[i].unsqueeze(dim=0).float(), batch_first=True, padding_value=0,max_length=350)
                # prediction = model(test_sub[i].unsqueeze(dim=0).float(), pad_sequence_my(
                #     test_rev[i].unsqueeze(dim=0).float(), batch_first=True, padding_value=0, max_length=350)).float()

                prediction = model(test_sub[i].unsqueeze(dim=0).float(), 
                    test_rev[i].unsqueeze(dim=0).float()).float()
            loss = criterion(prediction, y_test[i].argmax(dim=0).float())
            loss_test += loss.item()

            class_label = torch.round(prediction).squeeze(
                dim=0)  # .argmax(dim=1)
            trg_label = y_test[i].argmax(dim=0)
            
            # d1, d2 = model.get_distance_attn_reviewer_sub(test_sub[i].unsqueeze(dim=0).float(), pad_sequence_my(
            #     test_rev[i].unsqueeze(dim=0).float(), batch_first=True, padding_value=0, max_length=350))
            d1, d2 = model.get_distance_attn_reviewer_sub(test_sub[i].unsqueeze(dim=0).float(), 
                test_rev[i].unsqueeze(dim=0).float())

            distance = (d1, d2)

            rev_id = reviewer_ids[i]
            submission_id = submitter_ids[i]
            if rev_id not in rev_dict:
                rev_dict[rev_id] = [loss.item()]
                reviewer_entropies[rev_id] = get_reviewer_entropy(
                    test_rev[i].unsqueeze(dim=0).float()).item()
            else:
                rev_dict[rev_id].append(loss.item())

            if rev_id not in reviewer_distances_mean:
                reviewer_distances_mean[rev_id] = [d1]
                reviewer_distances_attn[rev_id] = [d2]
                reviewer_mses[rev_id] = [loss.item()]

            else:
                reviewer_distances_mean[rev_id].append(d1)
                reviewer_distances_attn[rev_id].append(d2)
                reviewer_mses[rev_id].append(loss.item())

            y_true[i] = class_label
            y_pred[i] = trg_label

            if class_label == trg_label:
                correct = correct+1
            else:
                correct = correct + \
                    torch.sum(class_label == trg_label).item()

        print("Test Loss:", loss_test/len(y_test),
              ": Test Accuracy:", correct/len(y_test))
        print("========================================================")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

    for rev_id in rev_dict:
        avg = Average(rev_dict[rev_id])
        rev_dict[rev_id] = avg

    with open(saved_evaluation+"rev_dict_for_plot", "wb") as output_file:
        pickle.dump(rev_dict, output_file)
    print('evalute done')

    with open(saved_evaluation+"reviewer_entropies_for_plot", "wb") as o:
        pickle.dump(reviewer_entropies, o)
    print('evalute done')

    with open(saved_evaluation+"reviewer_distances_mean_for_plot", "wb") as o:
        pickle.dump(reviewer_distances_mean, o)

    with open(saved_evaluation+"reviewer_distances_attn_for_plot", "wb") as o:
        pickle.dump(reviewer_distances_attn, o)

    with open(saved_evaluation+"reviewer_mses", "wb") as o:
        pickle.dump(reviewer_mses, o)
