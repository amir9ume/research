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
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import entropy

from reviewer_expertise.other_models import Regression_Attention_Over_docs, Regression_Simple
from reviewer_expertise.models import Match_LR, Rank_Net 
from reviewer_expertise.utilities_model import get_train_test_data_from_hidden_representations, format_time, make_plot_training, prepare_data_bow, prepare_data, get_batch_eval, calculate_entropy_element, get_reviewer_entropy
from reviewer_expertise.utilities_model import get_test_data_from_hidden_representations_with_ids, Average, pad_sequence_my
from reviewer_expertise.utilities_model import calculate_entropy_element , load_pre_trained_weights_to_model, get_cosine_similiarity_values
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
if os.path.exists(saved_evaluation):
    pass
else:
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
    
    folder_saved_test_data='data/'
    test_sub= torch.load(folder_saved_test_data+'test_sub.pt')
    test_rev=torch.load(folder_saved_test_data+'test_rev.pt')
    y_test=torch.load(folder_saved_test_data+'y_test.pt')
    submitter_ids=torch.load(folder_saved_test_data+'submitter_ids.pt')
    reviewer_ids= torch.load(folder_saved_test_data+'reviewer_ids.pt')

    device = 'cpu'
    # device='cuda:0'
    # train_sub, val_sub, test_sub, train_rev, val_rev, test_rev, y_train, y_val, y_test, reviewer_ids, submitter_ids = get_test_data_from_hidden_representations_with_ids(
    #     rep, data_path, device)

    print('Model Name ', model_name, ' -- representation used --', rep,
          'Attention over docs: ', str(Attention_over_docs), ' KL used :', KL_flag)
    #this attention hardcoded hidden size can cause error in future. careful.
    model = Match_LR(batch_size, 25, 24, 4, device, Attention_over_docs)
    
    load_pre_trained_weights_to_model(model, saved_models+model_path)
    
    #model.load_state_dict(torch.load(saved_models+model_path))
    criterion = torch.nn.MSELoss(reduction='sum')

    model.to(device)
    model.eval()
    y_true = np.zeros(len(y_test))
    y_pred = np.zeros(len(y_test))

    observations=[]
    rev_dict = {}
    reviewer_entropies = {}
    attention_entropies= {}
    reviewer_mses = {}
    with torch.no_grad():
        correct = 0
        wrong = 0
        loss_test = 0

        dict_rev_subm_pair={}
        for i in range(0, len(y_test)):
        
            #   test_rev[i]=pad_sequence_my(test_rev[i].unsqueeze(dim=0).float(), batch_first=True, padding_value=0,max_length=350)
            # prediction = model(test_sub[i].unsqueeze(dim=0).float(), pad_sequence_my(
            #     test_rev[i].unsqueeze(dim=0).float(), batch_first=True, padding_value=0, max_length=350)).float()

            prediction = model(test_sub[i].unsqueeze(dim=0).float(), 
                test_rev[i].unsqueeze(dim=0).float()).float()
            loss = criterion(prediction, y_test[i].argmax(dim=0).float())
            loss_test += loss.item()


            class_label = torch.round(prediction).squeeze(
                dim=0)  # .argmax(dim=1)
            #this trg_label is the bid value
            trg_label = y_test[i].argmax(dim=0)
            
            attention_scores= model.get_attention_scores(test_sub[i].unsqueeze(dim=0).float(), 
                test_rev[i].unsqueeze(dim=0).float())

            # THIS IS CRUCIAL FOR TODAY
            attention_entropy= torch.sum(calculate_entropy_element(attention_scores[0][0]))

            rev_id = reviewer_ids[i]
            submission_id = submitter_ids[i]
            reviewer_entropy = get_reviewer_entropy(test_rev[i].unsqueeze(dim=0).float()).item()
            cs_max= torch.max(get_cosine_similiarity_values(test_sub[i].unsqueeze(dim=0).float(), test_rev[i].unsqueeze(dim=0).float())).item()
            concat_rev_sub_id= str(rev_id)+'-'+str(submission_id)

            #this should be unique key.
            if concat_rev_sub_id not in dict_rev_subm_pair:
                observations.append(
                    {
                    'rev_id': rev_id,
                    'reviewer_entropy': reviewer_entropy,
                    'sid': submission_id,
                    'bid': trg_label.item(),
                    'mse': loss.item(),
                    'attention_entropy':attention_entropy.item(),
                    'cosine_max': cs_max
                }
                )
                    
            
            y_true[i] =  trg_label
            y_pred[i] = class_label

            if class_label == trg_label:
                correct = correct+1
            else:
                correct = correct + \
                    torch.sum(class_label == trg_label).item()

        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=observations)

        # # Use the 'epoch' as the row index.
        # df_stats = df_stats.set_index('epoch')

        print(df_stats.sample(5))
        df_stats.to_csv(saved_evaluation+"observation_stats_pretrain",index=False)
        
        print("Test Loss:", loss_test/len(y_test),
              ": Test Accuracy:", correct/len(y_test))
        print("========================================================")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

    # for rev_id in rev_dict:
    #     avg = Average(rev_dict[rev_id])
    #     rev_dict[rev_id] = avg

    # with open(saved_evaluation+"rev_dict_for_plot", "wb") as output_file:
    #     pickle.dump(rev_dict, output_file)
    # print('evalute done')

