import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pickle
import pandas as pd


def prepare_data_bow(submitter, reviewer, df, device='cpu'):
    train_data_sub = []
    train_data_rev = []
    submit = submitter.keys()
    submitter_ids = []
    reviewer_ids = []
    rev = reviewer.keys()
    labels = []
    for i in range(len(df)):
        pid_curr = str(df.iloc[i]['pid'])
        rev_curr = str(df.iloc[i]['anon_id'])
        if pid_curr in submit and rev_curr in reviewer:
            train_data_sub.append(submitter[pid_curr])  # .cuda()
            train_data_rev.append(torch.mean(
                torch.stack(reviewer[rev_curr]), dim=0))  # .cuda()
            idx = int(df.iloc[i]['bid'])
            temp = torch.LongTensor([0, 0, 0, 0], device=device)  # .cuda()
            for i in range(4):
                if i == idx:
                    temp[i] = 1
            labels.append(temp)
            submitter_ids.append(df.iloc[i]['pid'])
            reviewer_ids.append(df.iloc[i]['anon_id'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids


def prepare_data(submitter, reviewer, df, device='cpu'):

    train_data_sub = []
    train_data_rev = []
    submit = submitter.keys()
    submitter_ids = []
    reviewer_ids = []
    rev = reviewer.keys()
    labels = []
    for i in range(len(df)):
        pid_curr = str(df.iloc[i]['pid'])
        rev_curr = str(df.iloc[i]['anon_id'])
        if pid_curr in submit and rev_curr in reviewer:
            train_data_sub.append(torch.tensor(
                submitter[pid_curr], requires_grad=True, device=device))  # .cuda()
            train_data_rev.append(torch.tensor(
                reviewer[rev_curr], requires_grad=True, device=device))  # .cuda()
            idx = int(df.iloc[i]['bid'])
            temp = torch.LongTensor([0, 0, 0, 0]).to(device)  # .cuda()
            for j in range(4):
                if j == idx:
                    temp[j] = 1
            labels.append(temp)
            submitter_ids.append(df.iloc[i]['pid'])
            reviewer_ids.append(df.iloc[i]['anon_id'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids


def prepare_data_find_reviewer_ids(submitter, reviewer, df, find_reviewer_ids, device='cpu'):
    train_data_sub = []
    train_data_rev = []
    submit = submitter.keys()
    submitter_ids = []
    reviewer_ids = []
    rev = reviewer.keys()
    labels = []
    for i in range(len(df)):
        pid_curr = str(df.iloc[i]['pid'])
        rev_curr = str(df.iloc[i]['anon_id'])
        if rev_curr in find_reviewer_ids:
            if pid_curr in submit and rev_curr in reviewer:
                train_data_sub.append(torch.tensor(
                    submitter[pid_curr], requires_grad=True, device=device))  # .cuda()
                train_data_rev.append(torch.tensor(
                    reviewer[rev_curr], requires_grad=True, device=device))  # .cuda()
                idx = int(df.iloc[i]['bid'])
                temp = torch.LongTensor([0, 0, 0, 0], device=device)  # .cuda()
                for j in range(4):
                    if j == idx:
                        temp[j] = 1
                labels.append(temp)
                submitter_ids.append(df.iloc[i]['pid'])
                reviewer_ids.append(df.iloc[i]['anon_id'])
    return train_data_sub, train_data_rev, labels, submitter_ids, reviewer_ids



def pad_sequence_my(sequences, batch_first=False, padding_value=0.0, max_length=350):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    max_len = max_length
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def get_batch_eval(model_name, paper_emb, rev_emb, trg_value, idx, batch_size, padding, rep):
    paper_lines = Variable(torch.stack(
        paper_emb[idx:idx+batch_size]), requires_grad=True)  # .permute(1,0)
    rev_papers = rev_emb[idx:idx+batch_size]
    if padding == True:
        #    reviewer_papers = pad_sequence(rev_papers, batch_first=True, padding_value=0) # padding as different reviewers can have different number of papers
        reviewer_papers = pad_sequence_my(
            rev_papers, batch_first=True, padding_value=0, max_length=350)
    elif (model_name == "Regression_Simple" or model_name == "Match_LR") and rep != "BOW":
        reviewer_papers = torch.stack(
            [torch.mean(r, dim=0) for r in rev_papers])
    elif rep == "BOW":
        reviewer_papers = torch.stack(rev_papers)
    trg = torch.stack(trg_value[idx:idx+batch_size]).squeeze()

    return paper_lines.float(), reviewer_papers.float(), trg


def get_train_test_data_from_hidden_representations(rep, data_path, device):

    if rep == 'BOW':
        folder = 'reviewer_expertise/summary_models/'
        reviewer_representation = pickle.load(
            open(folder + 'ac_reviewer_bag_words_tensor_nips19', "rb"))
        paper_representation = pickle.load(
            open(folder+'submitted_paper_bag_words_tensor_nips19', 'rb'))
    else:
        reviewer_representation = pickle.load(
            open(data_path + 'dict_all_reviewer_lda_vectors.pickle', "rb"))
        paper_representation = pickle.load(
            open(data_path + 'dict_paper_lda_vectors.pickle', "rb"))

    # bds_path='~/arcopy/neurips19_anon/anon_bids_file'
    #bds_path= '~/arcopy/workingAmir/data_info/loaded_pickles_nips19/bids_ac_anon_nips19'
    bds_path = data_path+'bids_ac_anon_nips19'

    df = pd.read_csv(bds_path)
    df = df.sample(frac=1)
    size = len(df.index)

    print('data size is ', len(df.index))
    print(df.sample(3))

    if rep == "BOW":
        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data_bow(
            paper_representation, reviewer_representation, df, device)
    else:
        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(
            paper_representation, reviewer_representation, df, device)

    train_length = int(0.7*len(data_sub))
    val_length = int(.15*len(data_sub))
    test_length = len(data_sub) - train_length - val_length

    train_sub = data_sub[:train_length]
    val_sub = data_sub[train_length: (train_length+val_length)]
    test_sub = data_sub[train_length+val_length:]

    train_rev = data_rev[:train_length]
    val_rev = data_rev[train_length: (train_length+val_length)]
    test_rev = data_rev[train_length+val_length:]

    y_train = data_y[:train_length]
    y_val = data_y[train_length:(train_length+val_length)]
    y_test = data_y[train_length+val_length:]

    return train_sub, val_sub, test_sub, train_rev, val_rev, test_rev, y_train, y_val, y_test


def Average(lst):
    return sum(lst) / len(lst)


def calculate_entropy_element(p):
    e = 0.00000001
    if torch.sum(p).item() < 0.01:
        return 0
    p = p + e
    return -1 * p * np.log2(p)


def get_reviewer_focus_entropy(reviewer_emb):
    # this includes 0 padding
    papers = [x for x in reviewer_emb[0]]
    entropy_total = 0
    count = 0
    for z in papers:
        entropy = torch.sum(calculate_entropy_element(z))
        if entropy > 0:
            entropy_total += entropy
            count += 1
    return entropy_total/count


def get_reviewer_entropy(reviewer_emb):
    # this includes 0 padding
    aggregate_work = torch.mean(reviewer_emb, dim=1).squeeze(dim=0)
    entropy = torch.sum(calculate_entropy_element(aggregate_work))
    return entropy


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def make_plot_training(df_stats, epochs):

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=.5)
    plt.rcParams["figure.figsize"] = (14, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([e for e in range(epochs)])

    plt.show()


def get_test_data_from_hidden_representations_with_ids(rep, data_path, device):

    if rep == 'BOW':
        folder = 'reviewer_expertise/summary_models/'
        reviewer_representation = pickle.load(
            open(folder + 'ac_reviewer_bag_words_tensor_nips19', "rb"))
        paper_representation = pickle.load(
            open(folder+'submitted_paper_bag_words_tensor_nips19', 'rb'))
    else:
        reviewer_representation = pickle.load(
            open(data_path + 'dict_all_reviewer_lda_vectors.pickle', "rb"))
        paper_representation = pickle.load(
            open(data_path + 'dict_paper_lda_vectors.pickle', "rb"))

    # bds_path='~/arcopy/neurips19_anon/anon_bids_file'
   # bds_path= '~/arcopy/workingAmir/data_info/loaded_pickles_nips19/bids_ac_anon_nips19'
    bds_path = data_path+'bids_ac_anon_nips19'
    df = pd.read_csv(bds_path)
    df = df.sample(frac=1)
    size = len(df.index)

    print('data size is ', len(df.index))
    print(df.sample(3))

    if rep == "BOW":
        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data_bow(
            paper_representation, reviewer_representation, df, device)
    else:
        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data(
            paper_representation, reviewer_representation, df, device)

    train_length = int(0.7*len(data_sub))
    val_length = int(.15*len(data_sub))
    test_length = len(data_sub) - train_length - val_length

    train_sub = data_sub[:train_length]
    val_sub = data_sub[train_length: (train_length+val_length)]
    test_sub = data_sub[train_length+val_length:]

    train_rev = data_rev[:train_length]
    val_rev = data_rev[train_length: (train_length+val_length)]
    test_rev = data_rev[train_length+val_length:]

    y_train = data_y[:train_length]
    y_val = data_y[train_length:(train_length+val_length)]
    y_test = data_y[train_length+val_length:]

    # for now only the test set ids
    reviewer_ids = reviewer_ids[train_length+val_length:]
    submitter_ids = submitter_ids[train_length+val_length:]

    return train_sub, val_sub, test_sub, train_rev, val_rev, test_rev, y_train, y_val, y_test, reviewer_ids, submitter_ids


def get_data_particular_reviewer_with_ids(rep, data_path, find_reviewer_ids, device):

    if rep == 'BOW':
        folder = 'reviewer_expertise/summary_models/'
        reviewer_representation = pickle.load(
            open(folder + 'ac_reviewer_bag_words_tensor_nips19', "rb"))
        paper_representation = pickle.load(
            open(folder+'submitted_paper_bag_words_tensor_nips19', 'rb'))
    else:
        reviewer_representation = pickle.load(
            open(data_path + 'dict_all_reviewer_lda_vectors.pickle', "rb"))
        paper_representation = pickle.load(
            open(data_path + 'dict_paper_lda_vectors.pickle', "rb"))

    bds_path = '~/arcopy/neurips19_anon/anon_bids_file'
    bds_path = '~/arcopy/workingAmir/data_info/loaded_pickles_nips19/bids_ac_anon_nips19'
    df = pd.read_csv(bds_path)
    df = df.sample(frac=1)
    size = len(df.index)

    print('data size is ', len(df.index))
    print(df.sample(3))

    if rep == "BOW":
        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data_find_reviewer_ids(
            paper_representation, reviewer_representation, df, find_reviewer_ids, device)
    else:
        data_sub, data_rev, data_y, submitter_ids, reviewer_ids = prepare_data_find_reviewer_ids(
            paper_representation, reviewer_representation, df, find_reviewer_ids, device)

    return data_sub, data_rev, data_y, reviewer_ids, submitter_ids
