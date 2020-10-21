import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle

import time
import datetime
from datetime import datetime

from models_text_rep import  LSTMWithAttentionAutoEncoder
from contrastive_loss import NTXentLoss, loss_fn, loss_towards_ds
from util_for_summary import read_data, get_batch
torch.manual_seed(1)


cfg = {'device': "cuda" if torch.cuda.is_available() else "cpu",
           'batch_size': 2,
           'epochs':20,
           'src_emb_dim':50,
            'src_hidden_dim':50,
            'pad_token_src':0,
            'lr': 0.009801,
           'momentum': 0.97,
           'optimizer': torch.optim.Adam

        }

device=cfg['device']

home = str(Path.home())
#archive_corpus_path=home+'ac_text_corpus_reviewers_nips19.pickle'
submissions_corpus_path=home+'/arcopy/research/tpms_expertise/reviewer_expertise/summary_models/'
paper_representation = pickle.load(open(submissions_corpus_path+'submission_corpus_reviewers_nips19.pickle', 'rb'))
#read 10% data for now, to have a proof of concept
data=[]
for p in paper_representation:
    data.append(paper_representation[p])

src, word2idx, idx2word = read_data(data)
n_words = len(word2idx)
vocab_size = n_words
print('vocab size is :', vocab_size)


model= LSTMWithAttentionAutoEncoder(cfg['src_emb_dim'],
        vocab_size,
        cfg['src_hidden_dim'],
        cfg['batch_size'],
        cfg['pad_token_src'])

model.to(device)
#self.nt_xent_criterion = NTXentLoss(device, cfg['batch_size'], **config['loss'])
#nt_xent_criterion = NTXentLoss(device, cfg['batch_size'], 0.5,True)
optimizer = cfg['optimizer'](model.parameters(), lr=cfg['lr'])

losses = []
training_stats = []
#harcoded max len for now
max_len=10
for e_num in range(cfg['epochs']):
    loss_ep = 0
    correct = 0
    wrong = 0
    # Measure how long the training epoch takes.
    t0 = time.time()

    model.train()
    for j in range(0, len(src), cfg['batch_size']):
        input_lines_src, transformed_lines, lengths_src = get_batch(
                        src, word2idx, j,
                        cfg['batch_size'], max_len) 

        #the encoder should take input line and transformed line, to do the operation
        z = model(input_lines_src)

        #for now imagine a random perturbation on z
        # to get z"

        z2=z
        optimizer.zero_grad()            
    #    loss= nt_xent_criterion(z.float(), z2.float())
    #0.5 is temperature value here.
        loss= loss_towards_ds(z.float(), z2.float(), 0.5)
        loss_ep += loss.item()
        loss.backward()
        optimizer.step()

#    losses.append(loss_ep/len(y_train))
    losses.append(loss_ep)
    training_time = format_time(time.time() - t0)
    print("Epoch:", e_num, " Loss:",
            losses[-1], ": Training epcoh took: {:}".format(training_time))

    training_stats.append(
        {
            'epoch': e_num,
            'Training Loss': losses[-1],
            'Training Time': training_time
        }
    )

