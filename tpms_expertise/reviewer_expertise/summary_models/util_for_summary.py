
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
from torchtext.utils import download_from_url
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import operator

import time
import datetime

def build_vocab(src):
    vocab = dict()

    for line in src:
        for w in line:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

    if '<s>' in vocab:
        del vocab['<s>']
    if '<\s>' in vocab:
        del vocab['<\s>']
    if '<unk>' in vocab:
        del vocab['<unk>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    
    sorted_vocab = sorted(vocab.items(),
            key=operator.itemgetter(1),
            reverse=True)

    sorted_words = [x[0] for x in sorted_vocab[:30000]]

    word2idx = {'<s>' : 0,
            '</s>' : 1,
            '<unk>' : 2,
            '<pad>' : 3 }

    idx2word = { 0 : '<s>',
            1 : '</s>',
            2 : '<unk>',
            3 : '<pad>' }

    for idx, w in enumerate(sorted_words):
        word2idx[w] = idx+4
        idx2word[idx+4] = w
    
    return word2idx, idx2word

def read_data(srcfile):
    word2idx, idx2word = build_vocab(srcfile)
    return srcfile, word2idx, idx2word


def trasform_function(input_line):

    return None

#Gives batch data of size 2N. Inlcudes augmented data
def get_batch(src, word2idx, idx, batch_size, max_len):
#    print('index ',idx)
    lens = [len(line) for line in src[idx:idx+batch_size]]
    src_lines = []

    for paper in src[idx:idx+batch_size]:
        temp = []
#        temp=torch.zeros(512)
    #    print('length of paper ',len(paper))
        # cnt=0
        # if len(paper)>0:
        for w in paper:
            if w not in word2idx:
                temp.append(word2idx['<unk>'])
                # temp[cnt]=word2idx['<unk>']
            else:
                temp.append(word2idx[w])
            #    temp[cnt]=word2idx[w]
        if len(temp) < max_len:
            for i in range(len(temp), max_len):
                temp.append(word2idx['<pad>']) 
        src_lines.append(temp)
    # else:
        #     temp= [0 for j in range(0,512)]
        #     src_lines.append(temp)
    src_lines = torch.LongTensor(src_lines)
#    print('shape src_lines', src_lines.shape)
    transformed_lines = src_lines

    return src_lines, transformed_lines, lens



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
