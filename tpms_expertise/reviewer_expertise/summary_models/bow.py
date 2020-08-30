#credits: Meghana Moorthy Bhat#
import os
from random import sample
import operator
import pickle
import torch

file='../../../data_info/bow_w2ix_nips19_reverse'
if os.path.exists(file):
    with open(file,'rb') as f:
        bow_w2ix= pickle.load(f)
    pass
else:
    load_dir= '../../../../neurips19_anon/'
    bow_w2ix = {}
    count = 0

    vocab_size= 30000
    flag=True

    def make_word_2_index(load_dir):
        for root, dirn, files in os.walk(load_dir):
            
            for f in files:
                if f.endswith(".bow") :
                    with open(os.path.join(root, f)) as fp:  
                        lines = fp.readlines()
                        if len(lines) > 500:
                            lines_t = lines
                            lines = sample(lines_t, 500)
                        for line in lines:
                            if line.split()[0] not in bow_w2ix:
                                bow_w2ix[line.split()[0]]=len(bow_w2ix)
                            if len(bow_w2ix) >= vocab_size:                        
                                return bow_w2ix

        return bow_w2ix

    bow_w2ix=make_word_2_index(load_dir)

    sorted_vocab = sorted(bow_w2ix.items(),
            key=operator.itemgetter(1),
            reverse=False)

    bow_temp = [x for x in sorted_vocab[:30004]]
    bow_w = dict(bow_temp)
    bow_w2ix = {v: i for i, v in enumerate(bow_w.keys())}


    vocab_size = len(bow_w2ix)
    print(vocab_size)

    with open('../../../data_info/bow_w2ix_nips19_reverse','wb') as f:
        pickle.dump(bow_w2ix, f)


def build_bow_embeddings( bow_w2ix ,ac_list):
    reviewer = dict()
    submitter = dict()
    
    #open submitted papers one by one. read the bow papers. and prepare vectors.    
    root='../../../../neurips19_anon/'
    folder='submitted_papers/'
    submitted_files= sorted(os.listdir(root + 'submitted_papers/'))
    i=0
    for paper in submitted_files:
        pid= paper.split('.')[0].split('paper')[1]
        paper_id = pid
        submitter[paper_id] = torch.zeros(len(bow_w2ix))
        i+=1
        with open(root+folder+paper, 'r') as f:
            if ".bow" in paper:
                lines = f.readlines()
                for line in lines:
                    words = line.split()
                    w = words[0]
                    count = int(words[1])
                    if w in bow_w2ix:
                        submitter[paper_id][bow_w2ix[w]]=count

    folder='archive_papers/'
    archive_files= sorted(os.listdir(root + folder))
    i=0
    #read only for ac list for now
    for rev_id in archive_files:
        author = rev_id
        if rev_id.isdigit():
            if int(rev_id)  in ac_list:
                reviewer_files= sorted(os.listdir(root + folder+ rev_id+'/'))
                if author not in reviewer:
                    reviewer[author] = []
                # else:
                #     reviewer[author].append(embedding[i])

                for r_file in reviewer_files:
                    vector= torch.zeros(len(bow_w2ix))
                    with open(root+folder+rev_id+'/'+r_file, 'r') as f:
                        if ".bow" in r_file:
                            lines = f.readlines()
                            for line in lines:
                                w = line.split()[0]
                                count = int(line.split()[1])
                                if w in bow_w2ix:
                                    #reviewer[author][bow_w2ix[w]]+=count
                                    vector[bow_w2ix[w]]+=count
                    reviewer[author].append(vector)
        
    return reviewer, submitter

ac_list_file= '../../../data_info/loaded_pickles_nips19/ac_ids'
with open(ac_list_file,'rb') as fp:
    ac_list= pickle.load(fp)

r,s= build_bow_embeddings(bow_w2ix,ac_list)
with open('ac_reviewer_bag_words_tensor_nips19','wb') as f:
    pickle.dump(r,f)

with open('submitted_paper_bag_words_tensor_nips19','wb') as fp:
    pickle.dump(s,fp)
