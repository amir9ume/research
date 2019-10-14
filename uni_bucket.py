import pickle
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import re
from functools import partial
import operator
import csv
import sys

file = open('ranking_uni.pkl', 'rb')
rankings = pickle.load(file)
#print(rankings)
file.close()

file=open('./data_info/shanghai_ranks.csv','r')
reader=csv.reader(file)
shanghai_ranks = dict((rows[1],int(rows[0])) for rows in reader)

folder='./data_info/'
u= pd.read_csv(folder+'author_and_affiliation_from_metadata', sep= '|')
#print(u.columns)
del u[' Gender']
del u[' Author size']
z= u.copy()

#use the top uni rankings for bucket. that is guaranteed
g= open('./data_info/top_uni', "r")
top_ranking_uni= g.read()
#print(top_ranking_uni)
#print(type(top_ranking_uni))

cs_rank_high= rankings[max(rankings.items(), key=operator.itemgetter(1))[0]]
shanghai_rank_high=shanghai_ranks [max(shanghai_ranks.items(), key=operator.itemgetter(1))[0]]
max_rank_present= max(cs_rank_high,shanghai_rank_high)
#print(max_rank_present)

def ranking_func(uni):
    #print(uni)
    uni=str(uni)
    if uni=='others':
        return max_rank_present+1

    if uni in rankings:
        return rankings[uni]
    elif uni in top_ranking_uni:
            return 1
    
    else: 
        uni_z= uni.lower()
        #uni_z=re.sub(r"[^a-zA-Z0-9]+", ' ', uni_z)
        #uni_z= ''.join(e for e in uni_z if e.isalnum())
        dict_ratios_cs={}
        for d in rankings:
            d_z= d.lower()  
            #d_z=re.sub(r"[^a-zA-Z0-9]+", ' ', d_z)
            Ratio = fuzz.ratio(uni_z,d_z)
            if len(d) < len(uni_z) :
                if d in uni_z:
                   Ratio=95
            else :
                if uni_z in d:
                    Ratio=95

            dict_ratios_cs[d]=Ratio
            #print('ratio:  ', Ratio, '---uni is ',uni_z,'---- d is ----',d_z)
        target_cs_ranking=max(dict_ratios_cs.items(), key=operator.itemgetter(1))[0]     
        
        #using shanghai rankings below to supplement cs rankings 
        dict_ratios_shanghai={}
        for d in shanghai_ranks:
            d_z= d.lower()  
            #d_z=re.sub(r"[^a-zA-Z0-9]+", ' ', d_z)
            Ratio = fuzz.ratio(uni_z,d_z)
            """
            if len(d) < len(uni_z) :
                if d in uni_z:
                   Ratio=95
            else :
                if uni_z in d:
                    Ratio=95
            """    
            dict_ratios_shanghai[d]=Ratio
            #print('ratio:  ', Ratio, '---uni is ',uni_z,'---- d is ----',d_z)
        
        target_shanghai_ranking=max(dict_ratios_shanghai.items(), key=operator.itemgetter(1))[0]

        if dict_ratios_cs[target_cs_ranking]>85:
            #r=(rankings[target_cs_ranking])
           # print('University given as input ', uni, '---- University found--',target_cs_ranking,'----Ranking-----', rankings[target_cs_ranking])
            return rankings[target_cs_ranking]
        elif dict_ratios_shanghai[target_shanghai_ranking]>85:
            #print('University given as input ', uni, '---- University found--',target_shanghai_ranking,'----Ranking-----', shanghai_ranks[target_shanghai_ranking])
            return shanghai_ranks[target_shanghai_ranking]
        else:
            return max_rank_present+1

def bucketing_universities(rank):
    rank= int(rank)
    if rank==0:
        return 5
    elif rank >0 and rank<10:
        return 1
    elif rank>=10 and rank<50:
        return 2
    elif rank>=50 and rank<100:
        return 3
    elif rank>=100 and rank<200:
        return 4
    else:
        return 5

u['ranking']= u[' Univeristy'].apply(ranking_func)
#u = u.dropna(subset=['rankings'])
u['ranking']= u['ranking'].apply(int)
print('size before droping ', len(u.index))
u=u.loc[u['ranking']!=max_rank_present+1]
print('size after droping ', len(u.index))
#qcut is doing the equal sized bucketing. For now lets comment it out and do bucketing based on the conventional if else logic
#or you could just bucket your papers as top 100 versus rest of the world

if sys.argv[2]=='equal':
    u['bucket']=pd.qcut(u['ranking'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
elif sys.argv[2]=='manual':
    u['bucket']= u['ranking'].apply(bucketing_universities)
print(u[60:100])
print('number of sids in each bucket')
print(u.groupby('bucket')[' Submission ID'].count() )

print(u.loc[u['bucket']==2])
print(u.loc[u['bucket']==3])
print(u.loc[u['bucket']==1])
print(u.loc[u['bucket']==4])
print(u.loc[u['bucket']==5])


#u.to_csv('author_uni_ranking_bucket', index=False)

#tag abstracts to these ids and read them.Make their document vectors
#read files according to each of the bins
#how to get abstract for each bin- sid to abstract- (based on author final) and sid to u bucket
u.columns=['Name','sid','author pos','country','university','ranking','bucket']
print(u[:30])
del z[' Country']
del z[' Univeristy']
z.columns= ['Name', 'sid','author pos']
print(z)

max_author= z.groupby('sid', sort= False)['author pos'].max()
max_author.columns=['sid','max_author']
#print(max_author.head())
#print(max_author.columns)
u_with_max_author= pd.merge(u,max_author, how='left',on='sid')
u_with_max_author.columns=['Name','sid','author pos','country','university','ranking','bucket','max_author']
print(u_with_max_author[:10])

if sys.argv[1]=='final':
    u_final=u_with_max_author.loc[u_with_max_author['author pos']==u_with_max_author['max_author']]
elif sys.argv[1]=='first':
    u_final=u_with_max_author.loc[u_with_max_author['author pos']==1]
print('number of sids in each bucket')
print(u_final.groupby('bucket')['sid'].count() )



"""
import pandas as pd
filepath= './data_info/author_uni_ranking_bucket'
author= pd.read_csv(filepath)
#print(author)
print(author.columns)
#[' Submission ID', ' Author Position', ' Country', ' Univeristy',
#       'ranking', 'bucket']
author.columns=['Name','sid','author_pos','country','university','ranking','bucket']
print(author.head())

fpath= './data_info/abstracts_processed'
abstracts= pd.read_csv(fpath)
#print(abstracts)
print(abstracts.columns)

abstracts_with_bucket= pd.merge(abstracts, author, how='left', on='sid')
print(abstracts_with_bucket.head())
print(abstracts_with_bucket.columns)


max_author= abstracts_with_bucket.groupby('sid', sort= False)['author_pos'].max()
abstracts_with_bucket_max_author= pd.merge(abstracts_with_bucket,max_author, how='left',on='sid')

abstracts_with_bucket_max_author.columns= ['sid', 'title', 'abstract', 'Name', 'author_pos', 'country',
       'university', 'ranking', 'bucket', 'max_author_pos']
print(abstracts_with_bucket_max_author.head(20))
#ab= abstracts_with_bucket.loc[abstracts_with_bucket['author_pos']==1]

ab_for_max_author= abstracts_with_bucket_max_author.loc[abstracts_with_bucket_max_author['author_pos']==abstracts_with_bucket_max_author['max_author_pos']]
print(ab_for_max_author.head())

def write_abstracts(ab,bin):
    
    fname='./data_info/uni_bin_abstracts_max_author/bin_'+str(bin)+'_uni_abstract'
    cols_to_keep=['title','abstract']
    ab[cols_to_keep].to_csv(fname, index= False)

#careful with the line below. all the writing is done using below data frame only 
abstracts_with_bucket= ab_for_max_author
print(abstracts_with_bucket.head())

a1= abstracts_with_bucket.loc[abstracts_with_bucket['bucket']==1]
write_abstracts(a1,1)

a2= abstracts_with_bucket.loc[abstracts_with_bucket['bucket']==2]
write_abstracts(a2,2)

a3= abstracts_with_bucket.loc[abstracts_with_bucket['bucket']==3]
write_abstracts(a3,3)

a4= abstracts_with_bucket.loc[abstracts_with_bucket['bucket']==4]
write_abstracts(a4,4)

a5= abstracts_with_bucket.loc[abstracts_with_bucket['bucket']==5]
write_abstracts(a5,5)
"""