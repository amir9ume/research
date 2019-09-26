import pickle
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np

file = open('ranking_uni.pkl', 'rb')
rankings = pickle.load(file)
file.close()

u= pd.read_csv('author_and_affiliation2', sep= '|')
print(u.columns)
del u[' Gender']
del u[' Author size']

#use the top uni rankings for bucket. that is guaranteed
g= open('./data_info/top_uni', "r")
contents= g.read()
print(contents)
print(type(contents))


def ranking_func(uni):
    if uni in rankings:
        return rankings[uni]
    elif uni in contents:
            return 1
    else:
        for d in rankings:
            Ratio = fuzz.partial_ratio(uni.lower(),d.lower())
            if Ratio>60:
                return rankings[d]
            else:
                return 0
        

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
u['bucket']= u['ranking'].apply(bucketing_universities)


print(u[:60])
u.to_csv('author_uni_ranking_bucket', index=False)

#tag abstracts to these ids and read them.Make their document vectors
#read files according to each of the bins
#how to get abstract for each bin- sid to abstract- (based on author final) and sid to u bucket
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
