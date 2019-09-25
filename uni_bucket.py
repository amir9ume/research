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

#bucket universities
