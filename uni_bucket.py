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

def ranking_func(uni):
    if uni in rankings:
        return rankings[uni]
    else:
        for d in rankings:
            Ratio = fuzz.ratio(uni.lower(),d.lower())
            if Ratio>60:
                return rankings[d]
        

def bucketing_universities(uni):
    for d in rankings:
    
        Ratio = fuzz.ratio(uni.lower(),d.lower())
        if Ratio>60:
            if rankings[d]<10:
                return 1
            if dict[d]>=10 and rankings[d]<50:
                return 2
            if dict[d]>=50 and rankings[d]<100:
                return 3
            if dict[d]>=100 and rankings[d]<200:
                return 4
        else:
            return 5

#u['bucket']= u[' Univeristy'].apply(bucketing_universities)
u['ranking']= u[' Univeristy'].apply(ranking_func)
print(u[30:60])

#bucket universities
