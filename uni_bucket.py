import pickle
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import re
from functools import partial
import operator
import csv
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import seaborn as sns
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt



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

dict_sid_to_bucket= pd.Series(u_final.bucket.values,index=u_final.sid).to_dict()
dict_bucket_corpus={}

def text_processing(corpus):
    for sen in range(0, len(corpus)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(corpus[sen]))
        
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        #removes numbers
        document = re.sub(r'[0-9]+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    
if sys.argv[3]=='prepare':
    list_files= sorted(os.listdir('../submitted_papers/'))
  
#    full_corpus=[]
    full_corpus=''
    c=0
    for filename in list_files:
        if len(filename.split('paper',1))>1:
            fname=filename.split('paper',1)[1]
            print(fname)
            sid=fname.split('.')[0]
            if int(sid) in dict_sid_to_bucket:
                bucket_result= dict_sid_to_bucket[int(sid)]
            else :
                bucket_result=6
                c+=1
            print('sid is ', sid,' bucket is ', bucket_result)
            f=open('../submitted_papers/'+filename)
            content=f.read()
            #data= content.splitlines()
            data=content
            #full_corpus.extend(data)
            full_corpus=full_corpus+data
            if bucket_result in dict_bucket_corpus:
                dict_bucket_corpus[bucket_result]+data

            else:
                dict_bucket_corpus[bucket_result]=data
            f.close()
            
    n_features=20000
    #vocabulary= text_processing(full_corpus)

    print('length of full corpus before processing ', len(full_corpus))
    
    full_corpus = re.sub('[^A-Za-z0-9]+', ' ', str(full_corpus))
    full_corpus=re.sub(r'[0-9]+', ' ', str(full_corpus))

    #text_processing(full_corpus)    
    print('length of full corpus after processing ', len(full_corpus))
    
    #min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
    #min_df = 5 means "ignore terms that appear in less than 5 documents"
    
    tf_vectorizer=TfidfVectorizer(norm='l2', use_idf=True,min_df= 0.3,strip_accents='unicode',stop_words='english' ,smooth_idf=False, sublinear_tf=True, max_features=n_features)
    
    #print('shape of full corpus ', full_corpus.shape)
    print('type of full corpus ', type(full_corpus),'shape of it is ', len(full_corpus))
    #print('full corpus is: ', full_corpus)
    z=tf_vectorizer.fit_transform([full_corpus])
    print('features are: ', tf_vectorizer.get_feature_names()[:-30])
    l=[]
    for b in sorted(dict_bucket_corpus.keys()):
        corpus= dict_bucket_corpus[b]
       # corpus=text_processing(corpus)
        corpus=re.sub('[^A-Za-z0-9]+', ' ', str(corpus))
        corpus= re.sub(r'[0-9]+', ' ', str(corpus))
        y=tf_vectorizer.transform([corpus])
       
        l.append(y)
        print('key in dictionary',b)
    
    #print((dict_bucket_corpus))

    #print(tf_vectorizer.get_feature_names()[:10])
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics.pairwise import paired_cosine_distances as pcd
    cosine_sim=np.zeros((5,5))
    for i in range(len(l)-1):
        c_sim=[]
        for j in range(len(l)-1):
          
            cc=cosine_similarity(l[i],l[j])    
         #   print('Cosine similarity between ',i,' and ',j,cc)
            c_sim.append(cc)
        #c_sim=np.asarray(c_sim)    
        #print(1- pcd(l[i], l[j]))
        cosine_sim[i]=c_sim
    #cosine_sim=np.asarray(cosine_sim)
    print(cosine_sim)
    
    sns.heatmap(cosine_sim, annot=True, cmap=plt.cm.Reds)
    plt.ylabel('buckets from 1 to 5')
    plt.xlabel('buckets from 1 to 5')
    plt.title('Heatmap cosine similarities Uni buckets-full paper')
    plt.show()
