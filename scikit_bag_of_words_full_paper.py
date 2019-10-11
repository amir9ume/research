import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist
import nltk; nltk.download('stopwords')
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix


import re
import numpy as np
from nltk.corpus import stopwords
from itertools import chain
import sys
import os

dir= './data_info/gender_ids_based_final_author/'
f=open(dir+'male_ids','r')
c= f.read()
data= c.splitlines()
f.close()
male_ids=[]
for d in data:
    male_ids.append(d)

female_ids=[]
f=open(dir+'female_ids','r')
c= f.read()
data= c.splitlines()
f.close()
for d in data:
    female_ids.append(d)

list_files= sorted(os.listdir('../submitted_papers/'))
#list_files=['paper1399.txt','paper163.txt']
#list_files=['paper163.txt']
#fname='../submitted_papers/paper1399.txt'
#f=open(fname,'r')
#content=f.read()
#data=content.splitlines()

n_features=5000
#tf_vectorizer = CountVectorizer(max_df=0.5, min_df=2,
#                                max_features=n_features,
#                                stop_words='english')
#tf_vectorizer=TfidfTransformer(use_idf=False)
tf_vectorizer = TfidfVectorizer(input='content', analyzer='word', 
                     min_df = 2, stop_words = 'english', sublinear_tf=True, use_idf=False, max_features=20000)

corpus=[]
for filename in list_files:
    if len(filename.split('paper',1))>1:
        fname=filename.split('paper',1)[1]
        print(fname)
        sid=fname.split('.')[0]

        f=open('../submitted_papers/'+filename)
        content=f.read()
        data= content.splitlines()
        f.close()
        corpus.extend(data)

#tf = tf_vectorizer.fit_transform(data)
tf=tf_vectorizer.fit(corpus)
print(tf_vectorizer.get_feature_names())
print('length of features ',len(tf_vectorizer.get_feature_names()))
df=pd.DataFrame()
#preparing vectors for each of them now
row_list=[]
for filename in list_files:
    if len(filename.split('paper',1))>1:
        fname=filename.split('paper',1)[1]
       # print(fname)
        sid=fname.split('.')[0]

        f=open('../submitted_papers/'+filename)
        content=f.read()
        #data= content.splitlines()
        f.close()
        v=tf_vectorizer.transform(data)
        row_list.append(content) 

        if sid in male_ids:
            label='male' 
        else:
            label='female'
df=pd.DataFrame(row_list)
print(df.head())
df.columns=['text']
sdf = pd.SparseDataFrame(tf_vectorizer.transform(df['text']),columns=tf_vectorizer.get_feature_names(),  default_fill_value=0 )
print(sdf.head())
print(sdf.memory_usage())



        #dataset= [re.sub('\s+',' ', str(sent)) for sent in dataset]
        #dataset= [re.sub("\'", " ", sent) for sent in dataset]
        #data_words= list(sent_to_words(dataset))
        #sys.argv[1] command line argument for remove stopwords or not. preferrable use keep as other argument coz of print
        #if sys.argv[1]=='remove':
        #    data_words= remove_stopwords(data_words)
        #data_words= [j for sub in data_words for j in sub]  
        #unigram=data_words



#getting the TF matrix


# adding "features" columns as SparseSeries
#for i, col in enumerate(tf_vectorizer.get_feature_names()):
 #   df[col] = pd.SparseSeries(tf[:, i].toarray().ravel(), fill_value=0)