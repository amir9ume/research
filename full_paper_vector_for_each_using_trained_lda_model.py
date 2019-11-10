import nltk; nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import LdaModel

import sys
import os

import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
stop_words= stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])

from gensim.test.utils import datapath
#loading previously trained lda model, corpus and dictionary
lda=LdaModel.load('./lda_trained/model_lda_nips12')
id2word= corpora.Dictionary.load('./lda_trained2/id2word_lda.dict')
corpus=corpora.MmCorpus('./lda_trained2/bow_corpus.mm')
#print('topics learnt by the lda model')

def cosine_similiarity(v1,v2):
    m1= np.linalg.norm(v1)
    m2= np.linalg.norm(v2)
    m= m1*m2
    d= np.dot(v1,v2)
    return d/(m)

num_topics=25
def make_vector_from_topic_distribution(document_topics):
    v= np.zeros(num_topics,dtype=float)
    for dc in document_topics:
        j=dc[0]
        v[j]= dc[1]
    return v

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def sent_to_words_new(sentence):
    yield(gensim.utils.simple_preprocess(str(sentence), deacc= True))

nlp= spacy.load('en',disable=['parser','ner'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts] 

def lemmatization(texts, allowed_postags=['NOUN','ADJ','VERB','ADV']):
    """https://spacy.io/api/annotation"""
    texts_out= []
    for sent in texts:
        doc= nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags]) 
    return texts_out


list_files= sorted(os.listdir('../submitted_papers/'))

document_vectors=[]
row_lists=[]
#l=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
#df = pd.DataFrame(columns=["sid","1","2","3","4","5",'6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25' ])
i=0
for filename in list_files:
    if len(filename.split('paper',1))>1:
        fname=filename.split('paper',1)[1]
        print(fname)
        sid=fname.split('.')[0]
    # print('sid is ',sid) 
        f=open('../submitted_papers/'+filename)
        content=f.read()
        df= content.splitlines()
        dataset= [d.split() for d in df]  
        f.close()

        #dataset=d
        #sid=d.split(',')[0]
        #dataset= [re.sub('\s+',' ', sent) for sent in dataset]
        #dataset= [re.sub("\'", " ", sent) for sent in dataset]
        dataset= list(sent_to_words_new(dataset))
        #doing without lemmatisation, in hope of picking up some traits
        data_words= remove_stopwords(dataset)
        
        #data_words= lemmatization(data_words, allowed_postags=['NOUN','ADJ','VERB','ADV'])
        data_words= [j for sub in data_words for j in sub]
        abstract_corpus= id2word.doc2bow(data_words)        
        
        #your lda is looking at all abstracts as independent it seems
        lda_result= lda[abstract_corpus] 
        s=0
        s=float(s)
        doc_topics= lda.get_document_topics( abstract_corpus, per_word_topics = False) 
        #pprint(doc_topics)
        for dc in doc_topics:
            s+=dc[1]
        print('sum of prob: ', s)

        v= make_vector_from_topic_distribution(doc_topics)
        print(len(v))
        print(sid)
        print(type(sid))
        print(v)   
        dict1= { "sid": sid,  "1":  v[0], "2": v[1],"3":  v[2], "4": v[3],"5":  v[4], "6": v[5],"7":  v[6], "8": v[7],
        "9":  v[8], "10": v[9],"11":  v[10], "12": v[11],"13":  v[12], "14": v[13],"15":  v[14], "16": v[15],
        "17":  v[16], "18": v[17],"19":  v[18], "20": v[19],"21":  v[20], "22": v[21],"23":  v[22], "24": v[23],
        "25":  v[24]}
        row_lists.append(dict1)

        #df.loc[i] = sid + list(v)
        #i+=1
    # df = df.append({ "sid": sid,  "1":  v[0], "2": v[1],"3":  v[2], "4": v[3],"5":  v[4], "6": v[5],"7":  v[6], "8": v[7],
    # "9":  v[8], "10": v[9],"11":  v[10], "12": v[11],"13":  v[12], "14": v[13],"15":  v[14], "16": v[15],
    # "17":  v[16], "18": v[17],"19":  v[18], "20": v[19],"21":  v[20], "22": v[21],"23":  v[22], "24": v[23],
    # "25":  v[24]}, ignore_index=True)

df=pd.DataFrame(row_lists)
df.to_csv('document_vectors_full_papers_nov',index=False)
print('data frame we are using',df.head(10))
# compare_all=[]
# for j in range(len(document_vectors)):
#         c= cosine_similiarity(document_vectors[0], document_vectors[j])        
#         compare_all.append(c)
#print('Cosine similarity',compare_all)
#pprint(document_vectors)