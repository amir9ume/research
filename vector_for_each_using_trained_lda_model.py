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
#pprint(lda.print_topics())
#print(len(lda.print_topics()))
#print(len((lda.print_topics()[1])))

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



#abstracts to read and build lda vectors. Read the folder name as command line argument
file= sys.argv[1]    
f= open('./data_info/'+file, "r")
contents= f.read()
data= contents.splitlines()
f.close()
document_vectors=[]

df = pd.DataFrame(columns=["sid", "document_vector"])

for d in data:
    dataset=d
    sid=d.split(',')[0]
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
    
    df = df.append({ "sid": sid,  "document_vector":  doc_topics }, ignore_index=True)


    #pprint(doc_topics)
    for dc in doc_topics:
        s+=dc[1]
    #print('sum of prob: ', s)

    v= make_vector_from_topic_distribution(doc_topics)
    document_vectors.append(v)

print(df.head(10))
print(df.count())
df.to_csv('document_vectors_all_abstracts_sid',index=False)
compare_all=[]
for j in range(len(document_vectors)):
        c= cosine_similiarity(document_vectors[0], document_vectors[j])        
        compare_all.append(c)
#print('Cosine similarity',compare_all)
#pprint(document_vectors)