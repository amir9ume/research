import nltk; nltk.download('stopwords')
from nltk.util import ngrams

import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from itertools import chain
import sys


stop_words= stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])

from gensim.test.utils import datapath
#loading previously trained lda model, corpus and dictionary
lda=LdaModel.load('./lda_trained/model_lda_nips12')
id2word= corpora.Dictionary.load('./lda_trained2/id2word_lda.dict')
corpus=corpora.MmCorpus('./lda_trained2/bow_corpus.mm')

#dic= corpora.Dictionary.load_from_text('../wordlist.txt')

#stores bag of words for comaprison
corpus_vectors=[]
for i in range(1,3):

    f= open('./data_info/'+sys.argv[i], "r")
    contents= f.read()
    data= contents.splitlines()
    dataset=[d.split() for d in data]

 
#deacc = True will remove punctuations
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts] 

    dataset= [re.sub('\s+',' ', str(sent)) for sent in dataset]
    dataset= [re.sub("\'", " ", sent) for sent in dataset]
    data_words= list(sent_to_words(dataset))
    data_words= remove_stopwords(data_words)
    data_words= [j for sub in data_words for j in sub]

    bigram= list(ngrams(data_words,2))
    unigram=data_words

#id is first field. Use both title and abstract text for finding out lda topics
#    nlp= spacy.load('en',disable=['parser','ner'])
    """https://spacy.io/api/annotation"""

#try once without lemmatization on bag of words

    def lemmatization(texts, allowed_postags=['NOUN','ADJ','VERB','ADV']):
        texts_out= []
        for sent in texts:
            doc= nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags]) 
        return texts_out

#z_words= lemmatization(z_words, allowed_postags=['NOUN','ADJ','VERB','ADV'])

    unigram= [u.split() for u in unigram]
#print(unigram[:20])
    unigram= [j for sub in unigram for j in sub]
    unigram_corpus= id2word.doc2bow(unigram)         

    number_of_words=0
    for id,freq in unigram_corpus:
        number_of_words+= freq
    print('total number of words', number_of_words)

    n_uni_corpus= [(id, freq/number_of_words) for id,freq in unigram_corpus]
#print(n_uni_corpus[:20])
    corpus_vectors.append(n_uni_corpus)
    k= [(id2word[id],freq) for id, freq in n_uni_corpus]   
    print(k[:20])


from nltk.cluster import KMeansClusterer
import nltk
nc=2
#kclusterer= KMeansClusterer(nc,distance=nltk.cluster.util.cosine_distance,repeats=25)
#as_c= kclusterer.cluster(corpus_vectors,assign_clusters=True)
#print(as_c)

plt.hist(corpus_vectors[0])
plt.show()


#bigram corpus will need bigram dictionary of its own. this unigram dictionary will not be able to catch it
#bigram =[j for sub in bigram for j in sub]
#print(bigram)
#bigram_corpus= [id2word.doc2bow(text) for text in bigram] 
#kb=[[(id2word[id], freq) for id,freq in cp] for cp in bigram_corpus]
#print(kb[:20])
#print(bigram_corpus)

#ld= lda[z_corpus]   



