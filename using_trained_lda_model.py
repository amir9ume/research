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
#pprint(lda.print_topics())
#print(len(lda.print_topics()))
#print(len((lda.print_topics()[1])))

#abstracts to read and build lda vectors. Read the folder name as command line argument
folder= sys.argv[1]
list_files= os.listdir('./data_info/'+folder)
for file_name in list_files:
    print(file_name)
    
    f= open('./data_info/'+folder+'/'+file_name, "r")
    contents= f.read()
    data= contents.splitlines()
    dataset=[d.split() for d in data]
    f.close()
    

    #dataset= [re.sub('\s+',' ', sent) for sent in dataset]
    #dataset= [re.sub("\'", " ", sent) for sent in dataset]
    #dataset= list(sent_to_words(dataset))

    #try once without lemmatization on bag of words


    #id is first field. Use both title and abstract text for finding out lda topics

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

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

    #doing without lemmatisation, in hope of picking up some traits
    data_words= list(sent_to_words(dataset))
    data_words= remove_stopwords(data_words)
    data_words= [j for sub in data_words for j in sub]

    #data_words= lemmatization(data_words, allowed_postags=['NOUN','ADJ','VERB','ADV'])
    abstract_corpus= id2word.doc2bow(data_words)        

    #your lda is looking at all abstracts as independent it seems
    lda_result= lda[abstract_corpus] 
    s=0
    doc_topics= lda.get_document_topics( abstract_corpus, per_word_topics = False) 
    pprint(doc_topics)
    for dc in doc_topics:
        s+=dc[1]
    print('sum of prob: ', s)
  

    #print('doc topics',doc_topics)
    #print('word topics', word_topics)
    #print('phi values', phi_values)

    #print(lda_result.print_topics())
    #print('length lda result',len(lda_result.print_topics))
    #print('length of one of the lda result',len(lda_result.print_topics[1]))
    #print(type(lda_result))
    """
    male_topic_vector=np.array(lda[z_corpus_male])
    female_topic_vector=np.array(lda[z_corpus_female])

    print('\n male topic vectors',len(male_topic_vector))
    print('\n female topic vectors',len(female_topic_vector))
    gender_bin_abstracts

    #diff= np.linalg.norm(male_topic_vector - female_topic_vector)
    #print(df)

    #vis= pyLDAvis.gensim.prepare(lda,z_corpus,id2word)


    """

    """
    #code section to visualize the Word Cloud
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud , STOPWORDS
    import matplotlib.colors as mcolors

    cols= [color for name, color in mcolors.TABLEAU_COLORS.items()]
    cloud= WordCloud(stopwords=stop_words,background_color='white',width=2500,height=1800, max_words=10, colormap='tab10', color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)
    topics= lda.show_topics(formatted=False)

    fig, axes = plt.subplots(3,3, figsize=(20,20), sharex= True, sharey= True) 

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words=dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic' +str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0,y=0)
    plt.tight_layout()
    plt.show()  


    """
