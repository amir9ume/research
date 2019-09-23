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


f= open('./data_info/male_abstracts', "r")
contents= f.read()
data_males= contents.splitlines()
dataset_males=[d.split() for d in data_males]
f.close()

f=open('./data_info/female_abstracts', "r")
contents=f.read()
data_females= contents.splitlines()
dataset_females= [d.split() for d in data_females]
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

def format_topics_sentences(ldamodel= lda, corpus= corpus, texts=data):
    sent_topics_df= pd.DataFrame()
   #get main topic in each document
    for i,row in enumerate (ldamodel[corpus]):
        print('row', row,'::',i)
       # row= sorted(row, key= lambda x: (x[1]), reverse =True)
       #get the dominant topic , percent contribution and keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j==0:
                wp= ldamodel.show_topic(topic_num)
                topic_keywords= ", ".join([word for word ,prop in wp]) 
                sent_topics_df= sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index= True)
            else:
                break
    sent_topics_df.columns=['Dominant_Topic', 'Percent_Contribution','Topic_Keywords']
    #adding original text to the end of the output. but why?
    contents= pd.Series(texts)
    sent_topics_df= pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)

  
z_words_male= remove_stopwords(dataset_males)
z_words_male= lemmatization(z_words_male, allowed_postags=['NOUN','ADJ','VERB','ADV'])
z_corpus_male= [id2word.doc2bow(text) for text in z_words_male ]        

z_words_female=remove_stopwords(dataset_females)
z_words_female= lemmatization(z_words_female, allowed_postags=['NOUN','ADJ','VERB','ADV'])
z_corpus_female= [id2word.doc2bow(text) for text in z_words_female]

#lda_result= lda[z_corpus]   
"""
male_topic_vector=np.array(lda[z_corpus_male])
female_topic_vector=np.array(lda[z_corpus_female])

print('\n male topic vectors',len(male_topic_vector))
print('\n female topic vectors',len(female_topic_vector))

#diff= np.linalg.norm(male_topic_vector - female_topic_vector)
#print(df)

#vis= pyLDAvis.gensim.prepare(lda,z_corpus,id2word)


"""

male_topics= lda[z_corpus_male]
print(male_topics)

female_topics= lda[z_corpus_female]
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
