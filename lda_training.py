import nltk; nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from pprint import pprint

#gensim 
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import LdaModel

#spacy for lemmatisation
import spacy
#plot tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
#matplotlib inline

#import loggin
#logging.basicConfig(format= '%(asctime)s : %(levelname)s : %(message)s ', level= logging.ERROR) 

import warnings
warnings.filterwarnings("ignore", category= DeprecationWarning)

from nltk.corpus import stopwords
stop_words= stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])

#df=pd.read_csv('../submitted_papers/allpapers.txt')
f=open('../submitted_papers/allpapers.txt')
content=f.read()
df= content.splitlines()
dataset= [d.split() for d in df]  
#print(df)

id2word= corpora.Dictionary(dataset)
print(id2word[5])
#saved the dictionary
id2word.save('id2word_lda.dict')
#create corpus
texts= dataset

#term document frequency , ie bag of words
corpus= [id2word.doc2bow(text) for text in texts]

#saving the corpus
corpora.MmCorpus.serialize('bow_corpus.mm',corpus)


#print([[(id2word[id], freq) for id , freq in cp ] for cp in corpus[:1] ])

#running for 25 topics only as is done in TPMS
lda_model= gensim.models.ldamodel.LdaModel(corpus= corpus, id2word= id2word, num_topics=25, random_state=100, update_every=1 , chunksize=100,passes=10, alpha='auto', per_word_topics= True)

pprint(lda_model.print_topics())
doc_lda= lda_model[corpus]

#saving the model
from gensim.test.utils import datapath

#temp_file= datapath('model_lda_nips12')
#lda_model.save(temp_file)
lda_model.save('model_lda_nips12')


#visualisation
#pyLDAvis.enable_notebook()
#vis=pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#pyLDAvis.save_html(vis, 'LDA_Visualization.html')

from matplotlib import pyplot as plt
from wordcloud import WordCloud , STOPWORDS
import matplotlib.colors as mcolors

cols= [color for name, color in mcolors.TABLEAU_COLORS.items()]
cloud= WordCloud(stopwords= stop_words, background_color='white', width= 2500, height= 1800, max_words=10, colormap='tab10', color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)

topics= lda_model.show_topics(formatted=False)

fig,axes= plt.subplots(3,2, figsize= (15,15), sharex= True, sharey=True)

for i , ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words= dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic' + str(i), fontdict= dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
    
