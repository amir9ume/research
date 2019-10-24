from gensim.models import FastText
#sentences_ted='He has a bad disease. malaria is curable but dysentritis is not'
import lxml
import numpy as np
import os
from random import shuffle
import re
list_files= sorted(os.listdir('../submitted_papers/'))
  
#    full_corpus=[]
full_corpus=''
c=0
for filename in list_files:
    if len(filename.split('paper',1))>1:
        fname=filename.split('paper',1)[1]
        print(fname)
        sid=fname.split('.')[0]
        print('sid is ', sid)
        f=open('../submitted_papers/'+filename)
        content=f.read()
        #data= content.splitlines()
        data=content
        #full_corpus.extend(data)
        full_corpus=full_corpus+data
        
        f.close()


input_text=full_corpus
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
# store as list of sentences
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
# store as list of lists of words
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

print(len(sentences_ted))
print(len(sentences_ted[0]))
#maybe just add your corpus vocabulary to this

#parameters of the model
# min count: ignores all words with frequency lower than this
# but I think 2 is a good threshold instead of current 5
# size is dimensionality of word vectors
# window is maximum distance between the current and predicted word within a sentences
# sg=1 means skip grams. otherwise CBOW. we want skip grams as it is better for rare words
# workers: use these many threads to train the model
# more threads = faster training with multicore machines
#i dont know whether we need the negative sampling or not here

model_ted = FastText(sentences_ted, size=100, window=5, min_count=2, workers=4,sg=1)
z=model_ted.wv.most_similar("tensor")
print(z) 
total_words = model_ted.corpus_total_words
print('total words used in this corpus for fast text :', total_words)

#persisting the model to disk with this
from gensim.test.utils import get_tmpfile

#fname= get_tmpfile('trained_fasttext_nips.model')
fname='trained_fasttext_nips.model'
print('fname used is : ', fname )
model_ted.save(fname)



