from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import sys
import os
import re
import pandas as pd
model = FastText.load("trained_fasttext_nips.model")
z=model.wv.most_similar("cross entropy")
print(z) 

y=model.doesnt_match(" loss entropy reinforcement network hinge matrix sparse likelihood reward policy gradient factorization".split())
print(y)

import numpy as np  # Make sure that numpy is imported

#this number of features is same as dimension of your embedding
#here the embeddings have been already learnt previously


num_features=100
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.vocab)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       #if counter%1000. == 0.:
          # print ("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1
       counter=int(counter)
    return reviewFeatureVecs


u_final= pd.read_csv("bucketing_file_for_uni_sid")
dict_sid_to_bucket= pd.Series(u_final.bucket.values,index=u_final.sid).to_dict()
dict_bucket_corpus={}
for_panda_row_lists=[]   

list_files= sorted(os.listdir('../submitted_papers/'))
#list_files=['paper1407.txt', 'paper168.txt']
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

        preprocess=True
        if preprocess==True:
            data = re.sub('[^A-Za-z0-9]+', ' ', str(data))
            data=re.sub(r'[0-9]+', ' ', str(data))
        dict_row={'sid':sid,'bucket':bucket_result,'text':data}
        for_panda_row_lists.append(dict_row)

    df=pd.DataFrame(for_panda_row_lists)
print(df.head())

print('for sid 54',df.loc[df['sid']=='54'])
trainDataVecs = getAvgFeatureVecs( df['text'], model, num_features )
print('type of this train data vecs ', type(trainDataVecs))
print(len(trainDataVecs))
print(len(trainDataVecs[3]))
print('--------------------')
print(len(trainDataVecs[5]))


print('y labels for this are ')
y_buckets= u_final['bucket'].to_numpy()
print(y_buckets[22])


y_sid= u_final['sid'].to_numpy()
print(y_sid[22])



#print "Creating average feature vecs for test reviews"

#testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 15000
# This is fixed.
folder= '../submitted_papers/'
f= open(folder+'paper171.txt','r')
content= f.read()

EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(content)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['Consumer complaint narrative'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

"""