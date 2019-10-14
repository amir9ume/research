
# coding: utf-8

# In[67]:

import os, random
import pandas as pd
import nltk
from nltk.corpus import names
nltk.download()
import gender_guesser.detector as gender
import requests

import json 
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



# In[68]:


def read_lines(file_path):
    array = []
    with open(file_path, "r") as ins:
        for line in ins:
            line = line.replace("\n","")
            if len(line)>0:
                array.append(line)
    return array


# In[69]:

def save_file(content,file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass
    with open(file_path, 'a') as out:
        out.write(content+'\n')


# In[70]:

def gender_features(word):
    """ feature extractor for the name classifier
    The feature evaluated here is the last letter of a name
    feature name - "last_letter"
    """
    n=len(word)
    dict={}
  #  for j in range(1,n):
  #      dict[j]=word[-j:]
  #  return dict	
    return {"last_letter": word[-1], "last_two_letters": word[-2:]}  	

# In[71]:

def gender_detector(name):
    return (d.get_gender(name))

def get_extended_names(file_path):
    lines=read_lines(file_path)
    l=[]
    for line in lines:
        u= line.split(' ')
        if u[0]== 'F':
            l.append((u[2],'female'))
        elif u[0]=='M':
            l.append((u[2], 'male'))
    return l

def gender_api(author_name):
    URL='https://api.aminer.cn/api/kit/gender/api/ch?'
    PARAMS={'name':author_name, 'aff':''}
    r= requests.get(url= URL, params= PARAMS)
    data=r.json()
    return data['Final']['gender']

def extract_gender(author_name,l):
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    xtended_names= list(set(labeled_names+ l))
    random.shuffle(xtended_names)
    featuresets = [(gender_features(n), gender) for (n, gender) in xtended_names]
    train_set, test_set = featuresets[100:], featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    predicted_gender = classifier.classify(gender_features(author_name))
    return predicted_gender


# In[72]:

def extract_ethnicity(author_name):
    MODELFN = "models/wiki/lstm/wiki_ln_lstm.h5"
    VOCABFN = "models/wiki/lstm/wiki_ln_vocab.csv"
    RACEFN = "models/wiki/lstm/wiki_race.csv"

    MODEL = resource_filename(__name__, MODELFN)
    VOCAB = resource_filename(__name__, VOCABFN)
    RACE = resource_filename(__name__, RACEFN)
    vdf = pd.read_csv(VOCAB)
    vocab = vdf.vocab.tolist()

    rdf = pd.read_csv(RACE)
    race = rdf.race.tolist()

    model = load_model(MODEL)


#def get_country():
    

# In[74]:

def extract_authors(meta_file, processed_file):
    lines = read_lines(meta_file)
    header = "Name| Gender| Submission ID| Author Position| Author size| Country| Univeristy"+'\n'
    text = header
    file_path='./workingAmir/names_gender.txt'
    l=get_extended_names(file_path)
    folder='./workingAmir/data_info/'
    file_json=folder+'universities.json'
    with open(file_json,'r') as f:
        datastore= json.load(f)
    
    c=0
 #   for d in datastore:
    for line in lines:
        text_line = ''
        info = line.split('","')[0].split(',')
        #print(info)
        
    # print(info[])
    
        size = len(info)
        i=1
        size = size-1
        author_rank = 1
        while i<size and line!=lines[0]:
            author_name = info[i]
            if author_name.startswith('"'):
                i+=10
            else:
                author = author_name.split(" ")
                #print(author)
                if len(author)>=2 and len(author[0])>1:
                 #   predicted_gender = extract_gender(author[0],l)
                 #   predicted_gender = gender_detector(author[0])
                    predicted_gender='gender'                 
                  #  if predicted_gender not in ['male', 'female']:
                  #      predicted_gender = extract_gender(author[0],l)
                    #print(author[0])
                    #print(predicted_gender)
                    #predicted_ethinicty = extract_ethnicity(author[0])
                    author_email = info[i+1]
                    if len(info)>(i+2):
                        author_uni_from_metadata=info[i+2]
                    university_name= author_uni_from_metadata
                    if len(author_email.split("@"))>1:
                        email_domain = author_email.split("@")[1]
                        y=None
                        xx=None
                        flag=True
                        for d in datastore:
                            if email_domain in d['domains']:
                                flag=False
                                y=d['name']
                                xx=d['country']
                                break
                            elif fuzz.ratio (email_domain, d['domains'])>75:
                                flag=False
                                y=d['name']
                                xx=d['country']
                        z=email_domain.split('.')
                        x= z[-2] +'.'+z[-1]
                        if flag==True:
                            for d in datastore:
                                if x in d['domains']:
                                    y=d['name'] 
                                    xx=d['country']
                                    break
                                elif fuzz.ratio (x, d['domains']) >75:
                                    y=d['name']
                                    xx=d['country'] 
                        if y is  None:
                            y='others'
                            xx='unknown'
                        if len(author_uni_from_metadata)<4 and y!='others':
                            university_name=y 
                        predicted_country=xx             
                    else:
                        if len(author_uni_from_metadata)<2:
                            university_name= 'others'
                        predicted_country='unknown'
                    #predicted_country = extract_country(domain[len(email_domain)-1])
                    text_line = author_name+'|'+predicted_gender+'|'+info[0]+'|'+str(author_rank)+'|'+str(size-1/2)+'|'+predicted_country+'|'+university_name
                    print(email_domain, predicted_country, university_name)
                    author_rank+=1
                    text +=text_line+'\n'
                    text_line = ''
                    i+=3
                else:
                    i+=1
    save_file(text,processed_file)
    
if __name__ == "__main__":
        d= gender.Detector()
#	extract_authors('metadata.csv','gender_predicted_all_features')
        extract_authors('metadata.csv','author_and_affiliation_from_metadata')



