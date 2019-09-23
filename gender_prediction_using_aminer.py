
# coding: utf-8

# In[67]:

import os, random
import pandas as pd
import nltk
from nltk.corpus import names
nltk.download()
import gender_guesser.detector as gender

import requests
import os.path

# In[68]:


def read_lines(file_path):
    array = []
    with open(file_path, "r") as ins:
        for line in ins:
            line = line.replace("\n","")
            if len(line)>0:
                array.append(line)
    return array

def maintain_state(line_number):
    f= open('file_state.txt',"w+")
    f.write(str(line_number))
    f.close()

def check_state(lines):
    f=open('file_state.txt',"r")
    d=f.read()
    print('line_number:  ',d)
    f.close()
    d= int(d)
    d= d-1
    return  lines[d:],d

# In[69]:

def save_file(content,file_path):
#    try:
#        os.remove(file_path)
#    except OSError:
#        pass
    with open(file_path, 'a+') as out:
        out.write(content+'\n')
        out.close()

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


# In[74]:

def extract_authors(meta_file, processed_file):
    lines = read_lines(meta_file)
    if os.path.exists(processed_file):
        text=''    
    else:
        header = "Name, Gender, Submission ID, Author Position, Author size, Ethnicity, Univeristy Rank"+'\n'
        text = header

    URL='https://api.aminer.cn/api/kit/gender/api/ch?'
    file_path='./workingAmir/names_gender.txt'
    l=get_extended_names(file_path)
    ctr=0
     
    if os.path.exists('file_state.txt'):
        lines,ctr = check_state(lines)
    for line in lines:
        text_line = ''
        info = line.split('","')[0].split(',')
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
                 print(author)
                 if len(author)>=2:
               # if len(author)>=2 and len(author[0])>1:
                 #   predicted_gender = extract_gender(author[0],l)
                  #  predicted_gender = gender_detector(author[0])
                  #  if predicted_gender not in ['male', 'female']:
                  #      predicted_gender = extract_gender(author[0],l)
                    #print(author[0])
                    #print(predicted_gender)
                    #predicted_ethinicty = extract_ethnicity(author[0])
                     PARAMS={'name':author_name, 'aff':' '}
                     try:
                         r= requests.get(url=URL, params= PARAMS)
                         if 'json' in r.headers.get('Content-Type'):
                             
                             data=r.json()
                             predicted_gender=data['Final']['gender']
                         else:
                             predicted_gender='na' 
                     except requests.exceptions.RequestException as e:
                         predicted_gender='na'
                         print(e) 
                    # data=r.json()
                     print(predicted_gender)    
                     author_email = info[i+1]
                     email_domain = author_email.split(".")
                    #predicted_country = extract_country(domain[len(email_domain)-1])
                     text_line = author_name+','+predicted_gender+','+info[0]+','+str(author_rank)+','+str(size-1/2)+','+'Ethnicity'+','+'Univeristy Rank'
                     author_rank+=1
                     text +=text_line+'\n'
                     text_line = ''
                     i+=3
                 else:
                     i+=1
        ctr+=1
        maintain_state(ctr)
         #    i+=1
        if ctr%3==0:
            save_file(text,processed_file)

if __name__ == "__main__":
        d= gender.Detector()
#	extract_authors('metadata.csv','gender_predicted_all_features')
        extract_authors('metadata.csv','gender_predicted_api_results')



