
import re
import lookup_word_id

f=open('my_lda_topic_probs.txt')
content=f.read()
#df= content.splitlines()

df= content.split(')')

def cleaner(line):
    rest=''
    ar=line.split(',')
    topic_number=ar[0]
    if len(ar)>1:
        rest=ar[1]
    keywords=re.findall(r'"([^"]*)"', rest)
        
    print('topic number is ', topic_number)
    #print('rest is ', rest)
    #print('keywords are ',keywords)
    lookup_word_id.find_names_of_word_ids(keywords)

ctr=0
for d in df:
   # print(type(d))
    
    d=d.replace('(','')
    d=d.replace('[','')
    d=d.replace(']','')
    #d=d[:-1]
    if ctr !=0:
        d=d[1:]
    #print(d)
    
    cleaner(d)
    print('------')
    ctr+=1
