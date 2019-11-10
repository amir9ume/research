import http.client
import json

conn = http.client.HTTPSConnection("api.aminer.org")

# headers = {
#     'cache-control': "no-cache",
#     'postman-token': "646ca127-5f26-94bd-4305-e2e8b112e911"
#     }

import unidecode

#unaccented_string = 

def query_with_name_and_org(name_person, org_name):
    org_par=''
    if len(org_name)>0:
        uni=org_name
        uni=uni.lower()
        uni=uni.replace(' ','%20')
        
        uni= uni.replace('\t','')
        uni= uni.replace('\r','')
        uni= uni.replace('\n','')
        uni= unidecode.unidecode(uni)
        org_par='&org='+uni
        #uni=uni.encode('utf-8').strip()
        
        
    n=name_person
    
    n=n.replace(' ','%20')
    person_name=n

    person_name= person_name.replace('\t','')
    person_name= person_name.replace('\r','')
    person_name= person_name.replace('\n','')

    person_name=unidecode.unidecode(person_name)
    name='name='+person_name
    query='/api/search/person/advanced?'
    query=query+name
    query=query+org_par
    #print('query was ', query)
    
    conn.request("GET",query )

    res = conn.getresponse()
    data = res.read()
    data=data.decode("utf-8")
    data=json.loads(data)
    d=data['result']
    if len(d)>0:
        return d[0]
    return d

def print_results_response(d, record):
    name=''
    dep=''
    n_cit=''
    h_ind=''
    n_pub=''
    if 'name' in d:
        #print('name of person is ', d['name'])
        name=d['name']
    if 'aff' in d and 'desc' in d['aff']:
        #print('department name is ', d['aff']['desc'])
        dep=d['aff']['desc']
    if 'indices' in d :
        #print('number of citations ', d['indices']['num_citation'], '--- h index is ',d['indices']['h_index'] )
        #print('number of publications is ', d['indices']['num_pubs'])
        n_cit=d['indices']['num_citation']
        h_ind=d['indices']['h_index']
        n_pub=d['indices']['num_pubs']
        
    print(name,'|',record[2],'|',record[3],'|',record[5],'|',record[6],'|',dep,'|',n_cit,'|',h_ind,'|',n_pub)
   # print('-----------------<>-------------')


#header = "Name| Gender| Submission ID| Author Position| Author size| country| uni"+'\n'
header = "name|sid|author_pos|country| uni|dep_from_aminer|n_cit|h_ind|n_pub"

#import pandas as pd
fname= './data_info/author_and_affiliation_from_metadata'
f= open(fname,'r')
reviewers= f.read().splitlines()

print(header)
#print('email','|','name','|','dep','|','n_cit','|','h_ind','|','n_pub')
ctr=0
for r in reviewers:
    if ctr>0:
    # print(r)
        r= r.split('|')
        #email=r[0]
        name_person=r[0]
        org_name=r[6]
    # print('name of person ', name_person,'---- org name is ', org_name)


        data= query_with_name_and_org(name_person,org_name)       
        if len(data)>0:
            print_results_response(data,r)
        else:
            #print('first query failed')
            data=query_with_name_and_org(name_person,'')
            print_results_response(data,r)   

    ctr+=1