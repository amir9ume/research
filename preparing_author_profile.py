import json
import os

import dask
import dask.bag as db

folder= './data_info/data_dump_aminer/'
list_files= sorted(os.listdir(folder))

b = db.read_text(folder+'*.txt').map(json.loads)

#print(b.take(2))

#if 'computer science ' in d  or 'computer' in d or 'computer' in z
#y=b.filter(lambda record:  record['orgs'] is not None )
#yy=y.take(1)
# for yyy in yy:
#     print(yyy)
#     print('------')
#     print(type(yyy))
#     print(yyy['h_index'])

z=b.filter(lambda record: 'orgs' in record and 'name' in record and 'h_index' in record ) 
print(z.take(2))

def flatten(record):
    
    return {
        'name': record['name'],
        'orgs': record['orgs'],
        #'org': record['org'],
       'h_index': record['h_index']
    }

df = z.map(flatten).to_dataframe()

def computer_department_name_checker(dept):
    dept= [d.lower() for d in dept]
    d= [ x.split() for x in dept]
    d=[j for sub in d for j in sub]
    
    if 'computer science ' in d  or 'computer' in d :
        print('True')
        print(d)
        return True
    else:
       # print('False')
        return False

df['computer_check']= df['orgs'].apply(computer_department_name_checker )

print(df.head())
#print('number of rows here',df.shape[0].compute())

x=df.loc[df['computer_check']==True].compute()
print(x.head())
x.to_csv('computer_science_profs',index=False)

"""
fname='aminer_authors_19.txt'
data = []


b = db.read_text('data/*.json').map(json.loads)



file=folder+fname
with open(file) as f:
    for line in f:
        data.append(json.loads(line))
    f.close()
c=0

dict_author={}
for author in data:
    if 'orgs' in author:
        if len(author['orgs'])>0  :
            dept= author['orgs']
            dept= [d.lower() for d in dept]
            d= [ x.split() for x in dept]
            zz=False
            if 'org' in author:
                zz=True
                z= author['org']
            #print(d)
            if zz==True:

                if 'computer science ' in d  or 'computer' in d or 'computer' in z:
                    n=author['name']
                    orgs=author['orgs']
                    h=author['h_index']
                    n_pubs= author['n_pubs']
                    dict_author[n]={'h_index':h, 'orgs': orgs, 'n_pubs':n_pubs}
                    print(dict_author)
                    c+=1
                if c>100:
                    break
            else:
                if 'computer science ' in d  or 'computer' in d :
                    n=author['name']
                    orgs=author['orgs']
                    h=author['h_index']
                    n_pubs= author['n_pubs']
                    dict_author[n]={'h_index':h, 'orgs': orgs, 'n_pubs':n_pubs}
                    print(dict_author)
                    c+=1
                if c>100:
                    break
"""