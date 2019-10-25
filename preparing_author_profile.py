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
#print(z.take(2))

def flatten(record):
    
    return {
        'id':record['id'],
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
        #print('True')
        #print(d)
        return True
    else:
       # print('False')
        return False

df['computer_check']= df['orgs'].apply(computer_department_name_checker )

#print(df.head())
#print('number of rows here',df.shape[0].compute())

x=df.loc[df['computer_check']==True].compute()
#print(x.head())
x.to_csv('computer_science_profs2',index=False)
