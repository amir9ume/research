import json
import os

import dask
import dask.bag as db
import dask.dataframe as dd
import pandas as pd

from fuzzywuzzy import fuzz

#defining my utility functions
def flatten(record):
    
    return {
        'id':record['id'],
        'title': record['title'],
        'authors': record['authors'],
        'year': record['year'],
       'venue': record['venue']
    }
def nips_checker(record):
    venue=record['venue']
    if venue is not None:
        if 'raw' in venue :
            venue= venue['raw']
            if venue is not None:
                venue=venue.lower()
                venue=venue.strip()
                #print(venue)
               # if 'nips' in venue or 'neural' in venue or 'neural information processing systems' in venue:
                if 'nips' in venue or 'neural information processing systems' in venue :
                    print(venue)
                    print(record['title'])
                    return True
    else:
        return False


folder= './data_info/data_dump_papers_aminer/'
list_files= sorted(os.listdir(folder))
number_of_files= len(list_files)

#print('sorted list of files', list_files)

b = db.read_text(folder+'*.txt').map(json.loads)
#z=b.filter(lambda record: 'title' in record and 'authors.name' in record and 'author.id' in record and 'year' in record and 'venue.raw' in record and record['year']==2012)
z=b.filter(lambda record: 'title' in record and 'authors' in record and 'year' in record and 'venue' in record and record['year']==2012 ) 

t=z.filter(lambda x: nips_checker(x))
print(t.take(10)[1]['venue'])
venues_data = t.map(flatten).to_dataframe()


print('-----<>-------')
print(venues_data.head())
print('-----<>-------')
print('length of this dataframe is ',len(venues_data.index))
venues_data.to_csv('venues_data_nips_2012',index=False)
"""
#print(df.head())

df= df[df.year ==2012]

df.columns=['paper_id','title','authors_list','year','venue']
#df= df.repartition(npartitions= 1+df.memory_usage(deep=True).sum().compute() // n )
df=df.repartition(partition_size="100MB")
print(df.head())
df['nips_check']=df['venue'].apply(nips_checker)
print(df.head())
#df= df.persist()
"""
"""
#This saving operation is going to fail. It is too large
#df.to_csv('all_papers_2012',index=False)
filename= './data_info/computer_science_authors'
authors=dd.read_csv(filename)
authors.columns= ['author_id','author_name','orgs_list','h_index','computer_check']
print('-----<><><>---------')
print(authors.head())

#do search operation for papers you want
nips_papers= pd.read_csv('all_paper_titles')
paper_titles=list(nips_papers['title'])
print(nips_papers.head())

#search for same papers in aminer papers table, as you have in nips_papers table. match by titles
def searching_for_nips12_papers(paper_from_table):
    paper_from_table= paper.lower()
    paper_from_table=paper.strip()
    
    for title in paper_titles:
        title= title.lower()
        title= title.strip()

        ratio = fuzz.ratio(title,paper_from_table)
        if ratio>70:
            print(paper_from_table)
            return True

df['paper_title_match']= df['title'].apply(searching_for_nips12_papers)
"""