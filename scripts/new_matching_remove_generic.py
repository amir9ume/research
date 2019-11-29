import pandas as pd
import numpy as np


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
import utilities

folder='../../'


papers=pd.read_csv(folder+'submissions_lda.csv',header=None)
reviewers=pd.read_csv(folder+'reviewers_lda.csv',header=None)

'''
Matching before removal
'''


papers.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
print(papers.columns)

del papers['12']
del papers['13']
del papers['7']
del papers['22']

print(papers)
papers=papers.values
print(papers)



reviewers.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']

del reviewers['12']
del reviewers['13']
del reviewers['7']
del reviewers['22']


print(reviewers)
reviewers=reviewers.values
print(reviewers)

matching= np.dot(papers,reviewers.T)
print('shape matching ',matching.shape)
print(matching)

'''
matching after removal 
'''
matching=pd.DataFrame(matching)
print('match after removal')


matching= matching.apply(lambda x: utilities.get_max_index(x,5),axis= 1)
matching=pd.DataFrame(matching)
matching['pid']=matching.index
matching.columns=['match_after_removal','pid']
print(matching)

'''
Matching before removal
'''

papers=pd.read_csv(folder+'submissions_lda.csv',header=None)
reviewers=pd.read_csv(folder+'reviewers_lda.csv',header=None)
papers=papers.values
reviewers=reviewers.values
match_before=np.dot(papers,reviewers.T)
match_before=pd.DataFrame(match_before)
print('match before')

match_before= match_before.apply(lambda x: utilities.get_max_index(x,5),axis= 1)
match_before=pd.DataFrame(match_before)
match_before['pid']=match_before.index
match_before.columns=['match_before_removal','pid']
print(match_before)

matching_compare=pd.merge(match_before,matching,how='left',on='pid')
print(matching_compare)

def find_intersection(record):
    a=record['match_before_removal']
    b=record['match_after_removal']
    return len(set(a) & set(b)) 

matching_compare['intersection']=matching_compare.apply(lambda x: find_intersection(x),axis=1 )    
print(matching_compare)

print('ratio of intersection ', matching_compare['intersection'].sum() /5)

zero_intersect= matching_compare.loc[matching_compare['intersection']==0]
print(zero_intersect)

all_intersect=matching_compare.loc[matching_compare['intersection']==5]
print(all_intersect)