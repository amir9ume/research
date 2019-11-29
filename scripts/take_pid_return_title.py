import pandas as pd 

folder='../old_work/'
abstracts=pd.read_csv(folder+'extracted_abstracts.csv',header=None)
abstracts.columns=['sid','title','abstract']
#print(abstracts)

sid_to_pid=pd.read_csv('../../submissionID_to_paperID.csv',header=None)
sid_to_pid.columns=['sid','pid']
#print(sid_to_pid)

abstracts_pid_sid=pd.merge(abstracts,sid_to_pid,how='left',on='sid')
#print(abstracts_pid_sid)


import sys

pid=int(sys.argv[1])
print('--------------')
print(abstracts_pid_sid.loc[abstracts_pid_sid['pid']==pid]['title'].item())
print('--------------')
print(abstracts_pid_sid.loc[abstracts_pid_sid['pid']==pid]['abstract'].item())

# 0 intersection pids 1,4,10,13,19
# 5 intersection pids 2,5,9,12,14