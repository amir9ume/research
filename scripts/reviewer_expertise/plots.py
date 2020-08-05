import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 

data_path= "../rev_dict_for_plot"

with open(data_path, "rb") as input_file:
    rev_dict = pickle.load(input_file)


#get count of files for each of the given reviewer ids
folder="../../../neurips19_anon/archive_papers/"


rev_files={}
y=[]
x=[]
for rev_id in rev_dict:
    path=folder+(str(rev_id))
    files= os.listdir( path )
    text_files= [file for file in files if '.txt' in file]
    rev_files[rev_id]= len(text_files)
    y.append(len(text_files))
    loss= rev_dict[rev_id]
    loss=np.log(loss)
    x.append(loss)
print('done')
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#dy=reviewer_proxy_merge.describe()['std']

plt.scatter(x, y,  alpha=0.5)
plt.ylabel('Number of reviewer papers')
plt.xlabel('Log Avg Mean Square error over bids')
plt.title('Number of reviewer papers against MSE')
plt.show()
