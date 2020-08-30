import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 
import torch 

folder="../meeting_183/"
data_path= folder+"reviewer_entropies_for_plot"

with open(data_path, "rb") as input_file:
    reviewer_entropies = pickle.load(input_file)


rev= pd.DataFrame(reviewer_entropies.items())
rev.columns=['rev_id','entropy']



def Average(lst): 
    return sum(lst) / len(lst)


distance_path=folder+"reviewer_distances_mean_for_plot"
with open(distance_path, "rb") as input_file:
    reviewer_distances_mean = pickle.load(input_file)

distance_path=folder+"reviewer_distances_attn_for_plot"
with open(distance_path, "rb") as input_file:
    reviewer_distances_attn = pickle.load(input_file)


def get_reviewer_distance_mean(record):
    rid= int(record['rev_id'])
    if rid in reviewer_distances_mean:
        distance= (reviewer_distances_mean[rid])
        return Average(distance).item()#[0].item() 
    else:
        return -1

def get_reviewer_distance_attn(record):
    rid= int(record['rev_id'])
    if rid in reviewer_distances_attn:
        distance= (reviewer_distances_attn[rid])
        return Average(distance).item()#[0].item() 
    else:
        return -1


rev['distance_mean']=rev.apply(lambda x: get_reviewer_distance_mean(x) ,axis=1)
rev['distance_attn']=rev.apply(lambda x: get_reviewer_distance_attn(x) ,axis=1)

rev= rev.loc[rev['distance_mean']!=-1]
rev= rev.loc[rev['distance_attn']!=-1]
print(rev.sample(5))


max_val= max(rev['distance_mean'].max(),rev['distance_attn'].max())


def remove_outliers(distances):
    mean= distances.mean()
    std= distances.std()
    threshold= mean + 2* std
    distances[distances > threshold] = 0
    return distances    




#plot against mean d1
x= rev['entropy']
y=rev['distance_mean']

mean= rev['distance_mean'].mean()
std= rev['distance_mean'].std()
threshold= mean + 2* std
rev=rev[rev['distance_mean']<threshold]


#y= remove_outliers(y)
plt.scatter(x, y,  alpha=0.5, c= 'r',label="mean dist")
plt.xlabel('entropy of reviewer papers')
plt.ylabel('KL distance ')
#plt.yticks(np.arange(0, max_val))
#plt.title('entropy versus KL distance mean')
#



mean_2= rev['distance_attn'].mean()
std_2= rev['distance_attn'].std()
threshold_2= mean_2 + 2* std_2
rev=rev[rev['distance_attn']<threshold_2]

rev['delta']= rev['distance_mean'] - rev['distance_attn']
rev['delta']=rev['delta'].abs()
rev= rev.sort_values(by ='delta',ascending=False )
print(rev.head(10))
# #plot against attn d2
x= rev['entropy']
y=rev['distance_attn']
plt.scatter(x, y,  alpha=0.5,c='g',label="attn dist")
# plt.xlabel('entropy of reviewer papers')
# plt.ylabel('KL distance attn')
#plt.yticks(np.arange(0, max_val))
#plt.title('entropy versus KL distance attn')
plt.legend(loc="upper right")
plt.show()



