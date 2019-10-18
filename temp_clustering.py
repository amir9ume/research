import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from sklearn import metrics


df= pd.read_csv('sid_bucket_each_document_vector_bow')
#removing the noisy bucket 6
df= df.loc[df['bucket']!=6]
print(df.head())
n_rows=len(df.index)

print(df.columns[3:])
X=df[df.columns[3:]]
true_k=5
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10)
km.fit(X)
labels=km.labels_

print(type(labels))

df['cluster_label']=labels

df['accuracy_check']=np.where(df['cluster_label']+1==df['bucket'],1,0)
print(df[:10])
print('accuracy is :',(df['accuracy_check'].sum())/n_rows )

sc=metrics.silhouette_score(X, labels, metric='euclidean')
print('score silhouette ', sc)
from sklearn.metrics import davies_bouldin_score
dav=davies_bouldin_score(X, labels) 
print('davies_bouldin_score ',dav)
"""
#print(df.mean())

bucket_1=df.loc[df['bucket']==1]
bucket_2=df.loc[df['bucket']==2]
bucket_3=df.loc[df['bucket']==3]
bucket_4=df.loc[df['bucket']==4]
bucket_5=df.loc[df['bucket']==5]

print(bucket_1.head())
print(bucket_2.head())


l1=bucket_1.mean()[2:]
print('type of l1 is ', type(l1))
print('length of l1', len(l1))
l1= l1.values
print('-----------------')
print('type of l1 is ', type(l1))
print('shape of l1 is ', l1.shape)
print('length of l1', len(l1))


l2=bucket_2.mean()[2:].values

l3=bucket_3.mean()[2:].values

l4=bucket_4.mean()[2:].values

l5=bucket_5.mean()[2:].values


l=[]
l.append(l1)
l.append(l2)
l.append(l3)
l.append(l4)
l.append(l5)


print('printing for l1',l1)
print('shape of l1', l1.shape)
print('type of l1', type(l1))

from sklearn.metrics.pairwise import cosine_similarity


#cc=cosine_similarity(l[0].reshape(1,-1),l[1].reshape(1,-1)) 
#print('Cosine similarity between ',1,' and ',2,cc)


#from sklearn.metrics.pairwise import paired_cosine_distances as pcd
cosine_sim=np.zeros((5,5))
for i in range(5):
    c_sim=[]
    for j in range(5):
        
        cc=cosine_similarity(l[i].reshape(1,-1),l[j].reshape(1,-1))    
        #   print('Cosine similarity between ',i,' and ',j,cc)
        c_sim.append(cc)
    #c_sim=np.asarray(c_sim)    
    #print(1- pcd(l[i], l[j]))
    cosine_sim[i]=c_sim
#cosine_sim=np.asarray(cosine_sim)
print(cosine_sim)

sns.heatmap(cosine_sim, annot=True, cmap=plt.cm.Reds)
plt.ylabel('buckets from 1 to 5')
plt.xlabel('buckets from 1 to 5')
plt.title('Heatmap cosine similarities Uni buckets-full paper')
plt.show()
"""