import pandas as pd
import numpy as np

folder_path= './data_info/'

g= pd.read_csv(folder_path+'gender_predicted_api_results')


g.columns= ['Name', 'gender', 'sid', 'author_pos', 'author_size',
       'Country', 'uni_rank']
del g['Country']
del g['author_size']


g=g.loc[g['Name']!="Name"]
g['sid2']= g['sid'].apply(int)
del g['sid']
g['pos2']= g['author_pos'].apply(int)
del g['author_pos']
g.columns=['Name', 'gender','uni_rank','sid','author_pos']

doc_vec= pd.read_csv('document_vectors_all_abstracts_sid2')
doc_vec= doc_vec.loc[doc_vec['sid']!='sid']
#print(doc_vec.head(10))

max_author= g.groupby('sid', sort= False)['author_pos'].max()
g_with__max_author= pd.merge(g,max_author, how='left',on='sid')
g_with__max_author.columns=['Name', 'gender', 'uni_rank', 'sid', 'author_pos',  'max_author']

g_only_max_author= g_with__max_author.loc[g_with__max_author['author_pos']==g_with__max_author['max_author']]

#print(g_only_max_author.head())
doc_vec['sid']=doc_vec['sid'].astype(int)

g_with_vector= pd.merge(g_only_max_author,doc_vec, how='left', on='sid')
#print(g_with_vector.head(10))

del g_with_vector['Name']
del g_with_vector['uni_rank']
del g_with_vector['author_pos']
del g_with_vector['max_author']

#print(g_with_vector.head(10))
#g_with_vector.to_csv('./data_info/abstracts_vectors_gender_sid2', index=False)

features=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"]
#g_with_vector['1']=g_with_vector['1'].astype(int)

#print(g_with_vector['1'].max())
#g_with_vector["1"]=np.nan_to_num(g_with_vector["1"])
X = g_with_vector["1"]
X= X.apply(float)
y = g_with_vector['gender']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#print(X_train)

from sklearn.svm import SVC
#using a support vector machine
s_clf = SVC()

X_train=np.array(X_train)
X_train= np.nan_to_num(X_train)
X_train=X_train.reshape(-1,1)
s_clf.fit(X_train,y_train)

X_test=np.array(X_test)
X_test=np.nan_to_num(X_test)
s_prediction = s_clf.predict(X_test.reshape(-1,1))
print(s_prediction)

from sklearn.metrics import accuracy_score
s_acc = accuracy_score(s_prediction,y_test)

print(s_acc)