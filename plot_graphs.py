import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure()

b= pd.read_csv('improved_bid_score_delta_accepted_check_gender_first_author')
#del b['sub_id']
del b['email']
#del b['bid']
#del b['pid']
del b['score']
#del b['accepted_check']
#del b['rid']

#print(b.head())
#b.plot(kind='scatter', x='bid', y='score', color= 'blue')
#plt.xlabel('bid values')
#plt.ylabel('TPMS score values')
#plt.title('Scatter Plot- Bid values against TPMS scores')
#plt.show()

#b2= b
#del b2['accepted_check']

#bb= b.groupby('rid').agg({'delta':['mean','std']   }  )

#print('count all records',bb.count())

ba= b[b['accepted_check']==True ]
br= b[b['accepted_check']==False]
#print(ba.head())
#print(br.head())

#za= ba.groupby('rid').agg({'delta':['mean','std']}).reset_index()
#za=ba.groupby('pid').agg({'bid':['mean','std']}).reset_index()
baf= ba.loc[ba['gender']=='female']
bam= ba.loc[ba['gender']=='male']
print('baf std',baf['delta'].std())
print('baf average', baf['delta'].mean())
print('bam std',bam['delta'].std())
print('bam')

zaf=baf.groupby('pid').agg({'delta':['mean','std']}).reset_index()
zam=bam.groupby('pid').agg({'delta':['mean','std']}).reset_index()

#print(baf.head())
#print(bam.head())

#print(za.head())
#print('count accepted', za.count())
#za.columns=['pid','bid_mean','bid_std']
zaf.columns=['pid','delta_mean','delta_std']
zam.columns=['pid','delta_mean','delta_std']



print(zaf.head())
print(zam.head())
#print(zam.columns)

#zr= br.groupby('rid').agg({'delta':['mean','std'] }  ).reset_index()
#zr=br.groupby('pid').agg({'bid':['mean','std']}).reset_index()
brf=br.loc[br['gender']=='female']
brm=br.loc[br['gender']=='male']
zrf= brf.groupby('pid').agg({'delta':['mean','std']}).reset_index()
zrm= brm.groupby('pid').agg({'delta':['mean','std']}).reset_index()
#print('count rejected', zr.count() )
#zr.columns=['rid','delta_mean','delta_std']
#zr.columns=['pid','bid_mean','bid_std']
zrf.columns=['pid','delta_mean','delta_std']
zrm.columns=['pid','delta_mean','delta_std']
print('rejected females ',rf['delta_mean'].mean())

print('rejected males',zrm['delta_mean'].mean())

#print(zr.head())

bin= 80
#bin=np.arange(0,4)

fig, axes= plt.subplots(nrows=2 , ncols=2) 

#ax1= b.plot(kind='scatter', x='bid', y='delta', label='All papers', color='blue')
ax1= zrf.hist(column='delta_mean', label='rejected_females', color='blue', bins=bin, alpha=0.5, density= True, ax= axes[0,0])
#ax1= za.hist(column='bid_mean', label='accepted', color='blue', bins=bin, alpha=0.5, density= True)


#plt.show()

zrm.hist(column='delta_mean', label='rejected_males', color='red', alpha=0.7, bins= bin, density = True, ax= axes[0,1])
#zr.hist(column='bid_mean', label='rejected', color='red', ax=ax1, alpha=0.7, bins=bin, density=True)

plt.legend(loc='upper right')
#plt.title('Histogram of Delta Averaged across reviewers')
#plt.title('Histogram of Bid Averaged across papers')
#plt.title('Histogram of Delta averaged across papers ')
plt.title('Histogram of Delta averaged across papers- male vs female')

plt.ylabel('Normalised Frequency')
#plt.xlabel('Bid values Averaged across each of the papers')
plt.xlabel('Average Delta values grouped across papers')

plt.show()

#a= b[b['sid'].notnull()]
#a.plot(kind='scatter', x='bid', y='delta', label='Accepted papers', color='red', ax= ax1)

#plt.title('Delta values against Averaged values of bids across  Paper Ids')                     

#plt.xlabel('Averaged bid values across Paper Ids')
#plt.ylabel('Delta values')

#plt.plot(kind='scatter', x='bid', y='delta', color=['r','b'])

#plt.show()
#plt.show()

#b=pd.read_csv('bids_with_paper_id', header=1, sep=' ')
#del b['email']
#del b['sub_id']
#b.columns=['bid', 'pid']
#z=b.groupby('pid').count().reset_index()
#z.hist(column='bid', bins=40)

#plt.xlabel('Count of Bids received by a Paper')
#plt.ylabel('Count of Papers which received those many Bids')
#plt.title('Histogram of bids received by Papers')


#p= pd.read_csv('../bids_file.txt', header=None, sep=' ')
#p.columns=['sid','email','bid']
#del p['sid']

#p=pd.read_csv('improved_bid_score_delta_accepted_check')
#z= p.groupby(['accepted_check','pid']).agg({'bid':['mean','std']})
#z.columns=['bid_mean','bid_std']
#z.columns=['bid_mean','bid_min','bid_max','bid_std','delta_mean','delta_min','delta_max','delta_std']
#z=z.reset_index()
#z.to_csv('summarise_accepted_check_bids',index=False)


#z.hist(column='bid', bins=40)
#plt.xlabel('Count of bids made by reviewers')
#plt.ylabel('Count of such reviewers')
#plt.title('Histogram of bids made by reviewers')

#plt.show()





#b['accepted']= np.where(b['sid'].isnull(),False,True)
#sns.pairplot(b)
#b= pd.read_csv('summarise_accepted_check_bids')  
#ba= b[b['accepted_check']==True]
#br= b[b['accepted_check']==False]

#bin=np.arange(0,4)
#bin=20
#ax1= ba.hist(column='bid_std', label='accepted', color='b',bins=bin ,alpha=0.8)

#br.hist(column='bid_std', label='rejected', color='r',bins=bin ,alpha=0.3,ax= ax1)

#plt.legend(loc='upper right')
#plt.xlabel(' Std Bid values grouped across papers')
#plt.ylabel('Frequency')
#plt.title('Histogram of Bid Std across papers ')

#plt.show()
