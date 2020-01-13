import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '../')
import utilities

def read_papers_topic_lda_prepare_data_frame(filename):
    papers=pd.read_csv(filename,header=None)
    papers.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
    print(papers.columns)
    return papers

def read_reviewers_topic_lda_prepare_data_frame(reviewers_filename):
    reviewers=pd.read_csv(reviewers_filename,header=None)
    reviewers.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
    return reviewers

def delete_topic_columns(list_topics_to_delete,papers,reviewers):
    for topic in list_topics_to_delete:
        del papers[topic]
        del reviewers[topic]
    print('--printing after deleting the topics--')
    print(papers)
    print('--printing reviewers after deleting the topics--')
    print(reviewers)

def matching_using_dot_product_topk_assignments(papers,reviewers,topk,name):

    matching_using_dot_product= np.dot(papers,reviewers.T)
    matching_using_dot_product_dataframe=pd.DataFrame(matching_using_dot_product)
    
    matching_using_dot_product_dataframe= matching_using_dot_product_dataframe.apply(lambda x: utilities.get_max_index(x,topk),axis= 1)
    matching_using_dot_product_dataframe=pd.DataFrame(matching_using_dot_product_dataframe)
    matching_using_dot_product_dataframe['pid']=matching_using_dot_product_dataframe.index
    name='match_'+name+'_removal'
    matching_using_dot_product_dataframe.columns=[name,'pid']
    print(matching_using_dot_product_dataframe)
    return matching_using_dot_product_dataframe, matching_using_dot_product



folder='../../'
filename_papers=folder+'submissions_lda.csv'
reviewers_filename=folder+'reviewers_lda.csv'
list_topics_to_delete=['12','13','7','22']

def get_new_assignment_paper_reviewer_after_deletion_topics(file_papers,file_reviewers,list_topics_to_delete):

    
    print('file name papers ',file_papers)
    papers=read_papers_topic_lda_prepare_data_frame(file_papers)
    
    reviewers=read_reviewers_topic_lda_prepare_data_frame(file_reviewers)

    delete_topic_columns(list_topics_to_delete,papers,reviewers)
    papers=papers.values
    reviewers=reviewers.values
    topk=5
    print('--matches after removal of topics--')
    matching_using_dot_product_dataframe, matching_using_dot_product= matching_using_dot_product_topk_assignments(papers,reviewers,topk,name='after')

    print(matching_using_dot_product)
    print(matching_using_dot_product.shape)
    return matching_using_dot_product_dataframe, matching_using_dot_product


matching_using_dot_product_dataframe, matching_using_dot_product= get_new_assignment_paper_reviewer_after_deletion_topics(filename_papers,reviewers_filename,list_topics_to_delete)

'''


Matching before removal


papers=pd.read_csv(folder+'submissions_lda.csv',header=None)
reviewers=pd.read_csv(folder+'reviewers_lda.csv',header=None)
papers=papers.values
reviewers=reviewers.values





print('--matches before removal--')
match_before,_=matching_using_dot_product_topk_assignments(papers,reviewers,topk,name='before')



matching_compare=pd.merge(match_before,matching_using_dot_product_dataframe,how='left',on='pid')
print(matching_compare)

def find_intersection(record):
    #print('--finds intersection between reviewers before and after removal of generic topics--')
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
'''