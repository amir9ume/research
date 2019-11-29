import pandas as pd
import numpy as np
import utilities

'''
The code below is for finding out entropy values of papers based on topic probabilities on each of the 25 topics
Runs through all the p values, and calculates, -\sum_i p_i log p_i
----------
First section is for each of the paper representations

----------
Second section is about each of the reviewers
This is picked up from Laurents represenations only

Commenting out sections of the code , for ease of making changes

'''



def calculate_entropy_element(p): 
    
    return -1 * p * np.log2(p) 


# doc_vecs= pd.read_csv('./data_info/document_vectors_full_papers_nov')
# doc_vecs['entropy_vals']= (doc_vecs.iloc[:, 1:].apply(lambda x : calculate_entropy_element(x))).fillna(0).sum(axis = 1, skipna = True)
# print('--------------<>---------------')
# print(doc_vecs[:-8])
# print(len(doc_vecs.index))
# doc_vecs.to_csv('./data_info/doc_vecs_lda_entropy',index=False)
# print('prepared for each of the 1424 papers')
# print('--------------<>---------------')




print('now calculating entropy values for each of the reviewer representations')
reviewers_lda_topics= pd.read_csv('../reviewers_lda.csv', header=None)


#assumption is that reviewer ids are same as index ids here
reviewers_lda_topics['entropy_vals']= reviewers_lda_topics.apply(lambda x: calculate_entropy_element(x)).fillna(0).sum(axis=1, skipna=True)

print('entropy of reviewers')
print(reviewers_lda_topics.head())

#reviewers_lda_topics.to_csv('./data_info/reviewers_lda_entropy')

'''
This section of code finds out the top 3 topics for each person, and checks if they belong to one of the generic areas
or the niche research areas
'''

del reviewers_lda_topics['entropy_vals']

reviewers_with_topics_of_interest= reviewers_lda_topics.apply(lambda x: utilities.get_max_index(x,5),axis= 1)
reviewers_with_topics_of_interest=pd.DataFrame(reviewers_with_topics_of_interest)
reviewers_with_topics_of_interest['rid'] = reviewers_with_topics_of_interest.index
reviewers_with_topics_of_interest.columns= ['topics_of_interest','rid']
print('reviewers topics of interest are ')
print(reviewers_with_topics_of_interest[:10])


print('---finding top assignments ----')
scores_lda= pd.read_csv('../scores_lda.csv',header=None)
print('scores lda for all reviewers', scores_lda)


'''
found average of top 5 max assignment scores for each reviewer
'''
top_k=5

def investigate_reviewers_return_var_avg_topk(scores_lda, top_k):
    print('----- Now investigating reviewers -----')
    avg_of_max_assignment_scores_each_reviewer= scores_lda.apply(lambda x: utilities.get_average_of_max_values(x,top_k),axis= 1)
    avg_of_max_assignment_scores_each_reviewer=pd.DataFrame(avg_of_max_assignment_scores_each_reviewer)

    avg_of_max_assignment_scores_each_reviewer['rid']=avg_of_max_assignment_scores_each_reviewer.index
    avg_of_max_assignment_scores_each_reviewer.columns=['max_avg','rid']
    print('avg_of_max_assignment_scores_each_reviewer')
    print(avg_of_max_assignment_scores_each_reviewer)

    
    print('finding variance of top 5 max assignment scores for the each reviewer')
    var_of_max_assignment_scores_each_reviewer= scores_lda.apply(lambda x: utilities.get_var_of_max_values(x,top_k),axis=1)
    var_of_max_assignment_scores_each_reviewer=pd.DataFrame(var_of_max_assignment_scores_each_reviewer)
    
    print('var of max scores reviewers')
    var_of_max_assignment_scores_each_reviewer['rid']= var_of_max_assignment_scores_each_reviewer.index
    var_of_max_assignment_scores_each_reviewer.columns=['var_topk','rid']
    print(var_of_max_assignment_scores_each_reviewer.tail())

    return var_of_max_assignment_scores_each_reviewer, avg_of_max_assignment_scores_each_reviewer

var_of_max_assignment_scores_each_reviewer, avg_of_max_assignment_scores_each_reviewer= investigate_reviewers_return_var_avg_topk(scores_lda,top_k)


def investigate_papers_return_var_avg_topk(scores_lda, top_k):
    print('----now investigating for papers -------')
    scores_lda=scores_lda.T
    print('score lda is:')
    print(scores_lda)
    var_of_max_assign_each_paper=scores_lda.apply(lambda x: utilities.get_var_of_max_values(x,top_k),axis=1)
    var_of_max_assign_each_paper=pd.DataFrame(var_of_max_assign_each_paper)
    print('var of max scores papers ,taking top-',top_k)
    var_of_max_assign_each_paper['pid']=var_of_max_assign_each_paper.index
    var_of_max_assign_each_paper.columns=['var_topk','pid']
    print(var_of_max_assign_each_paper)


    avg_of_max_assignment_scores_each_paper=scores_lda.apply(lambda x: utilities.get_average_of_max_values(x,top_k),axis=1)
    avg_of_max_assignment_scores_each_paper=pd.DataFrame(avg_of_max_assignment_scores_each_paper)
    print('avg of max scores papers, taking top- ',top_k)
    avg_of_max_assignment_scores_each_paper['pid']=avg_of_max_assignment_scores_each_paper.index
    avg_of_max_assignment_scores_each_paper.columns=['avg_topk','pid']
    print(avg_of_max_assignment_scores_each_paper)

    return var_of_max_assign_each_paper, avg_of_max_assignment_scores_each_paper

var_of_max_assign_each_paper, avg_of_max_assignment_scores_each_paper= investigate_papers_return_var_avg_topk(scores_lda,top_k)

top_k_paper_stat= pd.merge(avg_of_max_assignment_scores_each_paper,var_of_max_assign_each_paper,how='left',on='pid')
print('for each paper, looking at assignment stats, taking top-',top_k)

# top_k_paper_stat['bucket_avg']=pd.qcut(top_k_paper_stat['avg_topk'].rank(method='first'), 2, labels=['low_max','high_max'], duplicates='drop')
# top_k_paper_stat['bucket_var']=pd.qcut(top_k_paper_stat['var_topk'].rank(method='first'), 2, labels=['low_var','high_var'], duplicates='drop')



#don't prepare cells. instead either look at bottom 20% quantile 
#or do a scatter plot
def cell_write_from_values(record):
    s=record['avg_topk']
    v=record['var_topk']

    if s> 0.2749:
        score='high_max'
    else:
        score='low_max'
    
    if v>0.00122:
        var='high_var'
    else:
        var='low_var'
    
    if score=='high_max' and var=='high_var':
        return 4
    elif score=='high_max' and var=='low_var':
        return 2
    elif score=='low_max' and var=='high_var':
        return 3
    else:
        return 1



def cell_write(record):
    score=record['bucket_avg']
    var=record['bucket_var']
    if score=='high_max' and var=='high_var':
        return 4
    elif score=='high_max' and var=='low_var':
        return 2
    elif score=='low_max' and var=='high_var':
        return 3
    else:
        return 1

'''
top_k_paper_stat['cell']= top_k_paper_stat.apply(lambda x: cell_write_from_values(x),axis=1)

print(top_k_paper_stat[:20])
top_k_paper_stat= top_k_paper_stat.sort_values('var_topk')




print('avg of avg max scores topk is ', top_k_paper_stat['avg_topk'].mean())
print('avg of var max scores topk is ', top_k_paper_stat['var_topk'].mean())
print('number of papers in each cell')
print(top_k_paper_stat.groupby('cell')['pid'].count())
'''

'''
print('---investigating what the top matching reviewers are working on ')
z= pd.merge(avg_of_max_assignment_scores_each_reviewer,reviewers_with_topics_of_interest,how='left',on='rid')
print(z)

def generic_topic_counter(array):
    count_generic=0
    generic_list=[12,13,7,22]
    for ar in array:
        if ar in generic_list:
            count_generic+=1
    return count_generic

def niche_topic_counter(array):
    niche_list=[4,11,15,24]
    count_niche=0
    for ar in array:
        if ar in niche_list:
            count_niche+=1
    return count_niche


z['generic_count']= z['matches'].apply(lambda x: generic_topic_counter(x))
z['niche_count']=z['matches'].apply(lambda x : niche_topic_counter(x))
z['check']=np.where(z['generic_count']>=z['niche_count'],1,0)
print('correlation_generic: ', z['generic_count'].corr(z['max_avg']))
print('correlation_niche : ' , z['niche_count'].corr(z['max_avg']))
print(z)

print('count of generic more than niche ',z['check'].sum())
print('ratio of total matches to those with generic topics ', z['check'].sum()/len(z.index))


print('---focus of reviewers----')
reviewers_topics= pd.read_csv('../reviewers_lda_topics.csv',header=None)
reviewers_topics['var']=reviewers_topics.var(axis=1)
print('reviewers_topics lda ', reviewers_topics)
reviewer_focus= reviewers_topics[['var']]

reviewer_focus['focus']=pd.qcut(reviewer_focus['var'].rank(method='first'), 2, labels=['low','high'], duplicates='drop')
reviewer_focus['rid']=reviewer_focus.index
print(reviewer_focus)

print('-----investigating matches against focus----')
reviewer_focus=reviewer_focus.loc[reviewer_focus['focus']=='high']
f= pd.merge(reviewer_focus,reviewers_with_topics_of_interest,how='left',on='rid')
# print(f)
f['generic_count']= f['matches'].apply(lambda x: generic_topic_counter(x))
f['niche_count']=f['matches'].apply(lambda x : niche_topic_counter(x))
f['check']=np.where(f['generic_count']>f['niche_count'],1,0)
# print('correlation_generic: ', f['generic_count'].corr(f['max_avg']))
# print('correlation_niche : ' , z['niche_count'].corr(z['max_avg']))
print(f)

print('count of generic more than niche ',f['check'].sum())
print('ratio of total matches to those with generic topics ', f['check'].sum()/len(f.index))
'''