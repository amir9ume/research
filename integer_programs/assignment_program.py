import cvxpy as cp
import numpy as np 
import pandas as pd
import sys

np.random.seed(1)

'''
Assuming papers are rows
Reviewers are in columns

'''

def transformed_sigmoid(x):
    alpha=20
    y= 1/(1+np.exp(alpha*(-x+0.4)))
    return y


df= pd.read_csv('../../scores_lda.csv', header=None).T
s= df.values
S=s.copy()

#transforms=['none','sigmoid']
transforms=['sigmoid']
transforms=['none']
for trans in transforms:
    if trans=='sigmoid':
        S= s.ravel()
        S= transformed_sigmoid(S).reshape(s.shape)
    
    #toy problem to validate idea
    #S= np.array([[0.3, 0.9],[0.1,0.2]])
    #print(S)

    x=cp.Variable(S.shape,boolean=True)
    t=cp.Variable(1)
    assignment= cp.multiply(x,S)


    paper_constraint= sys.argv[1]
    reviewer_lower_load= sys.argv[2]
    reviewer_upper_load= sys.argv[3]

#    stages=['global','maxmin']
    
    stages=['maxmin']
    #stages=['global']
    for st in stages:

        
        if st=='maxmin':
            # constraints= [ t <= cp.sum(assignment,axis=1), cp.sum(x, axis=1)==paper_constraint, reviewer_lower_load<= cp.sum(x,axis=0),
            # cp.sum(x,axis=0)<= reviewer_upper_load]
            # obj=cp.Maximize(t)

            """
            Working on the easiest constraints for now. Only with paper constraint, and a loose reviewer upper load
            """

            constraints= [ t <= cp.sum(assignment,axis=1), cp.sum(x, axis=1)==paper_constraint, 
            cp.sum(x,axis=0)<=reviewer_upper_load ]
            obj=cp.Maximize(t)

        elif st=='global':
            
            # constraints= [  cp.sum(x, axis=1)==paper_constraint, reviewer_lower_load<=cp.sum(x,axis=0),
            # cp.sum(x,axis=0)<= reviewer_upper_load]
            # obj=cp.Maximize(cp.sum(assignment))

            constraints= [  cp.sum(x, axis=1)==paper_constraint, 
            cp.sum(x,axis=0)<= reviewer_upper_load]
            obj=cp.Maximize(cp.sum(assignment))



        prob= cp.Problem(obj, constraints)
        
        try:
            result= prob.solve(solver=cp.MOSEK)
            #result= prob.solve(solver=cp.CPLEX)
        
            print('----------------------------------------')
            print('objective used ',st)
            print('transform used ', trans)
            print('paper constraint ', paper_constraint)
            print('reviewer lower load ', reviewer_lower_load)
            print('reviewer upper load',reviewer_upper_load)
            print('status ',prob.status)

            if prob.status=='optimal':
                #print(x)
                # if st=='global':
                #     x= (np.round(np.abs(x)))
                xx= pd.DataFrame(x.value)
                print('global cum sum value')
                print(cp.sum(assignment).value)

                xx.to_csv('./results/'+'results_'+st+'_'+str(paper_constraint)+'_'+'0'+'_'+'_reviewer_low'+str(reviewer_lower_load)
                +str(reviewer_upper_load),index=False)
                print('sum each row')
                print(cp.sum(assignment, axis=1).value)
                print('t value, which is the lower bound achieved ')
                if t.value != None:
                    print(t.value)
                else:
                    print(cp.min(cp.sum(assignment,axis=1)).value)
                

            print('----------------------------------------')
        except:
            print('time limit exceeded')