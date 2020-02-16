import cvxpy as cp
import numpy as np 
import pandas as pd
import sys

np.random.seed(1)

'''
Assuming papers are rows
Reviewers are in columns

'''

df= pd.read_csv('../../scores_lda.csv', header=None).T
s= df.values
S=s.copy()

transforms=['none','exp']
for trans in transforms:
    if trans=='exp':
        S= s.ravel()
        S= np.exp(S).reshape(s.shape)
    
    #toy problem to validate idea
    #S= np.array([[0.3, 0.9],[0.1,0.2]])
    #print(S)

    x=cp.Variable(S.shape,boolean=True)
    t=cp.Variable(1)
    assignment= cp.multiply(x,S)


    paper_constraint= sys.argv[1]
    reviewer_upper_load= sys.argv[2]

    stages=['global','maxmin']

    for st in stages:

        
        if st=='maxmin':
            constraints= [ t <= cp.sum(assignment,axis=1), cp.sum(x, axis=1)==paper_constraint, 
            cp.sum(x,axis=0)<= reviewer_upper_load]
            obj=cp.Maximize(t)

        elif st=='global':
            
            constraints= [  cp.sum(x, axis=1)==paper_constraint, 
            cp.sum(x,axis=0)<= reviewer_upper_load]
            obj=cp.Maximize(cp.sum(assignment))


        prob= cp.Problem(obj, constraints)
        #result= prob.solve(solver=cp.MOSEK)
        try:
            result= prob.solve(solver=cp.CPLEX, cplex_params={"timelimit": 2})
        
            print('----------------------------------------')
            print('objective used ',st)
            print('transform used ', trans)
            print('paper constraint ', paper_constraint)
            print('reviewer upper load',reviewer_upper_load)
            print('status ',prob.status)

            if prob.status=='optimal':
                #print(x)
                # if st=='global':
                #     x= (np.round(np.abs(x)))
                xx= pd.DataFrame(x.value)
                print('global cum sum value')
                print(cp.sum(assignment).value)

                xx.to_csv('results_'+st+'_'+str(paper_constraint)+'_'+'0'+'_'+str(reviewer_upper_load),index=False)
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