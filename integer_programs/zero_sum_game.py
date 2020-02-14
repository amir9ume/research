import cvxpy as cp
import numpy as np 

'''
simple code implementation for example given on

http://www.4er.org/CourseNotes/Book%20A/A-III.pdf
Page A-100
Optimization Models- Robert Fourer
'''
np.random.seed(1)
A= np.array([[4,-1,-1],[-2,4,-2],[-3,-3,4]])
print(A)

x= cp.Variable(3)
z= cp.Variable(1)

p_sum= cp.sum(x,axis=0)

k= A*x
constraints=[p_sum==1, cp.min(x)>=0, z<= k]
obj= cp.Maximize(z)

prob= cp.Problem(obj, constraints)
result= prob.solve()
print(prob.status)
if prob.status=='optimal':
    print(x.value)
    print('optimized z value')
    print(z.value)


print('solution in book')
print(23/107,' ',37/107,' ',47/107)



