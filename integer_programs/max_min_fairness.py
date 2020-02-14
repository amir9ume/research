import cvxpy as cp
import numpy as np 

np.random.seed(1)

#toy problem to validate idea
S= np.array([[0.3, 0.9],[0.1,0.2]])
print(S)

x=cp.Variable(S.shape,boolean=True)
t=cp.Variable(1)
assignment= cp.multiply(x,S)

constraints= [ cp.sum(x, axis=0)<=2, t <= cp.sum(assignment,axis=1),cp.sum(x)<=2]

obj=cp.Maximize(t)
prob= cp.Problem(obj, constraints)
result= prob.solve()
print('status ',prob.status)
print(x.value)
print('sum each row')
print(cp.sum(assignment, axis=1).value)
print('t value')
print(t.value)
"""
printed values

[[0.3 0.9]
 [0.1 0.2]]
status  optimal
[[-1.04596911e-11  1.00000000e+00]
 [-1.87566149e-11  1.00000000e+00]]
sum each row
[0.9 0.2]
t value
[0.2]

"""