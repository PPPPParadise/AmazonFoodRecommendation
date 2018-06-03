import pandas as pd
import numpy as np
import os

data = pd.read_csv('stack_test.csv')
mt = data.iloc[:,1:]
# mt['sum'] = mt.mean(axis=1)
multi = np.ones(mt.shape)
multi[:,0] = 4
multi[:,-1] = 3
mt_weighted = (mt*2-1)*multi

def findmulti(w1,w2,w3,w4,w5,w6,w7):
    multi = np.ones(mt.shape)
    multi[:,0] = w1
    multi[:,1] = w2
    multi[:,2] = w3
    multi[:,3] = w4
    multi[:,4] = w5
    multi[:,5] = w6
    multi[:,6] = w7
    mt_weighted = (mt*2-1)*multi
    a  = mt_weighted.sum(axis=1)
    decision = []
    for item in a:
        if item>0:
            decision.append(1)
        if item<=0:
            decision.append(0)
    out = pd.DataFrame([])
    out['PassengerId'] = id
    out['Survived'] = decision
    out.to_csv('result.csv',index=False)
# We set the first Randomforest, xgboot, decition tree to zero because they use the same methodoloy with the last one, but not better
# 2;1;1;4 is based on standard-scared cross-validation with some tiny manully changes(in order not to overfit)
# the cross-validtion processing is unfortunetely missing in kaggle's notebook kernel, just retain the results
findmulti(0,0,2,0,1,1,4)