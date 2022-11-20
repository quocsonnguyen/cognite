# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:57:42 2019

@author: Vu
"""

import sys

sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt
from bayes_opt.test_functions import psychology_tom_blackbox
from bayes_opt.visualization import visualization_psy
from sklearn.linear_model import LinearRegression

olddata=pd.read_csv('data_original.csv', header=None)
#print(olddata)
data=olddata.values

fig=plt.figure()
plt.plot(data[:,2])



#data = np.delete(data, (66,67), axis=0)




# Linear Regression

#data=np.reshape(data[:,-1],(-1,3))
#data=np.max(data[:,:],axis=1)

#data=data[:140,:]

#idxRow=np.where(data[:,-1]>0.3)[0]
#idxRow1=np.where(data[0:60,-1]<3)[0]
#idxRow2=60+np.where(data[60:,-1]>0.3)[0]
#idxRow=np.concatenate([idxRow1,idxRow2])
#print(idxRow)
#data=data[idxRow,:]

X=np.linspace(0,data.shape[0],data.shape[0])
y=data[:,-1]

X=np.reshape(X,(-1,1))
y=np.reshape(y,(-1,1))
reg = LinearRegression().fit(X, y)

y_pred=reg.predict(X)


fig=plt.figure()
plt.scatter(X,y)
plt.plot(X,y_pred,'r')
fig.savefig("scatter_plot.pdf",bbox_inches="tight")





perf=np.asarray(data[3*20:,-1])


baseline=np.asarray(data[:,2])
bl_inv=baseline.reshape((-1,3))

perf_inv=perf.reshape((-1,3))

print("pBO",np.mean(perf_inv))
print("pBO",np.std(perf_inv))

ave_perf_inv=np.max(perf_inv,axis=0)
#ave_perf_inv=np.mean(ave_perf_inv,axis=0)

std_perf_inv=np.std(perf_inv,axis=0)

fig=plt.figure()
plt.errorbar([0,1,2],ave_perf_inv,std_perf_inv,label="pBO Stage")
#plt.xlabel("Trials",fontsize=14)
#plt.ylabel("Ave Across Participants",fontsize=14)
#plt.title("pBO Stage: Participant 20-50",fontsize=16)
#fig.savefig("pBO_stage.pdf",bbox_inches="tight")




perf=np.asarray(data[3*0:3*20,-1])
perf_inv=perf.reshape((-1,3))

ave_perf_inv=np.mean(perf_inv,axis=0)
ave_perf_inv=np.max(perf_inv,axis=0)

std_perf_inv=np.std(perf_inv,axis=0)

#fig=plt.figure()
plt.errorbar([0,1,2],ave_perf_inv,std_perf_inv,label='Random Stage')
plt.xlabel("Trials",fontsize=14)
plt.ylabel("Ave Across Participants",fontsize=14)
plt.title("Random Stage: Participant 0-20",fontsize=16)
plt.legend()
fig.savefig("Random_pBO_stage.pdf",bbox_inches="tight")



print("Rand",np.mean(perf_inv))
print("Rand",np.std(perf_inv))

fig=plt.figure()
plt.bar([0,1],[1.255,1.378],color=['blue','red'])
plt.ylim([1.05,1.45])
plt.xticks([0,1], ['Random Phase 0-20','pBO Phase 20-50'],rotation=40,fontsize=14)
plt.ylabel("Average Performance",fontsize=14)
fig.savefig("bar_plot.pdf",bbox_inches="tight")

