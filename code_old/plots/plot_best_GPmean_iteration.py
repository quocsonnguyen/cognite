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

from tqdm import tqdm

olddata=pd.read_csv('..//data.csv')
#print(olddata)
data=olddata.values

from numpy import loadtxt
mybaseline = loadtxt("../baseline_input.csv", comments="#", delimiter=",", unpack=False)
mybaseline=np.asscalar(mybaseline)


init_Y=data[:,3]
#context_obs=data[:,2]


myfunction=psychology_tom_blackbox.PsychologyTom_3D()



gp_params = {'lengthscale':0.004,'noise_delta':1e-4} # the lengthscaled parameter will be optimized

acq_type={}
acq_type['name']='ucb' # expected regret minimization
acq_type['surrogate']='gp' # recommended to use tgp for ERM
acq_type['dim']=2


acq_params={}
acq_params['acq_func']=acq_type
acq_params['optimize_gp']='loo'


func_params={}
func_params['function']=myfunction

maxGPmean=[]
sig_at_maxGPmean=[]
sig_at_chosenpoint=[]

for ii in tqdm(range(1,147)):
    
    bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
      
    # initialize BO using 3*dim number of observation
    bo.init_with_data(data[:ii,:3],init_Y[:ii])
    
    #visualization_psy.show_optimization_progress(bo)
    
    new_x=bo.maximize_condition(data[ii,2])
    print(new_x,bo.maxGPmean)
    maxGPmean+=[bo.maxGPmean[-1]]
    sig_at_maxGPmean+=[bo.sig_at_max_GPmean]
    sig_at_chosenpoint+=[bo.sig_at_chosenpoint]


fig=plt.figure()
plt.plot(maxGPmean,linewidth=2)
plt.xlim([-1,150])
plt.xlabel('pBO Iteration',fontsize=14)
plt.ylabel("Max of GP Mean",fontsize=14)
plt.title("Best Predicted Performance",fontsize=16)
fig.savefig("best_pred_performance.pdf",bbox_inches="tight")

strPath="best_pred_performance.csv"
pd.DataFrame(np.asarray(maxGPmean)).to_csv(strPath,index=False,header=False)




fig=plt.figure()
plt.plot(sig_at_maxGPmean,linewidth=2)
plt.xlim([-1,150])
plt.xlabel('pBO Iteration',fontsize=14)
plt.ylabel("Uncertainty",fontsize=14)
plt.title("Uncertainty at Max GP Mean Prediction",fontsize=16)
fig.savefig("var_at_maxGPmean.pdf",bbox_inches="tight")


strPath="var_at_maxGPmean.csv"
pd.DataFrame(np.asarray(sig_at_maxGPmean)).to_csv(strPath,index=False,header=False)




fig=plt.figure()
plt.plot(sig_at_chosenpoint,linewidth=2)
plt.xlim([-1,150])
plt.xlabel('pBO Iteration',fontsize=14)
plt.ylabel("Uncertainty",fontsize=14)
plt.title("Uncertainty at the Chosen Point",fontsize=16)
fig.savefig("var_at_xt.pdf",bbox_inches="tight")


strPath="var_at_xt.csv"
pd.DataFrame(np.asarray(sig_at_chosenpoint)).to_csv(strPath,index=False,header=False)
