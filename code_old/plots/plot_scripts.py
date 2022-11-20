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
olddata=pd.read_csv('..//data.csv')
#print(olddata)
data=olddata.values

from numpy import loadtxt
mybaseline = loadtxt("../baseline_input.csv", comments="#", delimiter=",", unpack=False)
mybaseline=np.asscalar(mybaseline)

#print("=================================================================")
#for ii in range(data.shape[1]):
#    print("min={}, mean={}, max={}".format(np.min(data[:,ii]),np.mean(data[:,ii]),np.max(data[:,ii])))
    
# 3D Plot
#fig = plt.figure(figsize=(8,5))
#ax3D = fig.add_subplot(111, projection='3d')
#p3d = ax3D.scatter(data[:,0], data[:,1], data[:,2], s=30, c=data[:,3], marker='o')  

#plt.xlabel("Current")
#plt.ylabel("Frequency")
#plt.zlabel("Frequency")
#plt.colorbar(p3d)
#fig.savefig("Data_Plot.pdf")

init_Y=data[:,3]
#context_obs=data[:,2]





gp_params = {'lengthscale':0.004,'noise_delta':1e-6} # the lengthscaled parameter will be optimized

acq_type={}
acq_type['name']='ucb' # expected regret minimization
acq_type['surrogate']='gp' # recommended to use tgp for ERM
acq_type['dim']=2



acq_params={}
acq_params['acq_func']=acq_type
acq_params['optimize_gp']='loo'


# Current Frequency
myfunction=psychology_tom_blackbox.PsychologyTom_CF()

func_params={}
func_params['function']=myfunction


    
bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
  
# initialize BO using 3*dim number of observation
#bo.init_with_data(data[:,:2],init_Y)
bo.init_with_data(data[:60,:2],init_Y[:60])


#visualization_psy.show_optimization_progress(bo)

fig=plt.figure()
plt.plot(init_Y, linewidth=2)
plt.xlim([0,150])
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Performance',fontsize=14)
plt.title("Response at Iteration",fontsize=16)
fig.savefig("Performance_iteration.pdf",box_inches="tight")


strPath="Performance_iteration.csv"
pd.DataFrame(np.asarray(init_Y)).to_csv(strPath,index=False,header=False)

#GPmean_input,GPmean_output=[],[]
GPmean_input,GPmean_output=visualization_psy.plot_bo_2d_withGPmeans(bo, myxlabel="Current",myylabel="Frequency")






myfunction=psychology_tom_blackbox.PsychologyTom_CF()

func_params={}
func_params['function']=myfunction
bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
  
# initialize BO using 3*dim number of observation
#bo.init_with_data(data[:,:2],init_Y)
bo.init_with_data(data[60:,:2],init_Y[60:])


#visualization_psy.show_optimization_progress(bo)

fig=plt.figure()
plt.plot(init_Y, linewidth=2)
plt.xlim([0,150])
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Performance',fontsize=14)
plt.title("Response at Iteration",fontsize=16)
fig.savefig("Performance_iteration.pdf",box_inches="tight")


strPath="Performance_iteration.csv"
pd.DataFrame(np.asarray(init_Y)).to_csv(strPath,index=False,header=False)

#GPmean_input,GPmean_output=[],[]
GPmean_input,GPmean_output=visualization_psy.plot_bo_2d_withGPmeans(bo, myxlabel="Current",myylabel="Frequency")







# Current Frequency
myfunction=psychology_tom_blackbox.PsychologyTom_CA()

func_params={}
func_params['function']=myfunction

    
bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
bo.init_with_data(data[:,[0,2]],init_Y)

#GPmean_input,GPmean_output=[],[]
GPmean_input,GPmean_output=visualization_psy.plot_bo_2d_withGPmeans(bo, myxlabel="Current",myylabel="Ability")




# Frequency Ability
myfunction=psychology_tom_blackbox.PsychologyTom_FA()

func_params={}
func_params['function']=myfunction

    
bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
bo.init_with_data(data[:,[1,2]],init_Y)

#GPmean_input,GPmean_output=[],[]
GPmean_input,GPmean_output=visualization_psy.plot_bo_2d_withGPmeans(bo, myxlabel="Frequency",myylabel="Ability")