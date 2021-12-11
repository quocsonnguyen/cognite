# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:28:19 2019

@author: Vu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt
from bayes_opt.test_functions import psychology_tom_blackbox
from bayes_opt.visualization import visualization_psy
from tqdm import tqdm

#olddata=pd.read_csv('backup/data_original_no46.csv')
olddata=pd.read_csv('data_Nov.csv')

#print(olddata)
data=olddata.values

plt.plot(data[:,2])

fig=plt.figure()
plt.hist(data[:,2],30)
plt.ylabel("Histogram",fontsize=16)
plt.xlabel("Ability",fontsize=16)
fig.savefig("Hist_Ability.pdf")


from numpy import loadtxt
mybaseline = loadtxt("baseline_input.csv", comments="#", delimiter=",", unpack=False)
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

init_X=data[:,:2]
init_Y=data[:,3]
#context_obs=data[:,2]



acq_type={}
#acq_type['name']='ucb' # expected regret minimization
acq_type['name']='thompson' # expected regret minimization
acq_type['surrogate']='gp' # recommended to use tgp for ERM
acq_type['dim']=3

acq_params={}
acq_params['acq_func']=acq_type
acq_params['optimize_gp']='loo'


# ======================================================================
# Exporting plot condition on ability

# Init Bayes Opt 3D

myfunction=psychology_tom_blackbox.PsychologyTom_3D()
func_params={}
func_params['function']=myfunction

gp_params = {'lengthscale':0.025,'noise_delta':1e-6} # the lengthscaled parameter will be optimized


bo3d=BayesOpt(gp_params,func_params,acq_params,verbose=1)
bo3d.init_with_data(data[:,:3],data[:,3])
newinput=bo3d.maximize_condition(mybaseline)


print("=================================================================")
print("Suggested input: Current={:.3f} Frequency={:.1f} given Baseline={:.3f}".format(bo3d.X_original[-1][0],
      bo3d.X_original[-1][1],bo3d.X_original[-1][2]))


#newinput=[bo3d.X_original[-1][0],bo3d.X_original[-1][1],bo3d.X_original[-1][2]]

# check if newinput is existed, perform random exploration
if np.any(np.abs((bo3d.X_original[:-1] - newinput)).sum(axis=1) <=(bo3d.dim*1e-9)):
    print("rand")
    newinput = np.random.uniform(bo3d.bounds[:, 0],
                                      bo3d.bounds[:, 1],
                                      size=bo3d.bounds.shape[0])
    
newinput[2]=mybaseline
#print(newinput)
#np.savetxt("newinput.csv", np.asarray(newinput), delimiter=",")

T=200
mybaselines=np.linspace(0,1,T)

#myGPmean_inout=[0]*len(mybaselines)
all_data=ys = np.array([], dtype=np.float64).reshape(0,5)

max_mean=[0]*T
ave_mean=[0]*T
min_mean=[0]*T

ave_std=[0]*T
for ii,b in tqdm(enumerate(mybaselines)):

    b_org=b*(bo3d.bounds[-1,1]-bo3d.bounds[-1,0])+bo3d.bounds[-1,0]
    GPmean_input,GPmean_output,GPvar_output=visualization_psy.plot_bo_2d_condition(bo3d,b_org,flagOutFile=False)
    GPmean_output=np.reshape(GPmean_output,(-1,1))
    GPvar_output=np.reshape(GPvar_output,(-1,1))

    #baseline=[b]*len(GPmean_output)
    #baseline=np.asarray(baseline)
    temp=np.hstack((GPmean_input,GPmean_output,GPvar_output))
    #myGPmean_inout[ii]=temp
    all_data=np.vstack((all_data,temp))


    ave_mean[ii]=np.mean(GPmean_output)
    max_mean[ii]=np.max(GPmean_output)
    min_mean[ii]=np.min(GPmean_output)

    ave_std[ii]=np.std(GPvar_output)
    
strPath="GPmean_var_4d_Nov.csv"
pd.DataFrame(np.asarray(all_data)).to_csv(strPath,index=False,header=False)



fig=plt.figure()
plt.errorbar(np.arange(T),min_mean,ave_std)
plt.xlabel("Ability",fontsize=16)
plt.ylabel("Predicted Performance",fontsize=16)

mytick=np.linspace(0,T,5)
myxlabel=np.linspace(bo3d.bounds[2,0],bo3d.bounds[2,1],5)
plt.xticks(mytick,myxlabel)
plt.title("Predicted response to the cBO parameters",fontsize=18)
fig.savefig("Performance_cBO.pdf",box_inches="tight")

strPath="Performance_cBO_Nov.csv"

mybaseline_ori=np.linspace(bo3d.bounds[2,0],bo3d.bounds[2,1],T)
mydata=np.vstack((mybaseline_ori,np.asarray(ave_mean),np.asarray(ave_std))).T

print(mydata.shape)
pd.DataFrame(np.asarray(mydata)).to_csv(strPath,index=False,header=False)



# perform Thompson Sampling

"""
mybaselines=np.linspace(0,1,100)
all_data=ys = np.array([], dtype=np.float64).reshape(0,5)
for ii,b in tqdm(enumerate(mybaselines)):
    b_org=b*(bo3d.bounds[-1,1]-bo3d.bounds[-1,0])+bo3d.bounds[-1,0]

    GPmean_input,GPmean_output,GPvar_output=visualization_psy.plot_bo_2d_condition_TS(bo3d,b_org,
                                                                          flagOutFile=True)
    GPmean_output=np.reshape(GPmean_output,(-1,1))
    GPvar_output=np.reshape(GPvar_output,(-1,1))

    #baseline=[b]*len(GPmean_output)
    #baseline=np.asarray(baseline)
    temp=np.hstack((GPmean_input,GPmean_output,GPvar_output))
    #myGPmean_inout[ii]=temp
    #all_data=np.vstack((all_data,temp))
"""

#strPath="GPmean_var_4d_TS.csv"
#pd.DataFrame(np.asarray(all_data)).to_csv(strPath,index=False,header=False)