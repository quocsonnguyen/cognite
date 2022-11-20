# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:28:19 2019

@author: Vu
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt
from bayes_opt.sequentialBO.bayesian_optimization_uniqueGP import BayesOpt_UniqueLoc

#from bayes_opt.test_functions import psychology_tom_blackbox
from bayes_opt.test_functions import cognite_blackbox

#from bayes_opt.visualization import visualization_psy
from numpy import loadtxt

def get_frequency(data_path, personalised_score):

    olddata=pd.read_csv(data_path, names=range(6))
    data=olddata.values

    # personalised_score = loadtxt("personalised_score.csv", comments="#", delimiter=",", unpack=False)
    # personalised_score=np.asscalar(personalised_score)

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

    init_X=data[1:,:3].astype(float)
    init_Y=data[1:,4].astype(float)
    #context_obs=data[:,2]


    acq_type={}
    acq_type['name']='ucb'
    acq_type['surrogate']='pgp_unique'
    acq_type['dim']=[1,2]

    #myfunction=psychology_tom_blackbox.PsychologyTom_CF()

    # func_params={}
    # func_params['function']=myfunction

    # gp_params = {'lengthscale':0.004,'noise_delta':1e-6} # the lengthscaled parameter will be optimized


    acq_params={}
    acq_params['acq_func']=acq_type
    acq_params['optimize_gp']='maximize'

        
    #bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
    #bo=BayesOpt_UniqueLoc(gp_params,func_params,acq_params,verbose=1)

    # initialize BO using 3*dim number of observation
    #bo.init_with_data(init_X,init_Y)

    #GPmean_input,GPmean_output=[],[]
    # GPmean_input,GPmean_output=visualization_psy.plot_bo_2d_withGPmeans(bo)
    #visualization_psy.plot_bo_2d_withGPmeans_Sigma(bo)
    #visualization_psy.plot_bo_2d(bo)



    # GPmean_output=np.reshape(GPmean_output,(-1,1))
    # temp=np.hstack((GPmean_input,GPmean_output))
    # pd.DataFrame(np.asarray(temp)).to_csv("temp/GPmean_noability.csv",index=False,header=False)


    #bo.maximize()


    # ======================================================================
    # Exporting plot condition on ability

    # Init Bayes Opt 2D

    #myfunction=psychology_tom_blackbox.PsychologyTom_3D()
    myfunction = cognite_blackbox.CogniteFunc_3D()

    func_params={}
    func_params['function']=myfunction

    gp_params = {'lengthscale':0.02,'noise_delta':2e-2} # the lengthscaled parameter will be optimized


    #bo3d=BayesOpt(gp_params,func_params,acq_params,verbose=1)
    bo2d = BayesOpt_UniqueLoc(gp_params,func_params,acq_params,verbose=1)
    #bo2d.init_with_data(data[:,:3].astype(float),data[:,3].astype(float))
    bo2d.init_with_data(init_X[:,0].astype(float),init_X[:,[1,2]].astype(float),init_Y.astype(float))
    newinput=bo2d.maximize_given_2d_context(personalised_score)

    #print("=================================================================")
    #print("Suggested input: Current={:.3f} Frequency={:.1f} given Baseline={:.3f}".format(bo3d.X_original[-1][0],
    #      bo3d.X_original[-1][1],bo3d.X_original[-1][2]))


    #newinput=[bo3d.X_original[-1][0],bo3d.X_original[-1][1],bo3d.X_original[-1][2]]

    # check if newinput is existed, perform random exploration
    if np.any(np.abs((bo2d.X_original[:-1] - newinput)).sum(axis=1) <=(bo2d.dim*1e-6)):
        #print("rand")
        newinput = np.random.uniform(bo2d.bounds[:, 0],
                                        bo2d.bounds[:, 1],
                                        size=bo2d.bounds.shape[0])
      
    return newinput[0]


data_path = "../data/GLOBAL.csv"
temp = get_frequency(data_path, personalised_score=0.62) # old
temp = get_frequency(data_path, personalised_score=[60, 0.62]) # new
print(temp)


    # newinput[2]=personalised_score
    #print(newinput)
    # np.savetxt("suggested_frequency.csv", np.asarray(newinput), delimiter=",")




