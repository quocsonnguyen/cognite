# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:28:19 2019

@author: Vu
"""

from lib2to3.pgen2.pgen import DFAState
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt
from bayes_opt.sequentialBO.bayesian_optimization_uniqueGP import BayesOpt_UniqueLoc

#from bayes_opt.test_functions import psychology_tom_blackbox
from bayes_opt.test_functions import cognite_blackbox
from bayes_opt.visualization import visualization_psy as viz

def visualize(data_path, from_index=0, to_index=-1, skip=1):
    df=pd.read_csv(data_path, header=None, names=range(6))
    #data=df.values[from_index:to_index:skip] #===============================
    data=df.values[1:,:] #===============================
    data = data[from_index:to_index:skip,:]
    
    init_X=data[:,:3].astype(float)
    init_Y=data[:,4].astype(float)

    acq_type={}
    acq_type['name']='ucb'
    acq_type['surrogate']='pgp_unique'
    acq_type['dim']=[1,2] # 1 is #dim of X, 2 is #dim of T

    #myfunction=psychology_tom_blackbox.PsychologyTom_CA() # current and ability (personalised score)
    myfunction = cognite_blackbox.CogniteFunc_3D()

    func_params={}
    func_params['function']=myfunction

    gp_params = {'lengthscale':0.01,'noise_delta':2e-2} # the lengthscaled parameter will be optimized

    acq_params={}
    acq_params['acq_func']=acq_type
    #acq_params['optimize_gp']='loo'
    acq_params['optimize_gp']='maximize'

    
    bo=BayesOpt_UniqueLoc(gp_params,func_params,acq_params,verbose=1)
    
    # initialize BO using 3*dim number of observation
    bo.init_with_data(init_X[:,0], init_X[:,[1,2]] ,init_Y)

    # optimize GP hyperparameter
    temp=bo.gp.optimize_lengthscale(bo.gp.lengthscale_x_old, bo.gp.lengthscale_t_old,bo.gp.noise_delta)
    
    #GPmean_input,GPmean_output=[],[]
    # GPmean_input,GPmean_output=viz.plot_bo_2d_withGPmeans(bo,\
    #                                     myxlabel="Current Intensity", myylabel="Personalised Score")
    
    # plot Intensity and behavioural =========================================
    myfunction = cognite_blackbox.Cognite_2D_Intensity_Behavioural()
    func_params={}
    func_params['function']=myfunction
    
    acq_type['dim']=[1,1] # 1 is #dim of X, 2 is #dim of T
    acq_params['acq_func']=acq_type

    
    bo=BayesOpt_UniqueLoc(gp_params,func_params,acq_params,verbose=1)
    bo.init_with_data(init_X[:,0], init_X[:,2],init_Y)
    bo.gp.optimize_lengthscale(bo.gp.lengthscale_x_old, bo.gp.lengthscale_t_old,bo.gp.noise_delta)
    viz.plot_bo_2d_withGPmeans_Sigma(bo,myxlabel="Intensity", myylabel="Behavioral Score", saveflag="int_beha")
    viz.plot_2d_Acq_by_Personalisedscore(bo,myxlabel="Intensity", myylabel="Behavioral Score",saveflag="int_beha")

    # plot Intensity and spherical =========================================
    myfunction = cognite_blackbox.Cognite_2D_Intensity_Spherical()
    func_params={}
    func_params['function']=myfunction
    
    acq_type['dim']=[1,1] # 1 is #dim of X, 2 is #dim of T
    acq_params['acq_func']=acq_type
    
    bo=BayesOpt_UniqueLoc(gp_params,func_params,acq_params,verbose=1)
    bo.init_with_data(init_X[:,0],init_X[:,[1]],init_Y)
    bo.gp.optimize_lengthscale(bo.gp.lengthscale_x_old, bo.gp.lengthscale_t_old,bo.gp.noise_delta)
    viz.plot_bo_2d_withGPmeans_Sigma(bo,myxlabel="Intensity", myylabel="Spherical Head", saveflag="int_sphe")
    viz.plot_2d_Acq_by_Personalisedscore(bo,myxlabel="Intensity", myylabel="Spherical Head",saveflag="int_sphe")

    # plot spherical behavioral=========================================
    myfunction = cognite_blackbox.Cognite_2D_Spherical_Behavioural()
    func_params={}
    func_params['function']=myfunction
    
    acq_type['dim']=[1,1] # 1 is #dim of X, 2 is #dim of T
    acq_params['acq_func']=acq_type
    
    bo=BayesOpt_UniqueLoc(gp_params,func_params,acq_params,verbose=1)
    bo.init_with_data(init_X[:,[1]], init_X[:,[2]],init_Y)
    bo.gp.optimize_lengthscale(bo.gp.lengthscale_x_old, bo.gp.lengthscale_t_old,bo.gp.noise_delta)
    viz.plot_bo_2d_withGPmeans_Sigma(bo,myxlabel="Spherical Head", \
                                     myylabel="Behavioral Score", saveflag="sphe_beha")
    viz.plot_2d_Acq_by_Personalisedscore(bo,myxlabel="Spherical Head", \
                                         myylabel="Behavioral Score",saveflag="sphe_beha")



    return {
        'currentIntensity' : temp[0],
        'personalisedScore' : temp[1],
        'noise' : bo.gp.noise_delta
    }

#print(output)







