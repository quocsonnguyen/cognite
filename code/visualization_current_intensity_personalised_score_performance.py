# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:28:19 2019

@author: Vu
"""

from lib2to3.pgen2.pgen import DFAState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt
from bayes_opt.test_functions import psychology_tom_blackbox
from bayes_opt.visualization import visualization_psy as viz

def visualize(data_path, from_index=0, to_index=-1, skip=1):
    df=pd.read_csv(data_path)
    data=df.values[from_index:to_index:skip] #===============================

    init_X=data[:,[0,2]].astype(float)
    init_Y=data[:,3].astype(float)

    acq_type={}
    acq_type['name']='ucb'
    acq_type['surrogate']='pgp'
    acq_type['dim']=2

    myfunction=psychology_tom_blackbox.PsychologyTom_CA() # current and ability (personalised score)

    func_params={}
    func_params['function']=myfunction

    gp_params = {'lengthscale':0.01,'noise_delta':5e-1} # the lengthscaled parameter will be optimized

    acq_params={}
    acq_params['acq_func']=acq_type
    acq_params['optimize_gp']='loo'
    acq_params['optimize_gp']='maximize'

    
    bo=BayesOpt(gp_params,func_params,acq_params,verbose=1)
    
    # initialize BO using 3*dim number of observation
    bo.init_with_data(init_X,init_Y)

    # optimize GP hyperparameter
    temp=bo.gp.optimize_lengthscale(bo.gp.lengthscale_x_old, bo.gp.lengthscale_t_old,bo.gp.noise_delta)
    
    #GPmean_input,GPmean_output=[],[]
    # GPmean_input,GPmean_output=viz.plot_bo_2d_withGPmeans(bo,\
    #                                     myxlabel="Current Intensity", myylabel="Personalised Score")
    viz.plot_bo_2d_withGPmeans_Sigma(bo,myxlabel="Current Intensity", myylabel="Personalised Score")
    viz.plot_2d_Acq_by_Personalisedscore(bo,myxlabel="Current Intensity", myylabel="Personalised Score")

    return {
        'currentIntensity' : temp[0],
        'personalisedScore' : temp[1],
        'noise' : bo.gp.noise_delta
    }








