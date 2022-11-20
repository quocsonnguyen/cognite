import sys
sys.path.insert(0,'..')
#sys.path.insert(0,'../..')


from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt
from bayes_opt.sequentialBO.bayesian_optimization_ignoreconditional import BayesOpt_NoCond
from bayes_opt.sequentialBO.bayesian_optimization_conditional import BayesOpt_Cond
import numpy as np
from bayes_opt import auxiliary_functions

from bayes_opt.test_functions import functions
import warnings

import sys

from bayes_opt.utility import export_results
import itertools


np.random.seed(6789)

warnings.filterwarnings("ignore")


counter = 0

myfunction_list=[]

myfunction_list.append(functions.hartman_3d(sd=0.05))



acq_type_list=[]


temp={}
temp['name']='random' 
temp['surrogate']='gp' 
#acq_type_list.append(temp)

temp={}
temp['name']='ucb' # vanilla UCB
temp['surrogate']='gp' 
acq_type_list.append(temp)


temp={}
temp['name']='ei' # vanilla EI
temp['surrogate']='gp' 
#acq_type_list.append(temp)




for idx, (myfunction,acq_type,) in enumerate(itertools.product(myfunction_list,acq_type_list)):
    func=myfunction.func
    
    func_params={}
    func_params['function']=myfunction

    gp_params = {'lengthscale':0.01*myfunction.input_dim,'noise_delta':1e-4} # the lengthscaled parameter will be optimized

    yoptimal=myfunction.fstar*myfunction.ismax
    
    acq_type['dim']=myfunction.input_dim-1
    acq_type['debug']=0
    acq_type['fstar']=myfunction.fstar

    acq_params={}
    acq_params['optimize_gp']='maximize'#maximize
    acq_params['acq_func']=acq_type
    
    nRepeat=10
    
    ybest=[0]*nRepeat
    MyTime=[0]*nRepeat
    MyOptTime=[0]*nRepeat
    marker=[0]*nRepeat

    bo=[0]*nRepeat
   
    [0]*nRepeat
    
    
    for ii in range(nRepeat):
        
        bo[ii]=BayesOpt_NoCond(gp_params,func_params,acq_params,verbose=0)
        #bo[ii]=BayesOpt_Cond(gp_params,func_params,acq_params,verbose=0)
  
        ybest[ii],MyTime[ii]=auxiliary_functions.run_experiment(bo[ii],gp_params,
             n_init=3*myfunction.input_dim,NN=30*myfunction.input_dim,runid=ii)                                               
        MyOptTime[ii]=bo[ii].time_opt
        print("ii={} BFV={}".format(ii,np.max(ybest[ii])))                                              
        

    Score={}
    Score["ybest"]=ybest
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    
    export_results.print_result_sequential(bo,myfunction,Score,acq_type) 