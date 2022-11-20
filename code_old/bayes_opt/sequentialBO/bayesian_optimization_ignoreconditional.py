# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


import numpy as np
#from sklearn.gaussian_process import GaussianProcess

from bayes_opt.sequentialBO.bayesian_optimization_base import BO_Sequential_Base
from scipy.optimize import minimize
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from bayes_opt.gaussian_process.gaussian_process import GaussianProcess

from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name


import time

#import nlopt


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt_NoCond(BO_Sequential_Base):

    def __init__(self, gp_params, func_params, acq_params, verbose=1):
        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.theta:            to compute the kernel
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:        bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        try:
            bounds=func_params['function']['bounds']
        except:
            bounds=func_params['function'].bounds
        
        self.dim = len(bounds)-1

        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in list(bounds.keys()):
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)

 
        self.bounds=self.bounds[:-1,:]
        
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        
        # Some function to be optimized
        
        try:
            self.f = func_params['function']['func']
        except:
            self.f = func_params['function'].func
            
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']
        # acquisition function type
        
        self.acq=acq_params['acq_func']
        self.acq['scalebounds']=self.scalebounds
        
        if 'debug' not in self.acq:
            self.acq['debug']=0           
        if 'stopping' not in acq_params:
            self.stopping_criteria=0
        else:
            self.stopping_criteria=acq_params['stopping']
        if 'optimize_gp' not in acq_params:
            self.optimize_gp='maximize' # default is using maximum marginal llk
        else:                
            self.optimize_gp=acq_params['optimize_gp']    
            
            
        if 'marginalize_gp' not in acq_params:
            self.marginalize_gp=0
        else:                
            self.marginalize_gp=acq_params['marginalize_gp']
            
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None
        
        # performance evaluation at the maximum mean GP (for information theoretic)
        self.Y_original_maxGP = None
        self.X_original_maxGP = None
        
     
        
        self.time_opt=0

        
        self.gp_params=gp_params       
        self.acq_params=acq_params

        # Gaussian Process class
     
        if self.acq['surrogate']=='gp':
            self.gp=GaussianProcess(gp_params)
    

        #self.gp=GaussianProcess(gp_params)

        # acquisition function
        self.acq_func = None
    
        # stop condition
        self.stop_flag=0
        self.logmarginal=0
        
        # xt_suggestion, caching for Consensus
        self.xstars=[]
        self.xstar_accumulate=[]

        # theta vector for marginalization GP
        self.theta_vector =[]
        
    
        
        # store ystars
        #self.ystars=np.empty((0,100), float)
        self.ystars=[]

       
    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, n_init_points=3,seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        np.random.seed(seed)
        # Generate random points
        l = [np.random.uniform(x[0], x[1]) for _ in range(n_init_points) for x in self.bounds]
        #l=[np.linspace(x[0],x[1],num=n_init_points) for x in self.init_bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))
        
        self.X_original = np.asarray(init_X)
        self.X_original_maxGP= np.asarray(init_X)
        
        # Evaluate target function at all initialization   

        # randomly create a third dimension and append to init
        rand_vec=np.random.uniform(0,1,n_init_points)       
        rand_vec=np.reshape(rand_vec,(n_init_points,1))
        temp_init_X=np.asarray(init_X)
        temp_init_X=np.hstack((temp_init_X,rand_vec))
        temp_init_X=temp_init_X.tolist()
        y_init=self.f(temp_init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_original = np.asarray(y_init)      
        
        self.Y_original_maxGP=np.asarray(y_init)      
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)

        # convert it to scaleX
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X = np.asarray(temp_init_point)
        
        
    def init_with_data(self, init_X,init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """

        super(BayesOpt_NoCond, self).init_with_data(init_X,init_Y)
        
           
    def estimate_L(self,bounds):
        '''
        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        '''
        def df(x,model,x0):
            mean_derivative=gp_model.predictive_gradient(self.X,self.Y,x)
            
            temp=mean_derivative*mean_derivative
            if len(temp.shape)<=1:
                res = np.sqrt( temp)
            else:
                res = np.sqrt(np.sum(temp,axis=1)) # simply take the norm of the expectation of the gradient        

            return -res

        gp_model=self.gp
                
        dim = len(bounds)
        num_data=1000*dim
        samples = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): samples[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)

        #samples = np.vstack([samples,gp_model.X])
        pred_samples = df(samples,gp_model,0)
        x0 = samples[np.argmin(pred_samples)]

        res = minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (gp_model,x0), options = {'maxiter': 100})        
   
        try:
            minusL = res.fun[0][0]
        except:
            if len(res.fun.shape)==1:
                minusL = res.fun[0]
            else:
                minusL = res.fun
                
        L=-minusL
        if L<1e-6: L=0.0001  ## to avoid problems in cases in which the model is flat.
        
        return L    
      
            
    def maximize(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random' or self.acq['name']=='rand':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            rand_vec=np.random.uniform(0,1,1)       
            rand_vec=np.atleast_2d(rand_vec)
            temp_X_new_original=np.hstack((x_max,rand_vec))
            
            
            temp=self.f(temp_X_new_original)
            self.Y_original = np.append(self.Y_original, temp)
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
            self.time_opt=np.hstack((self.time_opt,0))
            
            # convert it to scaleX
            temp_init_point=np.divide((x_max-self.bounds[:,0]),self.max_min_gap)
            
            self.X = np.vstack((self.X,temp_init_point))
            return


        ur = unique_rows(self.X)
        self.Y=np.reshape(self.Y,(-1,1))
        self.gp.fit(self.X[ur], self.Y[ur])
            
 
        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        if  self.optimize_gp != "0" and len(self.Y)%(1*self.dim)==0:
            self.gp,self.gp_params=super(BayesOpt_NoCond, self).optimize_gp_hyperparameter()


        # Set acquisition function
        start_opt=time.time()

   
        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,
                        opt_toolbox=self.opt_toolbox,seeds=self.xstars)


        val_acq=self.acq_func.acq_kind(x_max,self.gp)

        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            #val_acq=self.acq_func.acq_kind(x_max,self.gp)

            self.stop_flag=1
            #print "Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria)
        
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
 
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))
        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
               
        rand_vec=np.random.uniform(0,1,1)       
        temp_X_new_original=np.hstack((temp_X_new_original,rand_vec))
        
        
        temp=self.f(temp_X_new_original)
        self.Y_original = np.append(self.Y_original, temp)
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)


#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
