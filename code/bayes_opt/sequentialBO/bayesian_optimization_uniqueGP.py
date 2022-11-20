# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


import numpy as np
#from sklearn.gaussian_process import GaussianProcess

from bayes_opt.sequentialBO.bayesian_optimization_base import BO_Sequential_Base
from scipy.optimize import minimize
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
#from bayes_opt.gaussian_process.gaussian_process import GaussianProcess
#from bayes_opt.gaussian_process.product_gaussian_process import ProductGaussianProcess
#from bayes_opt.gaussian_process.gaussian_process_unique_locations import GP_UniqueLocations
from bayes_opt.gaussian_process.product_gaussian_process_unique_loc import ProductGaussianProcess_UniqueLoc

from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name


import time

#import nlopt

#@author: Vu

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt_UniqueLoc(BO_Sequential_Base):

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

        # Find number of parameters
        
        super(BayesOpt_UniqueLoc, self).__init__(gp_params, func_params, acq_params, verbose)
        self.dim_X = acq_params['acq_func']['dim'][0]
        self.dim_T = acq_params['acq_func']['dim'][1]
        self.maxGPmean=None

      
       
    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X,self.T,self.Y) # last column is the personalized score

        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, gp_params, n_init_points=3,seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        super(BayesOpt_UniqueLoc, self).init(gp_params, n_init_points,seed)
        
    def init_with_data(self, init_X, init_T, init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """

        #super(BayesOpt_UniqueLoc, self).init_with_data(init_X,init_Y)
        
        #self.X_original=np.asarray(init_X)
        self.X_original = np.reshape( init_X,(-1,self.dim_X))
        
        temp_init_point=np.divide((init_X-self.bounds[:self.dim_X,0]),self.max_min_gap[:self.dim_X])
        self.X = np.reshape(temp_init_point,(-1,self.dim_X))

        self.T_original = np.reshape(init_T,(-1,self.dim_T))
        temp_init_point=np.divide((init_T-self.bounds[self.dim_X:,0]),self.max_min_gap[self.dim_X:])
        self.T = np.reshape(temp_init_point,(-1,self.dim_T))
        self.X_original_maxGP= np.asarray(init_X)

        self.Y_original = np.asarray(init_Y)
        self.Y_original_maxGP=np.asarray(init_Y)      

        #self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)        
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
        
        # Find unique rows of X to avoid GP from breaking
        self.gp.fit(self.X, self.T, self.Y)
        
           
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
            
        if self.acq['name']=='random':
            
            super(BayesOpt_UniqueLoc, self).generate_random_point()

            return

        # init a new Gaussian Process
        self.gp=ProductGaussianProcess_UniqueLoc(self.gp_params)
        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            #ur = unique_rows(self.X)
            #self.gp.fit(self.X[ur], self.Y[ur])
            self.gp.fit(self.X, self.Y)

 
        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        #if  len(self.Y)%(2*self.dim)==0:
        #self.gp,self.gp_params=super(BayesOpt, self).optimize_gp_hyperparameter()


        # Set acquisition function
        start_opt=time.time()

        #y_max = self.Y.max()
                      
    
        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)


        val_acq=self.acq_func.acq_kind(x_max,self.gp)

        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            #val_acq=self.acq_func.acq_kind(x_max,self.gp)

            self.stop_flag=1
            #print "Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria)
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        super(BayesOpt_UniqueLoc, self).augment_the_new_data(x_max)
        
    def maximize_given_2d_context(self,mycondition):
        """
        suggest the next experiment, given the [d-2,d-1] parameters fixed.
        
        mycondition: [2x1] array
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            super(BayesOpt_UniqueLoc, self).generate_random_point()
            return

        # init a new Gaussian Process
        self.gp=ProductGaussianProcess_UniqueLoc(self.gp_params)

        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            #ur = unique_rows(self.X)
            #self.gp.fit(self.X[ur], self.Y[ur])
            
            self.gp.fit(self.X, self.T, self.Y) # last two column is the personalized score

        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        #if  len(self.Y)%(2*self.dim)==0:
        #self.gp,self.gp_params=super(BayesOpt, self).optimize_gp_hyperparameter()


        # Set acquisition function
        start_opt=time.time()

        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
        # compute the scaled contexts
        mycondition_scale=(mycondition-self.bounds[-2:,0])/(self.bounds[-2:,1]-self.bounds[-2:,0])
        
        # we only optimize on the other dimension 
        # while keeping fixed the last two dimensions
        
        mybounds=np.copy(self.scalebounds)
        mybounds[-self.dim_T:,0]=mycondition_scale
        mybounds[-self.dim_T:,1]=mycondition_scale
        

        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=mybounds,
                        opt_toolbox=self.opt_toolbox)#seeds=self.xstars

        val_acq=self.acq_func.acq_kind(x_max,self.gp)

        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            self.stop_flag=1
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        
        #super(BayesOpt_UniqueLoc, self).augment_the_new_data(x_max[:self.dim_X])
        
        
        self.T = np.vstack(( self.T, np.reshape(mycondition_scale,(1,-1))))
        
        
        self.X = np.vstack((self.X, x_max[:self.dim_X].reshape((1, -1))))
        # compute X in original scale
        temp_XT_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        temp_XT_new_original = np.reshape(temp_XT_new_original ,(1,-1))
        self.X_original=np.vstack((self.X_original, temp_XT_new_original[:,:self.dim_X]))
        self.T_original=np.vstack((self.T_original, temp_XT_new_original[:,self.dim_X:]))

        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_original = np.append(self.Y_original, self.f(temp_XT_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)

        
        #x_mu_max_original=x_mu_max*self.max_min_gap+self.bounds[:,0]

        #self.Y_original_maxGP = np.append(self.Y_original_maxGP, self.f(x_mu_max_original))
        #self.X_original_maxGP = np.vstack((self.X_original_maxGP, x_mu_max_original))

        





        
        new_original=self.X_original[-1]
        self.X_original=self.X_original[:-1]
        self.Y_original=self.Y_original[:-1]
        self.X=self.X[:-1]
        self.Y=self.Y[:-1]
        self.X_original_maxGP=self.X_original_maxGP[:-1]
        self.Y_original_maxGP=self.Y_original_maxGP[:-1]

        return new_original
    

    def maximize_condition(self,mycondition):
        """
        suggest the next experiment, given the d-1 parameter fixed.
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            
            super(BayesOpt_UniqueLoc, self).generate_random_point()

            return

        # init a new Gaussian Process
        self.gp=ProductGaussianProcess_UniqueLoc(self.gp_params)

        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            #ur = unique_rows(self.X)
            #self.gp.fit(self.X[ur], self.Y[ur])
            
            self.gp.fit(self.X,self.Y) # last column is the personalized score

        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        #if  len(self.Y)%(2*self.dim)==0:
        #self.gp,self.gp_params=super(BayesOpt, self).optimize_gp_hyperparameter()


        # Set acquisition function
        start_opt=time.time()

        #y_max = self.Y.max()

        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
        mycondition_scale=(mycondition-self.bounds[-1,0])/(self.bounds[-1,1]-self.bounds[-1,0])
        # mybound
        mybounds=np.copy(self.scalebounds)
        mybounds[-1,0]=mycondition_scale
        mybounds[-1,1]=mycondition_scale

        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=mybounds,
                        opt_toolbox=self.opt_toolbox)#seeds=self.xstars

        val_acq=self.acq_func.acq_kind(x_max,self.gp)

        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            #val_acq=self.acq_func.acq_kind(x_max,self.gp)

            self.stop_flag=1
            #print "Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria)
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        
        super(BayesOpt_UniqueLoc, self).augment_the_new_data(x_max[:self.dim_X])
        
        
        
        
        new_original=self.X_original[-1]
        self.X_original=self.X_original[:-1]
        self.Y_original=self.Y_original[:-1]
        self.X=self.X[:-1]
        self.Y=self.Y[:-1]
        self.X_original_maxGP=self.X_original_maxGP[:-1]
        self.Y_original_maxGP=self.Y_original_maxGP[:-1]

        return new_original

        
#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
