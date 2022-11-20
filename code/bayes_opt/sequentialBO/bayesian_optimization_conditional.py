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
from bayes_opt.gaussian_process.product_gaussian_process import ProductGaussianProcess

from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name


import time

#import nlopt


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt_Cond(BO_Sequential_Base):

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
        
        super(BayesOpt_Cond, self).__init__(gp_params, func_params, acq_params, verbose)

     
       
       
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
        # rand_vec=np.random.uniform(0,1,n_init_points)       
        # rand_vec=np.reshape(rand_vec,(n_init_points,1))
        # temp_init_X=np.asarray(init_X)
        # temp_init_X=np.hstack((temp_init_X,rand_vec))
        # temp_init_X=temp_init_X.tolist()
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_original = np.asarray(y_init)      
        
        self.Y_original_maxGP=np.asarray(y_init)    
        self.maxGPmean=np.asarray(y_init)      

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
        super(BayesOpt_Cond, self).init_with_data(init_X,init_Y)       
           
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
      
            
    # def maximize(self):
    #     """
    #     Main optimization method.

    #     Input parameters
    #     ----------
    #     gp_params: parameter for Gaussian Process

    #     Returns
    #     -------
    #     x: recommented point for evaluation
    #     """

    #     if self.acq['name']=='random' or self.acq['name']=='rand':
    #         super(BayesOpt_Cond, self).generate_random_point()
    #         return


    #     ur = unique_rows(self.X)
    #     self.Y=np.reshape(self.Y,(-1,1))
    #     self.gp.fit(self.X[ur], self.Y[ur])
            
 
    #     acq=self.acq
       
    #     # optimize GP parameters after 10 iterations
    #     if  len(self.Y)%(5*self.dim)==0:
    #         self.gp,self.gp_params=super(BayesOpt_Cond, self).optimize_gp_hyperparameter()

    
    #     # Set acquisition function
    #     start_opt=time.time()

    #     #y_max = self.Y.max()
                      
    #     self.acq_func = AcquisitionFunction(self.acq)

    #     if acq['name']=="ei_mu":
    #         #find the maximum in the predictive mean            
    #         x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
    
    #     rand_vec=np.random.uniform(0,1,1)       
    #     mybound=np.copy(self.scalebounds)
    #     mybound[-1,0]=rand_vec
    #     mybound[-1,1]=rand_vec


    #     x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=mybound,
    #                     opt_toolbox=self.opt_toolbox)


    #     val_acq=self.acq_func.acq_kind(x_max,self.gp)

    #     if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
    #         #val_acq=self.acq_func.acq_kind(x_max,self.gp)

    #         self.stop_flag=1
    #         #print "Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria)
        
        
    #     self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
    #     mean,var=self.gp.predict(x_max, eval_MSE=True)
    #     var.flags['WRITEABLE']=True
    #     var[var<1e-20]=0
       
    #     # record the optimization time
    #     finished_opt=time.time()
    #     elapse_opt=finished_opt-start_opt
    #     self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        
    #     super(BayesOpt_Cond, self).augment_the_new_data(x_max)
    
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

      
        # draw a random value in the original space
        rand_vec=np.random.uniform(self.bounds[-1,0],self.bounds[-1,1],1)
        #print(rand_vec)
        
        return self.maximize_condition(rand_vec)

        
    def maximize_condition(self,mycondition):
        """
        suggest the next experiment, given the d-1 parameter fixed.
        """

        
        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            
            super(BayesOpt_Cond, self).generate_random_point()

            return

        # init a new Gaussian Process
        self.gp=ProductGaussianProcess(self.gp_params)
        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
            

        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        if  self.optimize_gp != "0" and len(self.Y)%(1*self.dim)==0:
            temp=self.gp.optimize_lengthscale(self.gp.lengthscale_x_old, self.gp.lengthscale_t_old,self.gp.noise_delta)
            self.gp_params['lengthscale_x']=temp[0]
            self.gp_params['lengthscale_t']=temp[1]


        # Set acquisition function
        start_opt=time.time()

        #y_max = self.Y.max()
  
        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)

        # scale the condition value from original scale to [0,1]
        mycondition_scale=(mycondition-self.bounds[-1,0])/(self.bounds[-1,1]-self.bounds[-1,0])
        mybounds=np.copy(self.scalebounds)
        mybounds[-1,0]=mycondition_scale
        mybounds[-1,1]=mycondition_scale

        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=mybounds,
                        opt_toolbox=self.opt_toolbox)
        
        #print("xmax",x_max)

        val_acq=self.acq_func.acq_kind(x_max,self.gp) 
        
        #print("val_acq",val_acq)

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
        
        
        super(BayesOpt_Cond, self).augment_the_new_data(x_max)
        
        new_original=self.X_original[-1]
        # self.X_original=self.X_original[:-1]
        # self.Y_original=self.Y_original[:-1]
        # self.X=self.X[:-1]
        # self.Y=self.Y[:-1]
        # self.X_original_maxGP=self.X_original_maxGP[:-1]
        # self.Y_original_maxGP=self.Y_original_maxGP[:-1]

        return new_original
    


#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
