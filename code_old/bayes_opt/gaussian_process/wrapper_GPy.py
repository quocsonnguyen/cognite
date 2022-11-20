

# define Gaussian Process class


import numpy as np
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from bayes_opt.gaussian_process.gaussian_process_base import GaussianProcessBase
from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name
import GPy

from scipy.spatial.distance import squareform

class GPyWrapper(GaussianProcessBase):
    
    def __init__ (self,param):
        # init the model
    
        # theta for RBF kernel exp( -theta* ||x-y||)
        if 'kernel' not in param:
            param['kernel']='SE'
            
        kernel_name=param['kernel']
        if kernel_name not in ['SE','ARD']:
            err = "The kernel function " \
                  "{} has not been implemented, " \
                  "please choose one of the kernel SE ARD.".format(kernel_name)
            raise NotImplementedError(err)
        else:
            self.kernel_name = kernel_name
            
        if 'flagIncremental' not in param:
            self.flagIncremental=0
        else:
            self.flagIncremental=param['flagIncremental']
            
        if 'lengthscale' not in param:
            self.lengthscale=param['theta']
        else:
            self.lengthscale=param['lengthscale']
            self.theta=self.lengthscale

        if 'lengthscale_vector' not in param: # for marginalize hyperparameters
            self.lengthscale_vector=[]
        else:
            self.lengthscale_vector=param['lengthscale_vector']
            
        #self.theta=param['theta']
        
        self.gp_params=param
        self.nGP=0
        # noise delta is for GP version with noise
        self.noise_delta=param['noise_delta']
        
        self.KK_x_x=[]
        self.KK_x_x_inv=[]
    
        self.X=[]
        self.Y=[]
        self.lengthscale_old=self.lengthscale
        self.flagOptimizeHyperFirst=0
        
        self.alpha=[] # for Cholesky update
        self.L=[] # for Cholesky update LL'=A

  
        
    def fit(self,X,Y):
        """
        Fit Gaussian Process model

        Input Parameters
        ----------
        x: the observed points 
        y: the outcome y=f(x)
        
        """ 
        # create simple GP Model
        self.model = GPy.models.GPRegression(X, Y)
        self.X=X
        self.Y=Y
        self.model.optimize('bfgs')
        print( "after optimized", self.model.rbf.lengthscale)



        
    def predict(self,xTest,eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        xTest=np.atleast_2d(xTest)
        try:
            mean,var = self.model.predict(xTest)
        except:
            print("bug")
        return mean.ravel(),var.ravel()
    
          