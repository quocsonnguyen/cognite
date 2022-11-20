# -*- coding: utf-8 -*-
# define Gaussian Process class


import numpy as np
from bayes_opt.acquisition_functions import unique_rows
from sklearn.preprocessing import MinMaxScaler

from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import scipy.linalg as spla
from bayes_opt.gaussian_process.gaussian_process_base import GaussianProcessBase

class ProductGaussianProcess_UniqueLoc(GaussianProcessBase):
    # in this class of Gaussian process, we define k( {x,t}, {x',t'} )= k(x,x')*k(t,t')
    
    #def __init__ (self,param):
    def __init__ (self,param):
        
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
            self.lengthscale_x=param['theta']
            self.lengthscale_t=param['theta']
        else:
            self.lengthscale_x=param['lengthscale']
            self.lengthscale_t=param['lengthscale']
            #self.theta=self.lengthscale_x

        if 'lengthscale_vector' not in param: # for marginalize hyperparameters
            self.lengthscale_vector=[]
        else:
            self.lengthscale_vector=param['lengthscale_vector']
            
        #self.theta=param['theta']
        
        self.gp_params=param
        
        # noise delta is for GP version with noise
        self.noise_delta=param['noise_delta']
        
        self.KK_x_x=[]
        self.KK_x_x_inv=[]
    
        self.X=[]
        self.Y=[]
        self.XT=[]
        self.T=[]
        self.X_full=[]
        self.T_full=[]
        self.XT_full=[]
        self.YT_full=[]
        self.repeated_counts=[]
    
        self.lengthscale_t_old=self.lengthscale_t
        self.lengthscale_x_old=self.lengthscale_x
        self.flagOptimizeHyperFirst=0
        
        self.alpha=[] # for Cholesky update
        self.L=[] # for Cholesky update LL'=A
       

    def cov_xx_pp(self, x1,t1,x2,t2,lengthscale,lengthscale_t):
        
        Euc_dist=euclidean_distances(x1,x2)
        exp_dist_x=np.exp(-np.square(Euc_dist)/lengthscale)
        
        Euc_dist=euclidean_distances(t1,t2)
        exp_dist_t=np.exp(-np.square(Euc_dist)/lengthscale_t)
        
        return exp_dist_x*exp_dist_t
               
    # identical row is when dist(x,x')<0.0001*d
    def find_unique_rows(self, X, y):
        
        y=y.ravel()
        
        distance = euclidean_distances(X,X)
        np.fill_diagonal(distance, 1)
        
        dim = X.shape[1]
        N = X.shape[0]
        
        idxRepeated = np.column_stack ( np.where(distance<0.01*dim) )
        
        if len(idxRepeated) == 0:
            return X, y, np.ones((len(y),1))
        
        #X_unique, repeated_counts = np.unique(X, return_counts=True)
        #y_unique = np.zeros((len(repeated_counts),1))
        
        X_unique = []
        y_unique = []
        repeated_counts = []
        
        selected_idx = []
        for iRow, xx in enumerate(X):
            
            if iRow in selected_idx:
                continue
            
            if iRow not in idxRepeated[:,0]:
                X_unique.append(xx)
                y_unique.append(y[iRow])
                repeated_counts.append(1)
                continue
            
            # look at the upper diagonal
            iCol = np.where(distance[iRow,int(np.floor(N*0.5)):] < 0.01*dim )[0]
            iCol = iCol + int(np.floor(N*0.5)) 
            
            # index = np.union(iCol, current_idx)
            idx = np.union1d(iRow, iCol)
            
            selected_idx += idx.tolist()
            
            new_X = np.mean( X[idx,:], axis=0)
            new_Y = np.mean(y[idx]).astype(float)
            
            X_unique.append(new_X)
            y_unique.append(new_Y)
            repeated_counts.append(len(idx))

            
        repeated_counts = np.asarray(repeated_counts).reshape((-1,1))
        y_unique = np.asarray(y_unique).reshape((-1,1))
        X_unique = np.asarray(X_unique).reshape((-1,dim))

        return X_unique, y_unique, repeated_counts
    
    def fit(self,X,T,Y):
        """
        Fit Gaussian Process model

        Input Parameters
        ----------
        x: the observed points including the personalized score
        y: the outcome y=f(x)
        
        """ 
        
        # XT = [X, T]
        XT = np.hstack((X,T))
        
        self.X_full=X
        self.T_full=T
        
        self.XT_full = XT
        self.Y_full = Y
        
        # now X,Y are X_unique, Y_unique
        self.XT, self.Y, \
            self.repeated_counts = self.find_unique_rows(self.XT_full,self.Y_full)
        
        XT = self.XT
        Y = self.Y
        
        
        T=np.reshape(XT[:,X.shape[1]:],(-1,T.shape[1])) # extract the last column as the personalized score
        X=np.reshape(XT[:,:X.shape[1] ],(len(Y),-1))
        
        self.X=X
        self.Y=Y
        self.T=T
        
            
        Euc_dist_x=euclidean_distances(X,X)
    
        Euc_dist_t=euclidean_distances(T,T)
    
        self.KK_x_x=np.exp(-np.square(Euc_dist_x)/self.lengthscale_x\
                           -np.square(Euc_dist_t)/self.lengthscale_t)+np.eye(len(X))*self.noise_delta/self.repeated_counts
          
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x")
        
        #self.KK_x_x_inv=np.linalg.pinv(self.KK_x_x)
        self.L=np.linalg.cholesky(self.KK_x_x)
        try:
            temp=np.linalg.solve(self.L,self.Y)
        except:
            print("bug")
        self.alpha=np.linalg.solve(self.L.T,temp)
        


    def compute_var(self,X,T,xTest,tTest):
        """
        compute variance given X and xTest
        
        Input Parameters
        ----------
        X: the observed points
        xTest: the testing points 
        
        Returns
        -------
        diag(var)
        """ 
        
        xTest=np.asarray(xTest)
        xTest=np.atleast_2d(xTest)
        
        tTest=np.asarray(tTest)
        tTest=np.atleast_2d(tTest)
        tTest=np.reshape(tTest,(-1,1))
        
        if self.kernel_name=='SE':
            myX=X
            myT=T
            
            Euc_dist_x=euclidean_distances(myX,myX)
        
            Euc_dist_t=euclidean_distances(myT,myT)
        
            KK=np.exp(-np.square(Euc_dist_x)/self.hyper['lengthscale_x']-np.square(Euc_dist_t)/self.hyper['lengthscale_t'])\
                +np.eye(len(myX))*self.noise_delta
                    
                 
            Euc_dist_test_train_x=euclidean_distances(xTest,X)
            
            Euc_dist_test_train_t=euclidean_distances(tTest,T)
            
            KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train_x)/self.hyper['lengthscale_x']-np.square(Euc_dist_test_train_t)/self.hyper['lengthscale_t'])
                
        try:
            temp=np.linalg.solve(KK,KK_xTest_xTrain.T)
        except:
            temp=np.linalg.lstsq(KK,KK_xTest_xTrain.T, rcond=-1)
            temp=temp[0]
            
        #var=KK_xTest_xTest-np.dot(temp.T,KK_xTest_xTrain.T)
        var=np.eye(xTest.shape[0])-np.dot(temp.T,KK_xTest_xTrain.T)
        var=np.diag(var)
        var.flags['WRITEABLE']=True
        var[var<1e-100]=0
        return var 

    
        
    def predict(self,xTest, eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    

        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]+1))
            
        tTest=xTest[:,-1]
        tTest=np.atleast_2d(tTest)
        tTest=np.reshape(tTest,(xTest.shape[0],-1))
        
        xTest=xTest[:,:-1]
        

        
        self.XT, self.Y, \
            self.repeated_counts = self.find_unique_rows(self.XT_full,self.Y_full)
        
        self.X = self.XT[:,:-1]
        self.T = self.XT[:,-1]
        X = self.X
        T= np.reshape( self.T, (-1,1))
                
        Euc_dist_x=euclidean_distances(xTest,xTest)
        Euc_dist_t=euclidean_distances(tTest,tTest)

        KK_xTest_xTest=np.exp(-np.square(Euc_dist_x)/self.lengthscale_x-np.square(Euc_dist_t)/self.lengthscale_t)\
            +np.eye(xTest.shape[0])*self.noise_delta
        
        Euc_dist_test_train_x=euclidean_distances(xTest,X)
        
        Euc_dist_test_train_t=euclidean_distances(tTest,T)
        
        KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train_x)/self.lengthscale_x-np.square(Euc_dist_test_train_t)/self.lengthscale_t)
            
        #Exp_dist_test_train_x*Exp_dist_test_train_t
  
        # using Cholesky update
        mean=np.dot(KK_xTest_xTrain,self.alpha)
        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        var=KK_xTest_xTest-np.dot(v.T,v)
        

        return mean.ravel(),np.diag(var)  

    def posterior(self,x):
        # compute mean function and covariance function
        return self.predict(self,x)
    
    def optimize_lengthscale_SE_maximizing(self,previous_theta,noise_delta):
        """
        Optimize to select the optimal lengthscale parameter
        """
                
        # define a bound on the lengthscale
        #SearchSpace_lengthscale_min=0.001
        #SearchSpace_lengthscale_max=0.2
        #mySearchSpace=[np.asarray([SearchSpace_lengthscale_min,SearchSpace_lengthscale_max]).T]
        

        mySearchSpace=np.asarray([[0.01,0.2],\
                            [0.01,0.2]])

        
        # Concatenate new random points to possible existing
        # points from self.explore method.
        lengthscale_tries = np.random.uniform(mySearchSpace[:, 0], mySearchSpace[:, 1],size=(20, mySearchSpace.shape[0]))

        #print lengthscale_tries

        # evaluate
        self.flagOptimizeHyperFirst=0 # for efficiency

        logmarginal_tries=self.log_marginal_lengthscale(lengthscale_tries,noise_delta)
        #print logmarginal_tries

        #find x optimal for init
        idx_max=np.argmax(logmarginal_tries)
        lengthscale_init_max=lengthscale_tries[idx_max]
        #print lengthscale_init_max
        
        
        dim=self.X.shape[1]
        myopts ={'maxiter':20*dim,'maxfun':20*dim}

        x_max=[]
        max_log_marginal=None
        
        res = minimize(lambda x: -self.log_marginal_lengthscale(x,noise_delta),lengthscale_init_max,
                      bounds=mySearchSpace,method="L-BFGS-B",options=myopts)#L-BFGS-B
        if 'x' not in res:
            val=self.log_marginal_lengthscale(res,noise_delta)    
        else:
            val=self.log_marginal_lengthscale(res.x,noise_delta)  
        
        # Store it if better than previous minimum(maximum).
        if max_log_marginal is None or val >= max_log_marginal:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_log_marginal = val
            #print res.x

        return x_max
    
    def optimize_lengthscale(self,previous_theta_x, previous_theta_t,noise_delta):

        prev_theta=[previous_theta_x,previous_theta_t]
        newlengthscale,newlengthscale_t=self.optimize_lengthscale_SE_maximizing(prev_theta,noise_delta)
        self.lengthscale_x=newlengthscale
        self.lengthscale_t=newlengthscale_t
        
        # refit the model
        self.fit(self.X_full, self.T_full, self.Y_full)
        
        return newlengthscale,newlengthscale_t
    
    def log_marginal_lengthscale(self,hyper,noise_delta):
        """
        Compute Log Marginal likelihood of the GP model w.r.t. the provided lengthscale, noise_delta and Logistic hyperparameter
        """

        def compute_log_marginal(lengthscale_x, lengthscale_t,noise_delta):
            # compute K

            myX = self.X
            myY = self.Y
            myT = self.T
            
            self.Euc_dist_x=euclidean_distances(myX,myX)
            self.Euc_dist_t=euclidean_distances(myT,myT)
        
            KK=np.exp(-np.square(self.Euc_dist_x)/lengthscale_x-np.square(self.Euc_dist_t)/lengthscale_t)\
                +np.eye(len(myX))*noise_delta
            
            try:
                temp_inv=np.linalg.solve(KK,myY)
            except: # singular
                return -np.inf
            
            try:
                #logmarginal=-0.5*np.dot(self.Y.T,temp_inv)-0.5*np.log(np.linalg.det(KK+noise_delta))-0.5*len(X)*np.log(2*3.14)
                first_term=-0.5*np.dot(myY.T,temp_inv)
                
                # if the matrix is too large, we randomly select a part of the data for fast computation
                if KK.shape[0]>200:
                    idx=np.random.permutation(KK.shape[0])
                    idx=idx[:200]
                    KK=KK[np.ix_(idx,idx)]
                #Wi, LW, LWi, W_logdet = pdinv(KK)
                #sign,W_logdet2=np.linalg.slogdet(KK)
                chol  = spla.cholesky(KK, lower=True)
                W_logdet=np.sum(np.log(np.diag(chol)))
                # Uses the identity that log det A = log prod diag chol A = sum log diag chol A
    
                #second_term=-0.5*W_logdet2
                second_term=-W_logdet
            except: # singular
                return -np.inf
            

            logmarginal=first_term+second_term-0.5*len(myY)*np.log(2*3.14)
                
            if np.isnan(np.float(logmarginal))==True:
                print("lengthscale_x={:f} lengthscale_t={:f} first term ={:.4f} second  term ={:.4f}".format(
                        lengthscale_x,lengthscale_t,np.float(first_term),np.float(second_term)))

            return np.float(logmarginal)
        
        logmarginal=0

        if not isinstance(hyper,list) and len(hyper.shape)==2:
            logmarginal=[0]*hyper.shape[0]
        
            lengthscale_t=hyper[:,1]
            lengthscale_x=hyper[:,0]
            for idx in range(hyper.shape[0]):
                logmarginal[idx]=compute_log_marginal(lengthscale_x[idx],\
                           lengthscale_t[idx],noise_delta)
        else:
            lengthscale_x,lengthscale_t=hyper
            logmarginal=compute_log_marginal(lengthscale_x,lengthscale_t,noise_delta)
        return logmarginal