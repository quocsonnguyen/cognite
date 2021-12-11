
import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats



counter = 0


class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq):
        
        self.acq=acq
        acq_name=acq['name']
        
        ListAcq=['bucb','ucb', 'ei', 'poi','random','thompson',    'mu',                     
                     'pure_exploration','kov_mes','mes','kov_ei',
                         'kov_erm','kov_cbm','kov_tgp','kov_tgp_ei']
        
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        self.dim=acq['dim']
        
        if 'scalebounds' not in acq:
            self.scalebounds=[0,1]*self.dim
            
        else:
            self.scalebounds=acq['scalebounds']
        
        # vector theta for thompson sampling
        #self.flagTheta_TS=0
        self.initialized_flag=0
        self.objects=[]
        
    def acq_kind(self, x, gp):
        
        y_max=np.max(gp.Y)
        
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'bucb':
            return self._bucb(x, gp, self.acq['kappa'])
        if self.acq_name == 'ucb':
            return self._ucb(x, gp)
        if self.acq_name=='kov_cbm':
            return self._cbm(x,gp,target=self.acq['fstar_scaled'])
        if self.acq_name == 'lcb':
            return self._lcb(x, gp)
        if self.acq_name == 'ei' or self.acq_name=='kov_tgp_ei':
            return self._ei(x, gp, y_max)
     
        if self.acq_name == 'pure_exploration':
            return self._pure_exploration(x, gp) 
       
        if self.acq_name == 'ei_mu':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'mu':
            return self._mu(x, gp)
      
        if self.acq_name == 'thompson':
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.ThompsonSampling(gp)
                self.initialized_flag=1
                return self.object(x,gp)
            else:
                return self.object(x,gp)
     

    @staticmethod
    def _mu(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        mean=np.atleast_2d(mean).T
        return mean
                
    @staticmethod
    def _lcb(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = 2 * np.log(len(gp.Y));

        return mean - np.sqrt(beta_t) * np.sqrt(var) 
        
    
    @staticmethod
    def _ucb(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
  
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        return mean + np.sqrt(beta_t) * np.sqrt(var) 
    
    @staticmethod
    def _cbm(x, gp, target): # confidence bound minimization
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
  
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        return -np.abs(mean-target) - np.sqrt(beta_t) * np.sqrt(var) 
    
  
   
    @staticmethod
    def _erm(x, gp, fstar):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        
        if mean.shape!=var.shape:
            print("bug")
            mean, var = gp.predict(x, eval_MSE=True)
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (mean - fstar)/np.sqrt(var2)        
        out=(fstar-mean) * (1-norm.cdf(z)) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        
        #print(out)
        if any(out)<0:
            print("out<0")
        return -1*out # for minimization problem
                    
    @staticmethod
    def _bucb(x, gp, kappa):
        mean, var = gp.predict_bucb(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        return mean + kappa * np.sqrt(var)
    
    @staticmethod
    def _ei(x, gp, y_max):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        return out
       
 
     
  

    @staticmethod      
    def _poi(x, gp,y_max): # run Predictive Entropy Search using Spearmint
        mean, var = gp.predict(x, eval_MSE=True)    
        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)
        z = (mean - y_max)/np.sqrt(var)        
        return norm.cdf(z)

    @staticmethod      
    def _poi_kov(x, gp,y_max): # POI with Known Optimal Value
        mean, var = gp.predict(x, eval_MSE=True)    
        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)
        z = np.abs(y_max-mean)/np.sqrt(var)        
        return -norm.cdf(z)
    
    class MaxValueEntropySearch(object):
        def __init__(self,gp,boundaries,ystars=[]):

            self.X=gp.X
            self.Y=gp.Y
            self.gp=gp
            if ystars==[]:
                print("y_star is empty for MES")                
            self.y_stars=ystars
                
        def __call__(self,x):
            mean_x, var_x = self.gp.predict(x, eval_MSE=True)

            acq_value=0
            for idx,val in enumerate(self.y_stars):
                gamma_ystar=(val-mean_x)*1.0/var_x
                temp=0.5*gamma_ystar*norm.pdf(gamma_ystar)*1.0/norm.cdf(gamma_ystar)-np.log(norm.cdf(gamma_ystar))
                acq_value=acq_value+temp
            #acq_value=acq_value*1.0/len(self.y_stars)
            return acq_value
        
    class ThompsonSampling(object):
        def __init__(self,gp):
            dim=gp.X.shape[1]
            # used for Thompson Sampling
            #self.WW_dim=200 # dimension of random feature
            
            self.WW_dim=30*dim # dimension of random feature
            self.WW=np.random.multivariate_normal([0]*self.WW_dim,np.eye(self.WW_dim),dim)/gp.lengthscale  
            self.bias=np.random.uniform(0,2*3.14,self.WW_dim)

            # computing Phi(X)^T=[phi(x_1)....phi(x_n)]
            Phi_X=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(gp.X,self.WW)+self.bias), np.cos(np.dot(gp.X,self.WW)+self.bias)]) # [N x M]
            
            # computing A^-1
            A=np.dot(Phi_X.T,Phi_X)+np.eye(2*self.WW_dim)*gp.noise_delta
            
            # theta ~ N( A^-1 Phi_T Y, sigma^2 A^-1
            gx=np.dot(Phi_X.T,gp.Y)
            self.mean_theta_TS=np.linalg.solve(A,gx)
            
        def __call__(self,x,gp):
            #phi_x=np.sqrt(1.0/self.UU_dim)*np.hstack([np.sin(np.dot(x,self.UU)), np.cos(np.dot(x,self.UU))])
            phi_x=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(x,self.WW)+self.bias), np.cos(np.dot(x,self.WW)+self.bias)])
            
            # compute the mean of TS
            gx=np.dot(phi_x,self.mean_theta_TS)    
            return gx  

def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]



class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
