
#import pandas as pd
import numpy as np
from collections import OrderedDict
#from sklearn.metrics import average_precision_score,accuracy_score

        
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
		
class CogniteFunc_2D:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([ ('Intensity',(0, 1.6)),
                                   ('BehaviouralScore',(0,1))  ])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 4
        self.ismax=1
        self.name='CogniteFunc_2D'

     
    def run_Blackbox(self,X):
        # given hyper-parameter X
        utility=0
		
        # return the utility
        return utility
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax 	
    

class Cognite_2D_Spherical_Behavioural:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('SphericalHead',(48,70)),
                                   ('BehaviouralScore',(0,1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 10
        self.ismax=1
        self.name='Cognite_2D_Spherical_Behavioural'

     
    def run_Blackbox(self,X):
        # given hyper-parameter X
        utility=0
		
        # return the utility
        return utility
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax 	    
    
class Cognite_2D_Intensity_Spherical:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([ ('Intensity',(0, 1.6)),
                                       ('SphericalHead',(48,70))])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 10
        self.ismax=1
        self.name='Cognite_2D_Intensity_Spherical'

     
    def run_Blackbox(self,X):
        # given hyper-parameter X
        utility=0
		
        # return the utility
        return utility
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax 	    
    
class Cognite_2D_Intensity_Behavioural:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([ ('Intensity',(0, 1.6)),
                                   ('BehaviouralScore',(0,1))])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 10
        self.ismax=1
        self.name='Cognite_2D_Intensity_Behavioural'

     
    def run_Blackbox(self,X):
        # given hyper-parameter X
        utility=0
		
        # return the utility
        return utility
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax 	    
    
class CogniteFunc_3D:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([ ('Intensity',(0, 1.6)),
                                       ('SphericalHead',(48,70)),
                                   ('BehaviouralScore',(0,1))])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 10
        self.ismax=1
        self.name='CogniteFunc_3D'

     
    def run_Blackbox(self,X):
        # given hyper-parameter X
        utility=0
		
        # return the utility
        return utility
  
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax 	