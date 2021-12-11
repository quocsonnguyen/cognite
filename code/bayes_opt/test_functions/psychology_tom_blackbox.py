

import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score,accuracy_score


#import matlab.engine
#import matlab
#eng = matlab.engine.start_matlab()
        
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
class PsychologyTom_CF:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 2
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('Current',(0, 1.6)),('Frequency',(4,50))  ])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 4
        self.ismax=1
        self.name='Psy_Tom_CF'

     
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
		
class PsychologyTom_FA:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 2
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('Frequency',(4,50)), ('Ability',(0.0205,0.1142))   ])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 4
        self.ismax=1
        self.name='Psy_Tom_FA'

     
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

class PsychologyTom_CA:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 2
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('Current',(0, 1.6)), ('Ability',(0.0205,0.1142))   ])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 4
        self.ismax=1
        self.name='Psy_Tom_FA'

    def run_Blackbox(self,X):
        # given hyper-parameter X
        utility=0
        return utility 
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
   
        if len(X.shape)==1: # 1 data point
            Utility=self.run_Blackbox(X)
        else:
            Utility=np.apply_along_axis( self.run_Blackbox,1,X)

        return Utility*self.ismax 
		
class PsychologyTom_3D:
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('Current',(0, 1.6)),('Frequency',(4,50)),
                                   ('Ability',(0.0205,0.1142))  ])
    
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 4
        self.ismax=1
        self.name='Psy_Tom'

     
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