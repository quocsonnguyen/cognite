# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:22:32 2016

@author: Vu
"""
from __future__ import division

import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'..')
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics.pairwise import euclidean_distances
from bayes_opt.acquisition_maximization import acq_max
from scipy.stats import norm as norm_dist

import random
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
import os
from pylab import *

cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.7),
                  (1.0, 1.0, 1.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'blue': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.5, 1.0))}

#my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
#my_cmap = plt.get_cmap('cubehelix')
my_cmap = plt.get_cmap('Blues')

        
counter = 0

#class Visualization(object):
    
    #def __init__(self,bo):
       #self.plot_gp=0     
       #self.posterior=0
       #self.myBo=bo


out_dir=""
        
def plot_bo(bo):
    if bo.dim==1:
        plot_bo_1d(bo)
    if bo.dim==2:
        plot_bo_2d(bo)
    
def plot_histogram(bo,samples):
    if bo.dim==1:
        plot_histogram_1d(bo,samples)
    if bo.dim==2:
        plot_histogram_2d(bo,samples)

def plot_mixturemodel(g,bo,samples):
    if bo.dim==1:
        plot_mixturemodel_1d(g,bo,samples)
    if bo.dim==2:
        plot_mixturemodel_2d(g,bo,samples)

def plot_mixturemodel_1d(g,bo,samples):
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]

    x_plot = np.linspace(np.min(samples), np.max(samples), len(samples))
    x_plot = np.reshape(x_plot,(len(samples),-1))
    y_plot = g.score_samples(x_plot)[0]
    
    x_plot_ori = np.linspace(np.min(samples_original), np.max(samples_original), len(samples_original))
    x_plot_ori=np.reshape(x_plot_ori,(len(samples_original),-1))
    
    
    fig=plt.figure(figsize=(8, 3))

    plt.plot(x_plot_ori, np.exp(y_plot), color='red')
    plt.xlim(bo.bounds[0,0],bo.bounds[0,1])
    plt.xlabel("X",fontdict={'size':16})
    plt.ylabel("f(X)",fontdict={'size':16})
    plt.title("IGMM Approximation",fontsize=16)
        
def plot_mixturemodel_2d(dpgmm,bo,samples):
    
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]
    dpgmm_means_original=dpgmm.truncated_means_*bo.max_min_gap+bo.bounds[:,0]

    #fig=plt.figure(figsize=(12, 5))
    fig=plt.figure()
    myGmm=fig.add_subplot(1,1,1)  

    x1 = np.linspace(bo.scalebounds[0,0],bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0],bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    x_plot=np.c_[x1g.flatten(), x2g.flatten()]
    
    y_plot2 = dpgmm.score_samples(x_plot)[0]
    y_plot2=np.exp(y_plot2)
    #y_label=dpgmm.predict(x_plot)[0]
    
    x1_ori = np.linspace(bo.bounds[0,0],bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0],bo.bounds[1,1], 100)
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)

    CS_acq=myGmm.contourf(x1g_ori,x2g_ori,y_plot2.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    myGmm.scatter(dpgmm_means_original[:,0],dpgmm_means_original[:,1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    myGmm.set_title('IGMM Approximation',fontsize=16)
    myGmm.set_xlim(bo.bounds[0,0],bo.bounds[0,1])
    myGmm.set_ylim(bo.bounds[1,0],bo.bounds[1,1])
    myGmm.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_histogram_2d(bo,samples):
    
    # convert samples from 0-1 to original scale
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]
    
    #fig=plt.figure(figsize=(12, 5))
    fig=plt.figure()
    myhist=fig.add_subplot(1,1,1)
    
    myhist.set_title("Histogram of Samples under Acq Func",fontsize=16)
    
    #xedges = np.linspace(myfunction.bounds['x1'][0], myfunction.bounds['x1'][1], 10)
    #yedges = np.linspace(myfunction.bounds['x2'][0], myfunction.bounds['x2'][1], 10)
    
    xedges = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 10)
    yedges = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 10)

    H, xedges, yedges = np.histogram2d(samples_original[:,0], samples_original[:,1], bins=50)   
    
    #data = [go.Histogram2d(x=vu[:,1],y=vu[:,0])]
    #plot_url = py.plot(data, filename='2d-histogram')

    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    myhist.pcolormesh(xedges,yedges,Hmasked)
    myhist.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    myhist.set_ylim(bo.bounds[1,0], bo.bounds[1,1])

def plot_histogram_1d(bo,samples):
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]


    fig=plt.figure(figsize=(8, 3))
    fig.suptitle("Histogram",fontsize=16)
    myplot=fig.add_subplot(111)
    myplot.hist(samples_original,50)
    myplot.set_xlim(bo.bounds[0,0],bo.bounds[0,1])
    
    myplot.set_xlabel("Value",fontsize=16)
    myplot.set_ylabel("Frequency",fontsize=16)
        
def plot_acq_bo_1d(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(10, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(8, 1, height_ratios=[3, 1,1,1,1,1,1,1]) 
    axis = plt.subplot(gs[0])
    acq_UCB = plt.subplot(gs[1])
    acq_EI = plt.subplot(gs[2])
    acq_POI = plt.subplot(gs[3])
    
    #acq_TS2 = plt.subplot(gs[5])
    acq_ES = plt.subplot(gs[4])
    acq_PES = plt.subplot(gs[5])
    acq_MRS = plt.subplot(gs[6])
    
    acq_Consensus = plt.subplot(gs[7])


    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='95% CI')
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})
    
    # UCB 
    acq_func={}
    acq_func['name']='ucb'
    acq_func['kappa']=2
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_UCB.plot(x_original, utility, label='Utility Function', color='purple')
    acq_UCB.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

       
    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    #acq_UCB.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_UCB.set_xlim((np.min(x_original), np.max(x_original)))
    acq_UCB.set_ylabel('UCB', fontdict={'size':16})
    acq_UCB.set_xlabel('x', fontdict={'size':16})
    
    
    # EI 
    acq_func={}
    acq_func['name']='ei'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_EI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_EI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_EI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_EI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_EI.set_ylabel('EI', fontdict={'size':16})
    acq_EI.set_xlabel('x', fontdict={'size':16})

    # POI 
    acq_func={}
    acq_func['name']='poi'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds
    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_POI.plot(x_original, utility, label='Utility Function', color='purple')
    acq_POI.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_POI.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_POI.set_xlim((np.min(x_original), np.max(x_original)))
    acq_POI.set_ylabel('POI', fontdict={'size':16})
    acq_POI.set_xlabel('x', fontdict={'size':16})

    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_EI.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    # MRS     
    acq_func={}
    acq_func['name']='mrs'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_MRS.plot(x_original, utility, label='Utility Function', color='purple')
    acq_MRS.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_MRS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_MRS.set_xlim((np.min(x_original), np.max(x_original)))
    acq_MRS.set_ylabel('MRS', fontdict={'size':16})
    acq_MRS.set_xlabel('x', fontdict={'size':16})
	

    # PES
    acq_func={}
    acq_func['name']='pes'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_PES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_PES.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_PES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_PES.set_xlim((np.min(x_original), np.max(x_original)))
    acq_PES.set_ylabel('PES', fontdict={'size':16})
    acq_PES.set_xlabel('x', fontdict={'size':16})
     
    # TS1   
    
    acq_func={}
    acq_func['name']='consensus'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_Consensus.plot(x_original, utility, label='Utility Function', color='purple')


    temp=np.asarray(myacq.object.xt_suggestions)
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq_Consensus.plot(xt_suggestion_original, [np.max(utility)]*xt_suggestion_original.shape[0], 's', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='red', markeredgecolor='k', markeredgewidth=1)
   
    max_point=np.max(utility)
    
    acq_Consensus.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
        
    #acq_TS.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_Consensus.set_xlim((np.min(x_original), np.max(x_original)))
    #acq_TS.set_ylim((np.min(utility)*0.9, np.max(utility)*1.1))
    acq_Consensus.set_ylabel('Consensus', fontdict={'size':16})
    acq_Consensus.set_xlabel('x', fontdict={'size':16})


    # ES     
    acq_func={}
    acq_func['name']='es'
    acq_func['dim']=1
    acq_func['scalebounds']=bo.scalebounds

    myacq=AcquisitionFunction(acq_func)
	
    utility = myacq.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq_ES.plot(x_original, utility, label='Utility Function', color='purple')
    acq_ES.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)

    max_point=np.max(utility)
    
    #acq_ES.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq_ES.set_xlim((np.min(x_original), np.max(x_original)))
    acq_ES.set_ylabel('ES', fontdict={'size':16})
    acq_ES.set_xlabel('x', fontdict={'size':16})
    
    strFileName="{:d}_GP_acquisition_functions.eps".format(counter)
    fig.savefig(strFileName, bbox_inches='tight')
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq_TS.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

	
def plot_bo_1d(bo):
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    fig=plt.figure(figsize=(8, 5))
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)**2
    
    axis.plot(x_original, y_original, linewidth=3, label='f(x)')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Obs', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    #temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original2=temp_yaxis*np.std(bo.Y_original)+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original2,alpha=.6, fc='c', ec='None', label='GP variance')
    
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    #axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp)
    acq.plot(x_original, utility, label='Acq Function', color='purple')
    acq.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Selection', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    #acq.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         #label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((np.min(x_original), np.max(x_original)))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    acq.set_yticks([])
    axis.set_yticks([])

    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    axis.legend(loc=4,ncol=4,fontsize=14)

    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=4,ncol=2,fontsize=14)
    #plt.legend(fontsize=14)

    strFileName="{:d}_GP_BO_1d.pdf".format(counter)
    strPath=os.path.join(out_dir,strFileName)
    fig.savefig(strPath, bbox_inches='tight')
    
    

def plot_bo_1d_variance(bo):
    
    global counter
    counter=counter+1
    
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    
    #fig=plt.figure(figsize=(8, 5))
    fig, ax1 = plt.subplots(figsize=(8.5, 4))

    mu, sigma = bo.posterior(x)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma*(np.max(bo.Y_original)-np.min(bo.Y_original))

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp)


    def distance_function(x,X):            
        Euc_dist=euclidean_distances(x,X)
          
        dist=Euc_dist.min(axis=1)
        #return np.log(dist+1e-6)
        return dist

        
    utility_distance=distance_function(x.reshape((-1, 1)),bo.X)
    idxMaxVar=np.argmax(utility)
    #idxMaxVar=[idx for idx,val in enumerate(utility) if val>=0.995]
    ax1.plot(x_original, utility, label='GP $\sigma(x)$', color='purple')  

    
    ax1.scatter(x_original[idxMaxVar], utility[idxMaxVar], marker='s',label='x=argmax $\sigma(x)$', color='blue',linewidth=2)            
          
    #ax1.scatter(x_original[idxMaxVar], utility[idxMaxVar], label='$||x-[x]||$', color='blue',linewidth=2)            

    ax1.plot(bo.X_original.flatten(), [0]*len(bo.X_original.flatten()), 'D', markersize=10, label=u'Observations', color='r')


    idxMaxDE=np.argmax(utility_distance)
    ax2 = ax1.twinx()
    ax2.plot(x_original, utility_distance, label='$d(x)=||x-[x]||^2$', color='black') 
    ax2.plot(x_original[idxMaxDE], utility_distance[idxMaxDE], 'o',label='x=argmax d(x)', color='black',markersize=10)            
           
   #ax2.set_ylim((0, 0.45))


         
    ax1.set_xlim((np.min(x_original)-0.01, 0.01+np.max(x_original)))
    #ax1.set_ylim((-0.02, np.max(utility) + 0.05))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    ax1.set_ylabel('\sigma(x)', fontdict={'size':18})
    ax2.set_ylabel('d(x)', fontdict={'size':18})

    ax1.set_xlabel('x', fontdict={'size':18})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #ax1.legend(loc=2, bbox_to_anchor=(1.1, 1), borderaxespad=0.,fontsize=14)
    #ax2.legend(loc=2, bbox_to_anchor=(1.1, 0.3), borderaxespad=0.,fontsize=14)

    plt.title('Exploration by GP variance vs distance',fontsize=22)
    ax1.legend(loc=3, bbox_to_anchor=(0.05,-0.32,1, -0.32), borderaxespad=0.,fontsize=14,ncol=4)
    ax2.legend(loc=3, bbox_to_anchor=(0.05,-0.46,1, -0.46), borderaxespad=0.,fontsize=14,ncol=2)

    #plt.legend(fontsize=14)

    strFileName="{:d}_var_DE.eps".format(counter)
    strPath=os.path.join(out_dir,strFileName)
    #fig.savefig(strPath, bbox_inches='tight')
    
    
def plot_bo_2d(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
  
    fig = plt.figure()
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d = fig.add_subplot(1, 1, 1)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=140,label='Selected')
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')

    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq2d.legend(loc='center left',ncol=3,bbox_to_anchor=(0, -0.2))
      
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


    
def plot_bo_2d_withGPmeans(bo,myxlabel="",myylabel=""):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 50)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 50)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 50)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 50)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(9, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 1, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    #acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    mu_original=np.clip(mu_original,0.8*np.min(bo.Y_original),1.2*np.max(bo.Y_original))
    
    idxMax=np.argmax(mu)
    CS=axis2d.contourf(x1g_ori,x2g_ori,mu_original.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Data', color='g')    
    
    labelStr="Best Inferred X1={:.2f}, X2={:.1f}".format(X_ori[idxMax,0],X_ori[idxMax,1])
    axis2d.scatter(X_ori[idxMax,0],X_ori[idxMax,1],color='r',s=70,label=labelStr)

    #axis2d.set_title('Gaussian Process Mean No Ability',fontsize=16)
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_xlabel(myxlabel,fontsize=14)
    axis2d.set_ylabel(myylabel,fontsize=14)
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    lgd=axis2d.legend(loc=2, bbox_to_anchor=(0, -0.15), borderaxespad=0.,ncol=2,fontsize=14)

    fig.colorbar(CS, ax=axis2d, shrink=0.9)
    strOutput="GPmean_plot_{}_{}.pdf".format(myxlabel,myylabel)
    fig.savefig(strOutput,                bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    return X_ori, mu_original

    #plt.colorbar(ax=axis2d)

    #axis.plot(x, mu, '--', color='k', label='Prediction')
    
    
    #axis.set_xlim((np.min(x), np.max(x)))
    #axis.set_ylim((None, None))
    #axis.set_ylabel('f(x)', fontdict={'size':16})
    #axis.set_xlabel('x', fontdict={'size':16})
    
    # plot the acquisition function
"""
    utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60,label='Suggested Exp')
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60,label='Best Found Exp')
    
    
    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
    acq2d.set_xlabel('Current')
    acq2d.set_ylabel('Frequency')
    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    axis2d.legend(loc=2, bbox_to_anchor=(0.01, -0.15), borderaxespad=0.,ncol=2)
    acq2d.legend(loc=2, bbox_to_anchor=(0.01, -0.15), borderaxespad=0.,ncol=2)
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)
"""


def plot_bo_2d_condition(bo,ability_org,newinput=[],flagOutFile=False):
    # plot the 2D GPmeans given the ability (third dimension)
    ability=(ability_org-bo.bounds[-1,0])/(bo.bounds[-1,1]-bo.bounds[-1,0])
    
    grid_sz=70
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], grid_sz)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], grid_sz)
    x3= np.asarray([ability]*(grid_sz*grid_sz))
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten(),x3.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], grid_sz)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], grid_sz)
    x3_ori= np.asarray([ability_org]*(grid_sz*grid_sz))
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten(),x3_ori.flatten()]
    
    mu, sigma = bo.posterior(X)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    mu_original=np.clip(mu_original,0.8*np.min(bo.Y_original),1.2*np.max(bo.Y_original))
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    
    """
    fig = plt.figure(figsize=(12, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 3, 1)
    var2d = fig.add_subplot(1, 3, 2)

    acq2d = fig.add_subplot(1, 3, 3)
    

    #sigma_original=sigma


    idxMax=np.argmax(mu)
    CS_mean=axis2d.contourf(x1g_ori,x2g_ori,mu_original.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], s=40,label=u'Data', color='g')    
    
    labelStr="Best Inferred C={:.2f}, F={:.1f}".format(X_ori[idxMax,0],X_ori[idxMax,1])
    axis2d.scatter(X_ori[idxMax,0],X_ori[idxMax,1],marker='s',color='r',s=50,label=labelStr)
    
    if newinput != []:
        labelStr="Selected C={:.2f}, F={:.1f}".format(newinput[0],newinput[1])
        axis2d.scatter(newinput[0],newinput[1],marker='s',color='b',s=50,label=labelStr)

    strTitle="GP Mean given Ability={}".format(ability_org)
    axis2d.set_title(strTitle,fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_xlabel('Current',fontsize=14)
    axis2d.set_ylabel('Frequency',fontsize=14)
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    lgd=axis2d.legend(loc=2, bbox_to_anchor=(0, -0.15), borderaxespad=0.,ncol=2,fontsize=14)

    fig.colorbar(CS_mean, ax=axis2d, shrink=0.9)
    
    # variance

    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_var=var2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(sigma)

    
    var2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    var2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    var2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    var2d.set_title('GP Variance',fontsize=16)
    fig.colorbar(CS_var, ax=var2d, shrink=0.9)

    
    # acquisition function
    
    
    
    utility = bo.acq_func.acq_kind(X, bo.gp)
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=140,label='Selected')
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],marker='s',color='r',s=30,label='Peak')

    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)
    """
    
    if flagOutFile==True:
        strOutput="GPmean_plot_ability_{}.pdf".format(ability_org)
        fig.savefig(strOutput,                bbox_extra_artists=(lgd,), bbox_inches='tight')

    return X_ori, mu_original,sigma_original

def plot_bo_2d_condition_TS(bo,ability_org,newinput=[],flagOutFile=True):
    # plot the 2D GPmeans given the ability (third dimension)
    ability=(ability_org-bo.bounds[-1,0])/(bo.bounds[-1,1]-bo.bounds[-1,0])
    
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 70)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 70)
    x3= np.asarray([ability]*4900)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten(),x3.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 70)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 70)
    x3_ori= np.asarray([ability_org]*4900)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten(),x3_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 3, 1)
    var2d = fig.add_subplot(1, 3, 2)

    acq2d = fig.add_subplot(1, 3, 3)
    
    mu, sigma = bo.posterior(X)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    mu_original=np.clip(mu_original,0.8*np.min(bo.Y_original),1.2*np.max(bo.Y_original))
    #sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)
    sigma_original=sigma

    idxMax=np.argmax(mu)
    CS_mean=axis2d.contourf(x1g_ori,x2g_ori,mu_original.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], s=40,label=u'Data', color='g')    
    
    labelStr="Best Inferred C={:.2f}, F={:.1f}".format(X_ori[idxMax,0],X_ori[idxMax,1])
    axis2d.scatter(X_ori[idxMax,0],X_ori[idxMax,1],marker='s',color='r',s=50,label=labelStr)
    
    if newinput != []:
        labelStr="Selected C={:.2f}, F={:.1f}".format(newinput[0],newinput[1])
        axis2d.scatter(newinput[0],newinput[1],marker='s',color='b',s=50,label=labelStr)

    strTitle="GP Mean given Ability={:.3f}".format(ability_org)
    axis2d.set_title(strTitle,fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_xlabel('Current',fontsize=14)
    axis2d.set_ylabel('Frequency',fontsize=14)
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    lgd=axis2d.legend(loc=2, bbox_to_anchor=(0, -0.15), borderaxespad=0.,ncol=2,fontsize=14)

    fig.colorbar(CS_mean, ax=axis2d, shrink=0.9)
    
    # variance

    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_var=var2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(sigma)

    
    var2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    var2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    var2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    var2d.set_title('GP Variance',fontsize=16)
    fig.colorbar(CS_var, ax=var2d, shrink=0.9)

    
    # acquisition function
    # ==================================================================================	
    # finding the xt of Thompson Sampling then use for PES, ES and VRS
    xstars=[]
    y_stars=[]
    xstars_VRS=[]
    numXtar=25*bo.dim
    for ii in range(numXtar):
        mu_acq={}
        mu_acq['name']='thompson'
        mu_acq['dim']=bo.dim
        mu_acq['scalebounds']=bo.scalebounds    
        acq_mu=AcquisitionFunction(mu_acq)
        xt_TS = acq_max(ac=acq_mu.acq_kind,gp=bo.gp,bounds=bo.scalebounds ,opt_toolbox='scipy')
        
        #y_xt_TS=acq_mu.acq_kind(xt_TS,bo.gp)
        #if y_xt_TS>mu_max:
        #y_stars.append(y_xt_TS)

        xstars.append(xt_TS)
        #if y_xt_TS>=y_max:
        xstars_VRS.append(xt_TS)
    
    
    temp=np.asarray(xstars_VRS) 
    # convert from scale data points to original data points
    xt_suggestion_original=temp*bo.max_min_gap+bo.bounds[:,0]
    
    acq2d.scatter(xt_suggestion_original[:,0],xt_suggestion_original[:,1],
              marker='*',color='y',s=150,label='xstars')

    acq2d.set_title('GP Samples',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    #fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    
    if flagOutFile==True:
        strOutput="TS_plot_ability_{:.3f}.pdf".format(ability_org)
        fig.savefig(strOutput,                bbox_extra_artists=(lgd,), bbox_inches='tight')

    return X_ori, mu_original,sigma_original



def collect_3d_grid_GPmean(bo,newinput=[]):
    # plot the 2D GPmeans given the ability (third dimension)   
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 38)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 38)
    x3= np.linspace(bo.scalebounds[2,0], bo.scalebounds[2,1], 38)
    
    x1g,x2g,x3g=np.meshgrid(x1,x2,x3)
    
    X=np.c_[x1g.flatten(), x2g.flatten(),x3g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 38)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 38)
    x3_ori = np.linspace(bo.bounds[2,0], bo.bounds[2,1], 38)
    
    x1g_ori,x2g_ori,x3g_ori=np.meshgrid(x1_ori,x2_ori,x3_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten(),x3g_ori.flatten()]
    
    mu, sigma = bo.posterior(X)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)


    return X_ori, mu_original

def collect_3d_grid_GPvar(bo,newinput=[]):
    # plot the 2D GPmeans given the ability (third dimension)   
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 38)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 38)
    x3= np.linspace(bo.scalebounds[2,0], bo.scalebounds[2,1], 38)
    
    x1g,x2g,x3g=np.meshgrid(x1,x2,x3)
    
    X=np.c_[x1g.flatten(), x2g.flatten(),x3g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 38)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 38)
    x3_ori = np.linspace(bo.bounds[2,0], bo.bounds[2,1], 38)
    
    x1g_ori,x2g_ori,x3g_ori=np.meshgrid(x1_ori,x2_ori,x3_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten(),x3g_ori.flatten()]
    
    mu, sigma = bo.posterior(X)
    sigma_original=sigma*np.std(bo.Y_original)+np.mean(bo.Y_original)


    return X_ori, sigma_original

def plot_bo_2d_withGPmeans_condition(bo,ability_org,newinput=[]):
    # plot the 2D GPmeans given the ability (third dimension)
    ability=(ability_org-bo.bounds[-1,0])/(bo.bounds[-1,1]-bo.bounds[-1,0])
    
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 50)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 50)
    x3= np.asarray([ability]*2500)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten(),x3.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 50)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 50)
    x3_ori= np.asarray([ability_org]*2500)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten(),x3_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(9, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 1, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    
    mu, sigma = bo.posterior(X)
    mu_original=mu*np.std(bo.Y_original)+np.mean(bo.Y_original)

    mu_original=np.clip(mu_original,0.8*np.min(bo.Y_original),1.2*np.max(bo.Y_original))

    idxMax=np.argmax(mu)
    CS=axis2d.contourf(x1g_ori,x2g_ori,mu_original.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], s=40,label=u'Data', color='g')    
    
    labelStr="Best Inferred C={:.2f}, F={:.1f}".format(X_ori[idxMax,0],X_ori[idxMax,1])
    axis2d.scatter(X_ori[idxMax,0],X_ori[idxMax,1],marker='s',color='r',s=50,label=labelStr)
    
    #if newinput != []:
        #labelStr="Selected C={:.2f}, F={:.1f}".format(newinput[0],newinput[1])
        #axis2d.scatter(newinput[0],newinput[1],marker='s',color='b',s=50,label=labelStr)


    strTitle="GP Mean given Ability={}".format(ability_org)
    axis2d.set_title(strTitle,fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_xlabel('Current',fontsize=14)
    axis2d.set_ylabel('Frequency',fontsize=14)
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    lgd=axis2d.legend(loc=2, bbox_to_anchor=(0, -0.15), borderaxespad=0.,ncol=2,fontsize=14)

    fig.colorbar(CS, ax=axis2d, shrink=0.9)
    strOutput="GPmean_plot_ability_{}.pdf".format(ability_org)
    fig.savefig(strOutput,                bbox_extra_artists=(lgd,), bbox_inches='tight')

    return X_ori, mu_original
    
    
def plot_bo_2d_withGPmeans_Sigma(bo,myxlabel="",myylabel="",saveflag=""):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 60)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 60)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 60)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 60)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #fig = plt.figure(figsize=(12, 3))
    fig = plt.figure(figsize=(6, 5))

    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 1, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    #acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    utility = bo.acq_func.acq_kind(X, bo.gp)

    #CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    #axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')   
    axis2d.scatter(bo.X_original[:,0],bo.T_original[:,0], label=u'Observations', color='g')   
    
    mu=mu.reshape(x1g.shape)
    idxMax=np.argmax(mu,axis=1)

    axis2d.scatter(x1_ori[idxMax],x2_ori, label=u'Max by Y-axis', color='r',marker='x',s=25)    



    axis2d.set_title('GP Predictive Mean',fontsize=16)
    axis2d.set_xlabel(myxlabel,fontsize=14)
    axis2d.set_ylabel(myylabel,fontsize=14)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    axis2d.legend()

    fig.colorbar(CS, ax=axis2d, shrink=0.9)
    fig.savefig(saveflag+"_gp_mean.png",dpi=600)

    fig = plt.figure(figsize=(6, 5))
    acq2d = fig.add_subplot(1, 1, 1)

    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    #acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    acq2d.scatter(bo.X_original[:,0],bo.T_original[:,0],color='g')  
    
        
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    #acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    acq2d.set_title('GP Predictive Variance',fontsize=16)
    acq2d.set_xlabel(myxlabel,fontsize=14)
    acq2d.set_ylabel(myylabel,fontsize=14)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    #acq2d.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    #acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
             
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)
    fig.savefig(saveflag+"_gp_var.png",dpi=600)


def plot_2d_Acq_by_Personalisedscore(bo,myxlabel="",myylabel="",saveflag=""):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 60)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 60)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 60)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 60)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    #fig = plt.figure(figsize=(12, 3))
    fig = plt.figure(figsize=(6, 5))

    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 1, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    #acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    
    
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    utility = bo.acq_func.acq_kind(X, bo.gp)
    
    utility=utility.reshape(x1g.shape)
    
    idxMax=np.argmax(utility,axis=1)

    #CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS=axis2d.contourf(x1g_ori,x2g_ori,utility,cmap=my_cmap,origin='lower')
    #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    #axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Obs', color='g')    
    axis2d.scatter(bo.X_original[:,0],bo.T_original[:,0], label=u'Obs', color='g')    
    
    
    axis2d.scatter(x1_ori[idxMax],x2_ori, label=u'Max by Y-axis', color='r',marker='x',s=25)    
    
    axis2d.set_title('GP UCB',fontsize=16)
    axis2d.set_xlabel(myxlabel,fontsize=14)
    axis2d.set_ylabel(myylabel,fontsize=14)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    axis2d.legend()
    fig.colorbar(CS, ax=axis2d, shrink=0.9)
    fig.savefig(saveflag+"_gp_acq.png",dpi=600)


    
def show_optimization_progress(bo):
        

    try: # batch
        bo.NumPoints
        fig=plt.figure(figsize=(6, 3))
        myYbest=[bo.Y_original[:int(sum(bo.NumPoints[:idx+1]))].max() 
        for idx in range(len(bo.NumPoints))]
        plt.plot(range(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
        plt.title('Best found value',fontsize=18)
        plt.xlabel('Iteration',fontsize=14)
        plt.ylabel('f(x)',fontsize=14)
        fig.savefig("P://Performance_Curve.pdf")
    
    except: # sequential
        fig=plt.figure(figsize=(6, 3))
        myYbest=[bo.Y_original[:idx+1].max() for idx,val in enumerate(bo.Y_original)]
        plt.plot(range(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
        plt.title('Best found value',fontsize=18)
        plt.xlabel('Iteration',fontsize=14)
        plt.ylabel('f(x)',fontsize=14)
        plt.tight_layout()
        

        fig.savefig("P://Performance_Curve.pdf")
        
        import pandas as pd
        strPath="P://BestFoundValue.csv"
        pd.DataFrame(np.asarray(myYbest)).to_csv(strPath,index=False,header=False)