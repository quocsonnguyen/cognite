import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bayes_opt import auxiliary_functions
from bayes_opt.utility import export_results
from sklearn.metrics.pairwise import euclidean_distances

from bayes_opt.test_functions import functions



xtrue=np.asarray([[0.10188981, 0.55132284, 0.85717205]])

def get_distance_to_optimum(mybo,xtrue):
    
    dist_to_opt=[0]*len(mybo)
    for idx in range(len(mybo)):
        np.random.seed(idx)
        xselected=mybo[idx].X_original
        if xselected.shape[1]==2:
            rand_vec=np.random.uniform(0,1,size=(xselected.shape[0],1))       
            xselected=np.hstack((xselected,rand_vec))
            
        Euc_dist=euclidean_distances(xselected,xtrue)
        Euc_dist_shorted=[np.min(Euc_dist[:ii+1]) for ii in range(len(xselected))]
        #dist_to_opt[idx]=Euc_dist_shorted[38:]
        dist_to_opt[idx]=Euc_dist_shorted[:61]

        #print(np.min(Euc_dist_shorted))

    return dist_to_opt

sns.set(style="ticks")


fig=plt.figure(figsize=(10, 6))

xmin=[0.114614,0.555649,0.852547]
##############
function_name='hartman_3d'
D=3
#optimal_value=-3.82
optimal_value=0

BatchSz0=D*3

start_point=BatchSz0
step=1
mylinewidth=2.5
alpha_level=0.3
std_scale=0.5

T=20

BatchSz=[1]*(D*T+1)
BatchSz[0]=3*D

x_axis=np.array(range(0,D*T+1))
x_axis=x_axis[::step]

# is minimization problem
IsMin=-1

IsLog=0


#std_level="std_1_cond"
std_level="std_0_cond"



#Rand
strFile="pickle_storage/{:s}/{:s}_{:d}_random_gp.pickle".format(std_level,function_name,D)

with open(strFile,"rb") as f:
    Rand = pickle.load(f,encoding='bytes')

myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(Rand[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)

myYbest=IsMin*np.asarray(myYbest)*1
myYbestRand=myYbest-optimal_value

if IsLog==1:
    myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStdRand=myStd*std_scale
plt.errorbar(x_axis,myYbestRand,yerr=myStdRand,linewidth=mylinewidth,color='k',linestyle='-.',marker='o',label='Random Search')




# BO  
strFile="pickle_storage/{:s}/{:s}_{:d}_ucb_gp_nocond.pickle".format(std_level,function_name,D)
with open(strFile, 'rb') as f:
    BO = pickle.load(f,encoding='bytes')
    
myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(BO[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)

myYbest=IsMin*np.asarray(myYbest)
myYbestUCB=myYbest-optimal_value
myYbestUCB[0]=myYbestRand[0]

if IsLog==1:
    myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStdUCB=myStd*std_scale
plt.errorbar(x_axis,myYbestUCB,yerr=myStdUCB,linewidth=mylinewidth,color='m',linestyle=':',marker='v', label='Bayes Opt')




# pBO
strFile="pickle_storage/{:s}/{:s}_{:d}_ucb_gp_cond.pickle".format(std_level,function_name,D)
with open(strFile,'rb') as f:
    pBO = pickle.load(f,encoding='bytes')
    

myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(pBO[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)
myYbest=IsMin*np.asarray(myYbest)


myYbestpBO=myYbest-optimal_value

if IsLog==1:
    myYbestpBO=np.log(myYbestpBO)
    myStd=np.log(myStd)
myStdpBO=myStd*std_scale

plt.errorbar(x_axis,myYbestpBO,yerr=myStdpBO,linewidth=mylinewidth,color='r',linestyle='-',marker='h', label='Personalized Bayes Opt')




#import pandas as pd
#strPath="P://Hartmann3d_{:s}.csv".format(std_level)
#mydata=np.vstack((myYbestRand,myStdRand,myYbestUCB,myStdUCB,myYbestpBO,myStdpBO))
#pd.DataFrame(np.asarray(mydata)).to_csv(strPath,index=False,header=False)



plt.xlabel('Iteration',fontdict={'size':20})



if IsLog==0:
    plt.ylabel('Best found value',fontdict={'size':20})
else:
    plt.ylabel('Log of Best Found Value',fontdict={'size':20})
    
plt.xlabel('Iteration',fontdict={'size':20})

plt.legend(loc='middle right',prop={'size':20},ncol=1)


plt.xlim([-1,T*D+1])

strTitle="Hartmann D={:d} $\sigma^2_n$={}".format(D,1*4)

plt.title(strTitle,fontdict={'size':24})

plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)

strFile="fig/{:s}_{:d}_{:s}.pdf".format(function_name,D,std_level)
plt.savefig(strFile, bbox_inches='tight')





std_scale=0.5


rand_dist=get_distance_to_optimum(Rand[2],xtrue)
ucb_nocond_dist=get_distance_to_optimum(BO[2],xtrue)
ucb_cond_dist=get_distance_to_optimum(pBO[2],xtrue)

fig=plt.figure(figsize=(10, 6))
x_axis=np.array(range(0,61))

plt.errorbar(x_axis,np.mean(rand_dist,axis=0),yerr=np.std(rand_dist,axis=0)*std_scale,linewidth=mylinewidth,color='k',linestyle='-.',marker='o',label='Random Search')
plt.errorbar(x_axis,np.mean(ucb_nocond_dist,axis=0),yerr=np.std(ucb_nocond_dist,axis=0)*std_scale,linewidth=mylinewidth,color='m',linestyle=':',marker='v', label='Bayes Opt')
plt.errorbar(x_axis,np.mean(ucb_cond_dist,axis=0),yerr=np.std(ucb_cond_dist,axis=0)*std_scale,linewidth=mylinewidth,color='r',linestyle='-',marker='h', label='Personalized Bayes Opt')
plt.legend(loc='middle right',prop={'size':18},ncol=1)
plt.ylabel("Euclidean dist to true parameter",fontdict={'size':20})
plt.xlabel('Iteration',fontdict={'size':20})
plt.title(strTitle,fontdict={'size':24})

plt.ylim([0,0.48])


strFile="fig/{:s}_{:d}_{:s}_dist.pdf".format(function_name,D,std_level)
plt.savefig(strFile, bbox_inches='tight')



#import pandas as pd
#strPath="P://Hartmann3d_{:s}_dist.csv".format(std_level)
#mydata=np.vstack((np.mean(rand_dist,axis=0),np.std(rand_dist,axis=0)*std_scale,np.mean(ucb_nocond_dist,axis=0),\
#                  np.std(ucb_nocond_dist,axis=0)*std_scale,np.mean(ucb_cond_dist,axis=0),np.std(ucb_cond_dist,axis=0)*std_scale))
#pd.DataFrame(np.asarray(mydata)).to_csv(strPath,index=False,header=False)

