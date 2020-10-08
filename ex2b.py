import numpy as np
from scipy.optimize import bisect
from scipy.stats import entropy
import matplotlib.pyplot as plt
import scipy.stats as ss
import math




def median(arr,S):
   
    ar=[arr[i] for i in S]
    ar.sort()
    l=len(ar)
    n=int(math.floor(l/2))
    return(ar[n])
    
    
def med_elim(K):    
    Arm=[i for i in range(K)]
    mean_gain=dict()
    mean_gain[0]=0.75
    
    for i in range(2,K+1):
        mean_gain[i-1]=0.75-1.0*i/40
    print "Running for K=",K
    print "Mean Gain of each arm: ",mean_gain    
    npaths=50
    delta=0.1
    epsilon=0.01
    best_arm=[]
    sample_com=[]
    for s in range(npaths):
        print "Running for Sample path: ",s
        S_l=Arm
        S_l2=[]
        l=1
        epsilon1=0.25*epsilon
        delta1=0.5*delta
        est_gain=[0 for i in range(K)]
        cum_reward=[0 for i in range(K)]
        counter=[0 for i in range(K)]
        n=0
        while len(S_l)>1:
           S_l2=[]
           t=int(math.ceil(1/((epsilon1/2)**2)*math.log(3.0/delta1))/10.0)
           n=n+len(S_l)*t
           for k in S_l:
               print "No. of samplings:",t
               for i in range(t):
                   reward=np.random.binomial(1,mean_gain[k])
                   cum_reward[k]=cum_reward[k]+reward
                   counter[k]=counter[k]+1
               est_gain[k]=1.0*cum_reward[k]/counter[k]
           med_l=median(est_gain,S_l)
         
           
           for k in S_l:
               if est_gain[k]>=med_l:
                   S_l2.append(k)
                
           S_l=S_l2
           epsilon1=0.75*epsilon1
           delta1=0.5*delta1
           l=l+1
        best_arm.append(S_l[0])
        sample_com.append(n)
    
    return([best_arm,sample_com])

def plot_med_elim():
    K=[5,10,15,20,25]
    mistake=[0 for k in range(len(K))]
    samp_com=[0 for i in range(len(K))]
    w=[]
    for k in K:
        w.append(med_elim(k))
    for w1 in w:    
        m=0
        c=0
        for i in w1[0]:
            if i!=0:
               m=m+1
               
        mistake[c]=m
        c=c+1
    sample_mean = []
    w3=[w[i][1] for i in range(len(w))]
    sample_err = []
    freedom_degree = len(w) - 2
    for com in w3:
        sample_mean.append(np.mean(com))
        sample_err.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(com))
    
    colors = list("rgbcmyk")
    shape = ['--^', '--d', '--v']
    plt.errorbar(K, sample_mean, sample_err, color=colors[0])
    plt.plot(K, sample_mean, colors[0] + shape[0])
    plt.xlabel("Number of arms")
    plt.ylabel("Sample Complexity")
    plt.title("Sample Complexity vs Number of arms for Median Elimination method ")
    plt.savefig("output2_b1.png")

    plt.figure()    
    plt.plot(K,mistake)
    plt.xlabel("Number of arms")
    plt.ylabel("Mistake bound")
    plt.title("Mistake Bound vs Number of arms for Median Elimination method ")
    plt.savefig("output2_b2.png")

    
plot_med_elim()           
           
        