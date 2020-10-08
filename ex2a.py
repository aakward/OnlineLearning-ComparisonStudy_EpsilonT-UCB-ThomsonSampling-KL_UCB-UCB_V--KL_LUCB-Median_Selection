import numpy as np
from scipy.optimize import bisect
from scipy.stats import entropy
import matplotlib.pyplot as plt
import scipy.stats as ss
import math


cum_regret=[]
def KL_LUCB(S,K):
    arm=[i for i in range(K)]
    mean_gain=dict()
    mean_gain[0]=0.75

    for i in range(2,K+1):
        mean_gain[i-1]=0.75-1.0*i/40
    print "Mean Gain of each arm: ",mean_gain   
    def check_condition(bound, arm_expected_reward, sample_count, log_val):
        p = [arm_expected_reward, 1 - arm_expected_reward]
        q = [bound, 1 - bound]
        return sample_count * entropy(p, qk=q) - log_val
    
    
    
    def compute_q1(arm_expected_reward, sample_count, log_val):
        try:
            upper_bound_val = bisect(check_condition, arm_expected_reward, 1, args=(arm_expected_reward, sample_count, log_val))
        except:
            upper_bound_val = 1
        return upper_bound_val
    def compute_q2(arm_expected_reward, sample_count, log_val):
                try:
                    lower_bound_val = bisect(check_condition, 0,arm_expected_reward, args=(arm_expected_reward, sample_count, log_val))
                except:
                    lower_bound_val = 0
                return lower_bound_val
    epsilon=0
    delta=0.1
    alpha=2
    kappa=4*np.exp(1)+4
    best_arm=[]    
    sample_com=[]
    for s in range(S):
        print "No. of samples:",s       
        est_gain=[0 for i in range(K)]
        cum_reward_t=[0 for i in range(K)]
        counter=dict()
        t=1
        for i in range(K):
            counter[i]=0
        U_t=[0 for i in range(K)]
        B_t=1
        L_t=[0 for i in range(K)]
        for i in range(K):
            reward_t=np.random.binomial(1,mean_gain[i])
            cum_reward_t[i]= cum_reward_t[i]+reward_t
            counter[i]=counter[i]+1
            est_gain[i]=1.0*cum_reward_t[i]/counter[i]
            c=math.log(kappa*K*(t**alpha)/delta) 
            c=c+math.log(c)
            U_t[i]=compute_q1(est_gain[i],counter[i],c)
            L_t[i]=compute_q2(est_gain[i],counter[i],c)
        while B_t>epsilon:
            choice=[]
            l_t=est_gain.index(max(est_gain))
            for i in range(K):
                if i!=l_t:
                    choice.append([i,U_t[i]])
                    
            u_dash_t=[choice[d][1] for d in range(len(choice))]
            u_t=choice[u_dash_t.index(max(u_dash_t))][0]
            reward_ut=np.random.binomial(1,mean_gain[u_t])
            cum_reward_t[u_t]= cum_reward_t[u_t]+reward_ut
            counter[u_t]=counter[u_t]+1
            est_gain[u_t]=1.0*cum_reward_t[u_t]/counter[u_t]
            reward_lt=np.random.binomial(1,mean_gain[l_t])
            cum_reward_t[l_t]= cum_reward_t[l_t]+reward_lt
            counter[l_t]=counter[l_t]+1
            est_gain[l_t]=1.0*cum_reward_t[l_t]/counter[l_t]
            t=t+1
            c=math.log(kappa*K*(t**alpha)/delta) 
            c=c+math.log(c)
            for i in range(K):
                U_t[i]=compute_q1(est_gain[i],counter[i],c)
                L_t[i]=compute_q2(est_gain[i],counter[i],c)
            choice=[]
            l_t=est_gain.index(max(est_gain))
            for i in range(K):
                if i!=l_t:
                    choice.append([i,U_t[i]])
                    
            u_dash_t=[choice[d][1] for d in range(len(choice))]
            u_t=choice[u_dash_t.index(max(u_dash_t))][0]    
                
            B_t=U_t[u_t]-L_t[l_t]
        best_arm.append(l_t)
        sample_com.append(2*(t-1)+K)
    return([best_arm,sample_com]) 
    
    
def plot_KL_LUCB():
    K=[5,10,15,20,25]
    mistake=[0 for k in range(len(K))]
    samp_com=[0 for i in range(len(K))]
    w=[]
    for k in K:
        w.append(KL_LUCB(50,k))
    for w1 in w:    
        m=0
        c=0
        for i in w1[0]:
            if i!=0:
               m=m+1
        c=c+1       
        mistake[c]=m
    print "Mistake: ",mistake     
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
    plt.xlabel("Number of arms we began with ->")
    plt.ylabel("Confidence Interval for number of sample complexity ->")
    plt.title("Sample Complexity Versus Number of arms chosen for KL-LUCB")
    plt.figure()    
    plt.plot(K,mistake)
    plt.xlabel("Number of arms we began with ->")
    plt.ylabel("Mistake bound for KL-LUCB ->")
    plt.title("Mistake Bound Versus Number of arms")
    plt.show()
        
plot_KL_LUCB()            
            