import scipy.stats as ss
from scipy.optimize import bisect
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt





def check_condition(bound, arm_expected_reward, sample_count, log_val):
    p = [arm_expected_reward, 1 - arm_expected_reward]
    q = [bound, 1 - bound]
    return sample_count * entropy(p, qk=q) - log_val


# Computing Upper and Lower bound
def compute_q(arm_expected_reward, sample_count, log_val):
    try:
        upper_bound_val = bisect(check_condition, arm_expected_reward, 1, args=(arm_expected_reward, sample_count, log_val))
    except:
        upper_bound_val = 1
    return upper_bound_val
    
    
colors = list("rgbcmyk")
shape = ['--^', '--d', '--v','*','-']



K=input("Enter the total no. of arms: ")
mean_gain_param=[0 for i in range(K)]
mean_gain_param[0]=0.5
arm=[i for i in range(K)]
for i in range(1,K):
    mean_gain_param[i]=0.5-((i+1)/70.0)
#print "Mean Gain Parameters: ",mean_gain_param
T=input("Enter the no. of rounds: ")
npath=input("Enter the total no. of sample paths: ")
def epsilon_greedy():
    cum_regret_t=[]
    print "Epsilon Greedy running!"
    for path in range(npath):
        print "Path No.: ",path
        cum_regret=[0.0 for i in range(T)]
        est_gain=[0.0 for i in range(K)]
        set_chosen_arm=[[] for i in range(K)]
        p_t=[1.0/K for i in range(K)]
        for t in range(T):
            #print "t=",t
            epsilon=1.0/(t+1)
            r=np.random.choice(arm,p=p_t)
            #print "Arm chosen: ",r+1
            if t==0:
                cum_regret[t]=mean_gain_param[0]-mean_gain_param[r]
                #print "r=",r
                #print "###########################",mean_gain_param[0],mean_gain_param[r],cum_regret[t]
            else:
                cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[r])
                #print "r=",r                
                #print "###########################",mean_gain_param[0],mean_gain_param[r],cum_regret[t]
            #print "cumulative regret for round",t,"is ",cum_regret[t]
            gain=np.random.binomial(1,mean_gain_param[r])
            #print "Gain by the chosen arm: ",gain
            set_chosen_arm[r].append(gain)            
            for i in range(K):
                if(len(set_chosen_arm[i])>0):
                    est_gain[i]=1.0*np.sum(set_chosen_arm[i])/len(set_chosen_arm[i])
                else:
                    est_gain[i]=0.0
            #print "Estimated Gains: ",est_gain
            max_est_gain=max(est_gain)
            best_est_arm=est_gain.index(max_est_gain)
            #print "Best estimated arm: ",best_est_arm
            for i in range(K):
                if(i==best_est_arm):
                    p_t[i]=1.0-epsilon+float(epsilon/K)
                else:
                    p_t[i]=float(epsilon/K)
            #print "Updated Probabiliies: ",p_t
        cum_regret_t.append(cum_regret)
    
    regret_mean = []
    regret_err = []
    time_epoch=[i for i in range(T)]
    cum_regret_tr=[[0 for i in range(npath)]for j in range(T)]    
    for i in range(T):
        for j in range(npath):
            cum_regret_tr[i][j]=cum_regret_t[j][i]
    freedom_degree = len(cum_regret_tr[0]) - 2
    for regret in cum_regret_tr:
        regret_mean.append(np.mean(regret))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) *ss.sem(regret))
#    colors = list("rgbcmyk")
#    shape = ['--^', '--d', '--v']
    plt.figure()
    plt.errorbar(time_epoch, regret_mean, regret_err, color=colors[0])
    plt.plot(time_epoch, regret_mean, colors[0] + shape[0], label='Epsilon-t Greedy')

def UCB():
    cum_regret_t=[]
    print "UCB Running!"
    alpha=1.5
    for path in range(npath):
        print "Path No.: ",path
        cum_regret=[0.0 for i in range(T)]
        est_gain=[0.0 for i in range(K)]
        set_chosen_arm=[[] for i in range(K)]
        p_t=[1.0/K for i in range(K)]
        decide=[0 for i in range(K)]
        for t in range(K):
            if t==0:
                cum_regret[t]=mean_gain_param[0]-mean_gain_param[t]
                #print "r=",r
                #print "###########################",mean_gain_param[0],mean_gain_param[r],cum_regret[t]
            else:
                cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[t])
            gain=np.random.binomial(1,mean_gain_param[t])
            set_chosen_arm[t].append(gain)            
            for i in range(K):
                if(len(set_chosen_arm[i])>0):
                    est_gain[i]=1.0*np.sum(set_chosen_arm[i])/len(set_chosen_arm[i])
                else:
                    est_gain[i]=0.0
        for t in range(K,T):
            for i in range(K):
                decide[i]=est_gain[i]+np.sqrt(1.0*alpha*np.log(t)/len(set_chosen_arm[i]))
            max_est_gain=max(decide)
            best_est_arm=decide.index(max_est_gain)
            gain=np.random.binomial(1,mean_gain_param[best_est_arm])
            #print "Gain by the chosen arm: ",gain
            set_chosen_arm[best_est_arm].append(gain)
            cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[best_est_arm])
            for i in range(K):
                est_gain[i]=1.0*np.sum(set_chosen_arm[i])/len(set_chosen_arm[i])
        
        
        cum_regret_t.append(cum_regret)
    
    regret_mean = []
    regret_err = []
    time_epoch=[i for i in range(T)]
    cum_regret_tr=[[0 for i in range(npath)]for j in range(T)]    
    for i in range(T):
        for j in range(npath):
            cum_regret_tr[i][j]=cum_regret_t[j][i]
    freedom_degree = len(cum_regret_tr[0]) - 2
    for regret in cum_regret_tr:
        regret_mean.append(np.mean(regret))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) *ss.sem(regret))
#    colors = list("rgbcmyk")
#    shape = ['--^', '--d', '--v']
    plt.errorbar(time_epoch, regret_mean, regret_err, color=colors[1])
    plt.plot(time_epoch, regret_mean, colors[1] + shape[1], label='UCB')
            
            
def thompson():
    cum_regret_t=[]
    print "Thompson Sampling Running!"
    for path in range(npath):
        print "Path No.: ",path
        cum_regret=[0.0 for i in range(T)]
        theta=[0.0 for i in range(K)]
        set_chosen_arm=[[] for i in range(K)]
        param=[[0,0] for i in range(K)]
        for t in range(T):
            for i in range(K):
                theta[i]=np.random.beta(param[i][0]+1,param[i][1]+1)
            max_est_gain=max(theta)
            best_est_arm=theta.index(max_est_gain)
            gain=np.random.binomial(1,mean_gain_param[best_est_arm])
            if gain==1:
                param[best_est_arm][0]=param[best_est_arm][0]+1
            else:
                param[best_est_arm][1]=param[best_est_arm][1]+1
            if t==0:
                cum_regret[t]=mean_gain_param[0]-mean_gain_param[best_est_arm]
                #print "r=",r
                #print "###########################",mean_gain_param[0],mean_gain_param[r],cum_regret[t]
            else:
                cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[best_est_arm])
        cum_regret_t.append(cum_regret)
        
    regret_mean = []
    regret_err = []
    time_epoch=[i for i in range(T)]
    cum_regret_tr=[[0 for i in range(npath)]for j in range(T)]    
    for i in range(T):
        for j in range(npath):
            cum_regret_tr[i][j]=cum_regret_t[j][i]
    freedom_degree = len(cum_regret_tr[0]) - 2
    for regret in cum_regret_tr:
        regret_mean.append(np.mean(regret))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) *ss.sem(regret))
#    colors = list("rgbcmyk")
#    shape = ['--^', '--d', '--v']
    plt.errorbar(time_epoch, regret_mean, regret_err, color=colors[2])
    plt.plot(time_epoch, regret_mean, colors[2] + shape[2], label='Thompson Sampling')
    
    
    

def UCB_V():
    cum_regret_t=[]
    print "UCB-V Running!"
    alpha=1.5
    for path in range(npath):
        print "Path No.: ",path
        cum_regret=[0.0 for i in range(T)]
        est_gain=[0.0 for i in range(K)]
        est_gain_var=[0.0 for i in range(K)]
        set_chosen_arm=[[] for i in range(K)]
        p_t=[1.0/K for i in range(K)]
        decide=[0 for i in range(K)]
        beta=1.2
        c=1
        for t in range(K):
            if t==0:
                cum_regret[t]=mean_gain_param[0]-mean_gain_param[t]
                #print "r=",r
                #print "###########################",mean_gain_param[0],mean_gain_param[r],cum_regret[t]
            else:
                cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[t])
            gain=np.random.binomial(1,mean_gain_param[t])
            set_chosen_arm[t].append(gain)            
            for i in range(K):
                if(len(set_chosen_arm[i])>0):
                    est_gain[i]=1.0*np.sum(set_chosen_arm[i])/len(set_chosen_arm[i])
                else:
                    est_gain[i]=0.0
        for t in range(K,T):
            for i in range(K):
                decide[i]=est_gain[i]+np.sqrt(2.0*beta*np.log(t)*est_gain_var[i]/len(set_chosen_arm[i]))+(3.0*c*beta*np.log(t)/len(set_chosen_arm[i]))
            max_est_gain=max(decide)
            best_est_arm=decide.index(max_est_gain)
            gain=np.random.binomial(1,mean_gain_param[best_est_arm])
            #print "Gain by the chosen arm: ",gain
            set_chosen_arm[best_est_arm].append(gain)
            cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[best_est_arm])
            for i in range(K):
                est_gain[i]=1.0*np.sum(set_chosen_arm[i])/len(set_chosen_arm[i])
                est_gain_var[i]=np.var(set_chosen_arm[i])
        
        cum_regret_t.append(cum_regret)
    
    regret_mean = []
    regret_err = []
    time_epoch=[i for i in range(T)]
    cum_regret_tr=[[0 for i in range(npath)]for j in range(T)]    
    for i in range(T):
        for j in range(npath):
            cum_regret_tr[i][j]=cum_regret_t[j][i]
    freedom_degree = len(cum_regret_tr[0]) - 2
    for regret in cum_regret_tr:
        regret_mean.append(np.mean(regret))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) *ss.sem(regret))
#    colors = list("rgbcmyk")
#    shape = ['--^', '--d', '--v']
    plt.errorbar(time_epoch, regret_mean, regret_err, color=colors[3])
    plt.plot(time_epoch, regret_mean, colors[3] + shape[3], label='UCB-V')
    
            
            
            
def KL_UCB():
    cum_regret_t=[]
    print "KL-UCB Running!"
    alpha=1.5
    for path in range(npath):
        print "Path No.: ",path
        cum_regret=[0.0 for i in range(T)]
        est_gain=[0.0 for i in range(K)]
        set_chosen_arm=[[] for i in range(K)]
        p_t=[1.0/K for i in range(K)]
        decide=[0 for i in range(K)]
        for t in range(K):
            if t==0:
                cum_regret[t]=mean_gain_param[0]-mean_gain_param[t]
                #print "r=",r
                #print "###########################",mean_gain_param[0],mean_gain_param[r],cum_regret[t]
            else:
                cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[t])
            gain=np.random.binomial(1,mean_gain_param[t])
            set_chosen_arm[t].append(gain)            
            for i in range(K):
                if(len(set_chosen_arm[i])>0):
                    est_gain[i]=1.0*np.sum(set_chosen_arm[i])/len(set_chosen_arm[i])
                else:
                    est_gain[i]=0.0
        for t in range(K,T):
            for i in range(K):
                decide[i]=compute_q(est_gain[i],len(set_chosen_arm[i]),np.log(t))
            max_est_gain=max(decide)
            best_est_arm=decide.index(max_est_gain)
            gain=np.random.binomial(1,mean_gain_param[best_est_arm])
            #print "Gain by the chosen arm: ",gain
            set_chosen_arm[best_est_arm].append(gain)
            cum_regret[t]=cum_regret[t-1]+(mean_gain_param[0]-mean_gain_param[best_est_arm])
            for i in range(K):
                est_gain[i]=1.0*np.sum(set_chosen_arm[i])/len(set_chosen_arm[i])
        cum_regret_t.append(cum_regret)
    
    regret_mean = []
    regret_err = []
    time_epoch=[i for i in range(T)]
    cum_regret_tr=[[0 for i in range(npath)]for j in range(T)]    
    for i in range(T):
        for j in range(npath):
            cum_regret_tr[i][j]=cum_regret_t[j][i]
    freedom_degree = len(cum_regret_tr[0]) - 2
    for regret in cum_regret_tr:
        regret_mean.append(np.mean(regret))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) *ss.sem(regret))
#    colors = list("rgbcmyk")
#    shape = ['--^', '--d', '--v']
    plt.errorbar(time_epoch, regret_mean, regret_err, color=colors[4])
    plt.plot(time_epoch, regret_mean, colors[4] + shape[4], label='KL-UCB')
           
    
            
        





epsilon_greedy()          
UCB()
thompson()
UCB_V()      
KL_UCB()
    
plt.legend(loc='upper right', numpoints=1)
plt.title("Cumulative Pseudo Regret vs T for T = 25000 and 20 Sample paths")
plt.xlabel("T")
plt.ylabel("Cumulative Pseudo Regret")