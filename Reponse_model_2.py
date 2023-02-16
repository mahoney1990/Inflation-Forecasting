# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:32:58 2023

@author: mahon
"""
import numpy as np
from interpolation import interp

from scipy.optimize import minimize_scalar, bisect
from scipy.interpolate import interp1d
from scipy.optimize import minimize

class ifp:
    
    def __init__(self,
                 r=0.0000,
                 beta=0.99999,
                 gamma=0.2,
                 y=(1, 0),
                 grid_size=101,
                 grid_max=10):
        
        self.R = 1+r
        self.beta, self.gamma = beta, gamma
        self.y =  np.array(y)
        self.asset_grid = np.linspace(0, grid_max, grid_size)
        
        assert self.R * self.beta < 1, "Shes unstable, Captain!"
    
    def u(self,c):
        return ((c)**(1-self.gamma))/(1-self.gamma)

    def u_prime(self, c):
        return (c+.0000000001)**-self.gamma
        
#Define objective      
def euler_diff(c, a, z, p, sigma_values, ifp):
    R,  y, beta, gamma = ifp.R,  ifp.y, ifp.beta, ifp.gamma
    asset_grid, u_prime = ifp.asset_grid, ifp.u_prime
    
    n=len(P)
    
    #
    def sigma(a,z):
        return interp(asset_grid, sigma_values[:,z], a)
        
    
    #Calculate Expectations conditional on current z
    expect=0.0
    for z_hat in range(n):
        expect += u_prime(sigma(R*(a)-p*c+y[z_hat],z_hat)) * P[z,z_hat]
    
    return (u_prime(c) - max(beta*R*expect, u_prime(a)))**2


#Contraction mapping 
def K(sigma, ifp):
    sigma_new=np.empty_like(sigma)
    
    for i, a in enumerate(asset_grid):
        for z in (0,1):
            result = minimize_scalar(euler_diff, bounds=(0,a+y[z]), 
                                     args=(a,z,p,sigma,ifp), method='bounded')
    
            sigma_new[i,z]=result.x
    
    return sigma_new 


def solver(model,sigma,tol=.001,max_iter=1000,verbose=True, print_skip=25):
    
    dist=tol+1
    i=0
    
    while i<max_iter and dist>tol:
        sigma_new=K(sigma, ifp)
        dist = np.max(np.abs(sigma-sigma_new))
        
        i+=1
        if verbose and i % print_skip ==0:
             print(f"Error at Iteration #"+str(i)+":"+str(dist))
        sigma=sigma_new
    
    if i == max_iter:
        print("Fuck!")
        
    if verbose and i < max_iter:
        print("Convergence Achieved")
    
    return sigma_new


ifp=ifp()

P=np.array([ [.05,.95] , [.75,.25] ])
p=1

# Initalize will consumption of all assets at all z
z_size=len(P)
asset_grid = ifp.asset_grid
a_size=len(asset_grid)
sigma_init=np.repeat(asset_grid.reshape(a_size,1), z_size, axis=1 )
y=ifp.y



sigma_star=solver(ifp, sigma_init)

#%% RUN THIS FOR BASELINE

#Define policy sequences for unemployment probabilities
#Here we will import from the neuralnet model, but for now lets just make some up

#Write transition matrices into a dictionary
policy_dict={}
sigma_dict={}

T=12

EE=np.zeros(T)
EE[:]=1-.045
EU=1-EE
UE=.25
UU=.75

for k in range(len(EE)):
    policy_k=np.zeros([2,2])
    policy_k[0,0]=EE[k]
    policy_k[0,1]=EU[k]
    policy_k[1,0]=UE
    policy_k[1,1]=UU
    
    policy_dict[k]=policy_k

#Grab that monthly CPI forecast 
pi=.02/12
price_forecast=np.zeros(T)


T=len(price_forecast)

for i in range(T):
    if i == 0:
        price_forecast[i]=1
    else:
        price_forecast[i]=price_forecast[i-1]*(1+pi)

#Calulate consumption at time t given assets and price/policy forecasts
for k in range(T):
    print(k)
    P=np.array(policy_dict[k])
    p=price_forecast[k]
    
    if k==0:
        sigma_dict[k]=solver(ifp, sigma_init,tol=.001)
    else:
        sigma_dict[k]=solver(ifp, sigma_dict[k-1],tol=.001)


#%% RUN THIS TO Import NN predictions



import pandas as pd

ag_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Good Paths\Aggressive-2023-02-07-11-09-13.csv")
lax_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Good Paths\Lax-2023-02-07-11-52-23.csv")
ob_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Good Paths\Observed.csv")

ag_data=ag_data[0:7]
lax_data=lax_data[0:7]

#Select scenario
scenario='Lax'
if scenario == 'Observed':
    CF_data=ob_data
elif scenario == 'Aggressive':
    CF_data=ag_data
elif scenario =='Lax':
    CF_data=lax_data

#Initialize policy dict/OG guess for sigma matrix
sigma_dict={}
sigma_init=np.repeat(asset_grid.reshape(a_size,1), z_size, axis=1 )   

policy_dict={}

#Generate Price level forecasts
CPI_forecast=CF_data['CPI']/1200
T=len(CPI_forecast)
price_forecast=np.zeros(T)



for i in range(len(CPI_forecast)):
    if i == 0:
        price_forecast[i]=1
    else:
        price_forecast[i]=price_forecast[i-1]*(1+CPI_forecast[i])

#Generate Transition Matrices
URATE_forecast=CF_data['URATE']/100

UE=.25
UU=.75

for k in range(len(URATE_forecast)):
    policy_k=np.zeros([2,2])
    policy_k[0,0]=1-URATE_forecast[k]
    policy_k[0,1]=URATE_forecast[k]
    policy_k[1,0]=UE
    policy_k[1,1]=1-UE    
    policy_dict[k]=policy_k

T=len(price_forecast)

#Calulate consumption at time t given assets and price/policy forecasts
for k in range(T):
    print(k)
    P=np.array(policy_dict[k])
    p=price_forecast[k]
    
    if k==0:
        sigma_dict[k]=solver(ifp, sigma_init,tol=.0005)
    else:
        sigma_dict[k]=solver(ifp, sigma_dict[k-1],tol=.0005)


#%%
#Okay so now we know how households respond, now we need to run our simulations
#The transition probabilites themselves are invariant -- gotta simulate

import random
from numpy.random import binomial
from numpy.random import multinomial

random.seed(420691337)
n_sims=100000
n_dist=5
step=int(n_sims/n_dist)
    
import numpy
from scipy import interpolate

#Build simulation result matrices
a_list=numpy.zeros((T,n_sims))
c_list=numpy.zeros((T,n_sims))
u_list=numpy.zeros((T,n_sims))
consutil_list=numpy.zeros((T,n_sims))

#Build aggregation matrices
c_mat_ag=np.zeros((T))
s_mat_ag=np.zeros((T))
a_mat_ag=np.zeros((T))
U_ag_mat=np.zeros((T))
consutil_mat_ag=np.zeros((T))

it=0

#Define a dictionary to hold draws
z_dict={}

#Pull in asset grid
agrid=ifp.asset_grid


#Generatre unequal asset distribution
a_list[0,0:50000]=1
a_list[0,50001:60000]=1.5
a_list[0,60001:70000]=2
a_list[0,70001:80000]=3
a_list[0,80001:90000]=4     
a_list[0,90001:95000]=5
a_list[0,95001:n_sims]=10    

#Verbose interval
print_skip=1000

import matplotlib.pyplot as plt

#Outer loop though number of simulations 
for j in range(n_sims):
        
    if j % print_skip ==0:
        print(f"Simulating Household Responses. Progress:"+str(j/n_sims*100)+"%")
     
    
    #Simulate a sequence of draws based on policy functions. Note price effect is not random
    z_vec=numpy.zeros((T,1))
    z_vec=z_vec.astype(int)
    
    
    for k in range(T):
        
        #Initalize by assigning intital employed/unemployed groups
        if k==0:
            P=np.array(policy_dict[k])    
            z_int = multinomial(1, [P[0,0], P[0,1]] )
            z_vec[k]=np.where(z_int==1)[0][0]
        
        else:
            z_vec[k]=z_new
        
        #Update draws
        p=price_forecast[k]
        P=np.array(policy_dict[k])    
        
        roll=multinomial(1, [P[z_vec[k][0],0], P[z_vec[k][0],1]] )
        
        z_new=np.where(roll==1)[0][0]
        

    z_dict[j]=z_vec
        
    Util=np.empty(T)
    cons=np.empty(T)
    income=np.empty(T)
    a=np.empty(T)

    #Simulate consumption
    for k in range(T):
        #Pull decision matrix for time k
        sigma=sigma_dict[k]
        
        #Grab state at time k
        z=z_vec[k][0]
        
        #Interpolate along the decision grid to find optimal consumption
        f=interp1d(agrid, sigma[:, z],kind='quadratic',fill_value="extrapolate")
        c_list[k,j]=f(a_list[k,j])
        
        #Calulate utility 
        consutil_list[k,j]=(ifp.beta**k)*(1/ifp.gamma)*c_list[k,j]**(1-ifp.gamma)
            
        income[k]=y[z]
            
        u_list[k,j]=ifp.u(c_list[k,j])
        
        
        #Update asset holdings
        if k<T-1:
            a_list[k+1,j]=a_list[k,j]+income[k]-c_list[k,j]


    c_mat_ag[:]=np.sum(c_list,axis=1)/n_sims
    a_mat_ag[:]=np.sum(a_list,axis=1)/n_sims
    consutil_mat_ag[:]=np.sum(consutil_list,axis=1)/n_sims
    
    for j in range(T-1):
        if j<T:
            s_mat_ag[j]=a_mat_ag[j+1]-a_mat_ag[j]
    
    
utility=0
    
    
E_util=np.zeros(T)
E_cons=np.zeros(T)
E_assets=np.zeros(T-1)
    
#AGGREGATION
for j in range(T-1):
    
    utility=0
    consum=0
    Assets=0
    
    for i in range(n_sims):
        utility+=(ifp.beta**j)*ifp.u(c_list[j,i])
        consum+=c_list[j,i]/n_sims
        Assets+=a_list[j,i]/n_sims
    

    E_cons[j]=consum
    E_util[j]=(ifp.beta**j)*ifp.u(consum)
    E_assets[j]=Assets
    U_ag_mat[j]=E_util[j]

#Consumption EQ
if scenario == 'Observed':
    X_star = sum(E_util)
else:
    lam=(X_star/(sum(E_cons))**(1/(1-ifp.gamma))) - 1    

    
savings=E_assets[2:(T-1)]-E_assets[1:(T-2)]
t = [1,2,3,4,5,6]

fig = plt.figure()
ax = plt.axes()

plt.plot(t,(E_cons[0:6]/E_cons[0]),"-r", label="Consumption")
plt.legend(loc="upper left")
plt.title(scenario+ ": Consumption July 2022-Jan 2023")
plt.xlabel('Month')

from datetime import datetime
now=datetime.now()
now=str(now)[0:19]
now=now.replace(':',"-")
now=now.replace(' ',"-")

path=r"C:/Users/mahon/Documents/Python Scripts/NN Practice/CF_Paths\Model Output/"+scenario+"-"+str(now)[0:19]+".csv"

dat=np.array([E_cons[0:6],E_util[0:6],E_assets])

dat=np.transpose(dat)
df=pd.DataFrame(dat,columns=['Cons','Util','Assets'])
df.to_csv(path)






