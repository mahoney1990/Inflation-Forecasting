# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:24:56 2022

@author: mahon
"""

#We're gonna need some packages here
# L O A D I N G
##################################################
import pandas
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
###################################################

######## F U N C T I O N S
def gen_ts(data,frame=1):
    dataY=[]
    dataX=[]
    frame=1
    for i in range(len(data)-frame-1):      
        add=data[i:(i+frame),:]
        dataX.append(add)
        dataY.append(data[i+frame,:])
    
    return np.array(dataX), np.array(dataY)



##Scenario Variables
end_period=4
size=.2

def CF_gen(model, trainX, periods=12,):
    CF_data=np.zeros([periods, len(col_names)])
    K=len(trainX)-1
    
    #Generate N+1 forecast
    forward=model.predict(trainX)[K,:].reshape(1,feature_dim)
    
    new_FFR=last_FFR+size
    last_FFR=new_FFR
    
    new_MSG=rate
    
    forward[0,FFR_idx] = (new_FFR - FFR_min) / (FFR_max - FFR_min)
    forward[0, MSG_idx] = (new_MSG - MSG_min) / (MSG_max - MSG_min)
    CF_data[0,:]=(forward)

    for i in range(periods-1):

        dat=np.vstack([trainX[1:N,:].reshape(K,feature_dim),forward])
        dat=dat.reshape(N,1,feature_dim)
        
        forward=model.predict(dat)[K,:].reshape(1,feature_dim)
        
        if i<end_period:
            new_FFR=last_FFR+size
            last_FFR=new_FFR
            
            forward[0,FFR_idx] = (new_FFR - FFR_min) / (FFR_max - FFR_min)
            forward[0,MSG_idx] = (rate - MSG_min) / (MSG_max - MSG_min)
             
        else:
            forward[0,FFR_idx]=CF_data[i,FFR_idx]
    
    CF_data[i+1]=(forward)      
   
        
    return CF_data


#Quarterly Interpolation
def interpolator(Var_name, dataset):
    for i in range(len(dataset)-3):
        if i % 3 == 0:
        
            lower_idx=i
            upper_idx=lower_idx+3
            
            lower=dataset[Var_name][lower_idx]
            upper=dataset[Var_name][upper_idx]
            
            step=(upper-lower)/3
            
            dataset[Var_name][lower_idx+1]=lower+step
            dataset[Var_name][lower_idx+2]=lower+2*step
        
    return dataset



#%%
############################################

#Multivariate time series data
dataset = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Household Data 1970.csv', usecols=range(1,7), engine='python')

#Interpolate and drop any nan rows
dataset=interpolator('NETEXP', dataset)
dataset=dataset.dropna()

#Colnames and indices for MP variables
col_names=list(dataset.columns)
FFR_idx=col_names.index('FFR')
MSG_idx=col_names.index('M2SL_PCH')
CPI_idx=col_names.index('CPI')
URATE_idx=col_names.index('URATE')

#Get number of months real quick
N=len(dataset)

#Split out our two outcome variables
CPI=dataset['CPI'].values
CPI=CPI.reshape(N,1)

Urate=dataset['URATE'].values
Urate=Urate.reshape(N,1)

#Convert everything to arrays
dataset=dataset.values

#Pull corresponding dates
dates = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Household Data 1970.csv', usecols=[0], engine='python')

#Reshape and convert to float32 (loaded in as object type)
dataset=dataset.astype('float32')

#Grab last MP variables
last_FFR=dataset[N-1,FFR_idx]
last_MSG=dataset[N-1,MSG_idx]

#Alright we want to predict the CPI inflation + URATE at t given all data before t
#Clean up dates, convert to usable list
import datetime as dt
dates=np.array(dates)
dates=dates[:127]
dates=dates.flatten()

dates_list = [dt.datetime.strptime(date, "%m/%d/%Y").date() for date in dates]

#Define Lag variable(s)
frame=1

#Careful here, your data needs the right shape
#Generate TS for CPI and URATE
trainX, trainY = gen_ts(dataset)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2] ))

#Grab min and max for later
FFR_min=min(trainX[:,FFR_idx])
FFR_max=max(trainX[:,FFR_idx])

MSG_min=min(trainX[:,MSG_idx])
MSG_max=max(trainX[:,MSG_idx])

#Lets normalize the dataset real fast
scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)
trainY=scaler.fit_transform(trainY)


#%%
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0] ,1, trainX.shape[1] ))

import random
random.seed(420)

N=trainX.shape[0]
time_dim=trainX.shape[1]
feature_dim=trainX.shape[2]

out_dim = trainY.shape[1]

#Lets build us a neural net!
model=Sequential()
model.add(LSTM(32, recurrent_dropout=0.15, input_shape=(1,feature_dim)))
#model.add(Dense(1024))
#model.add(Dense(512))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(out_dim, activation='linear'))

#Compile and estimate
model.compile(loss='mae', optimizer='adam',run_eagerly=())
model.fit(trainX, trainY, epochs=35, batch_size=20, verbose=2)


model.predict(trainX)

#Define QE scenarios
scenario='Aggressive' #Agressive, Lax, Observed

if scenario == 'Aggressive':
    periods=12
    end_period=7
    size=.75
    rate=-.1

elif scenario == 'Lax':
    periods=12
    end_period=7
    size=.25
    rate=-.05


#Initialize counterfactual paths
CF_data=np.zeros([periods, len(col_names)])
K=len(trainX)-1

#Generate N+1 forecast
forward=model.predict(trainX)[K,:].reshape(1,feature_dim)

new_FFR=last_FFR+size
last_FFR=new_FFR

new_MSG=rate

forward[0,FFR_idx] = (new_FFR - FFR_min) / (FFR_max - FFR_min)
forward[0, MSG_idx] = (new_MSG - MSG_min) / (MSG_max - MSG_min)
CF_data[0,:]=(forward)

for i in range(periods-1):


    dat=np.vstack([trainX[1:N,:].reshape(K,feature_dim),forward])
    dat=dat.reshape(N,1,feature_dim)
    
    forward=model.predict(dat)[K,:].reshape(1,feature_dim)
    
    if i<end_period:
        new_FFR=last_FFR+size
        last_FFR=new_FFR
        
        forward[0,FFR_idx] = (new_FFR - FFR_min) / (FFR_max - FFR_min)
        forward[0,MSG_idx] = (rate - MSG_min) / (MSG_max - MSG_min)
         
    else:
        forward[0,FFR_idx]=CF_data[i,FFR_idx]
    
    CF_data[i+1]=(forward)      



CF_data=scaler.inverse_transform(CF_data)

print('12 month URATE forecast')
print(CF_data[:,URATE_idx])

print('12 month CPI forecast')
print(CF_data[:,CPI_idx])

#%%




#%%



def model_forecast(model, series, window_size):

    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)

    forecast = model.predict(ds)
    return forecast

series=trainX
window_size=10
model_forecast(model, trainX, 1)
test_series=trainX

for time in range(0, 6):
   prediction = model.predict(test_series[-window_size:,:,:])
   prediction_unscaled = scaler.inverse_transform(prediction)
   test_series = np.append(test_series, prediction)


#%%

plt.plot(CF_data[:,1])
plt.gcf().autofmt_xdate()
plt.show()


#Lets test it
trainPredict = model.predict(trainX)
trainPredict=scaler.inverse_transform(trainPredict)
trainX=scaler.inverse_transform(trainX.reshape(N,3))

#Looks pretty good!
import matplotlib.dates as mdates

plt.plot(dates_list,trainX[:,0])
plt.plot(dates_list,trainPredict[:, 0])
plt.gcf().autofmt_xdate()
plt.show()


plt.plot(dates_list,CPI[1:128])
plt.plot(dates_list,trainPredict[:, URATE_idx])
plt.gcf().autofmt_xdate()
plt.show()


import shap
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

DE=shap.DeepExplainer(model, trainX)
shap_values = DE.shap_values(trainX) # X_validate is 3d numpy.ndarray

shap_values=np.array(shap_values).reshape(N,feature_dim)

shap.initjs()
shap.summary_plot(
    shap_values, 
    trainX.reshape(N,feature_dim),
    feature_names=col_names[1:],
    max_display=10,
    plot_type='bar')

print('August Predicted Inflation:')
model.predict(trainX[126].reshape(1,1,feature_dim))



import  statsmodels
statsmodels.hpfilter(dataset['PRC_Imports'],100)



