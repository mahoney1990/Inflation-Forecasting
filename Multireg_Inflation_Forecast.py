# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:14:20 2022

@author: mahon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:18:19 2022

@author: mahon
"""

import pandas
import matplotlib.pyplot as plt
import numpy as np

#Univariate time series data
dataset = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Inflation_data.csv', usecols=range(1,16), engine='python')
col_names=list(dataset.columns)

#Get number of months real quick

N=len(dataset)

#Split out our two outcome variables
CPI=dataset['CPI'].values
CPI=CPI.reshape(N,1)

Urate=dataset['Urate'].values
Urate=Urate.reshape(N,1)

#Convert everything to arrays
dataset=dataset.values

#Pull corresponding dates
dates = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Inflation_data.csv', usecols=[0], engine='python')

#Reshape and convert to float32 (loaded in as object type)
dataset=dataset.astype('float32')

ex=dataset.describe()

#We're gonna need some packages here
# L O A D I N G
##################################################
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

#Alright we want to predict the CPI inflation at t given all data before t
#Clean up dates, convert to usable list
import datetime as dt
dates=np.array(dates)
dates=dates[:127]
dates=dates.flatten()

dates_list = [dt.datetime.strptime(date, "%m/%d/%Y").date() for date in dates]

#Define Lag variable(s)
frame=1

#Careful here, your data needs the right shape
def gen_ts(data,frame=1):
    dataY=[]
    dataX=[]
    frame=1
    for i in range(len(data)-frame-1):      
        add=data[i:(i+frame),:]
        dataX.append(add)
        dataY.append(data[i+frame,:])
    
    return np.array(dataX), np.array(dataY)


#Generate TS for CPI and URATE
trainX, trainY = gen_ts(dataset)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2] ))

#Lets normalize the dataset real fast
scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)
trainY=scaler.fit_transform(trainY)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0] ,1, trainX.shape[1] ))

N=trainX.shape[0]
time_dim=trainX.shape[1]
feature_dim=trainX.shape[2]

out_dim = trainY.shape[1]

#Lets build us a neural net!
model=Sequential()
model.add(LSTM(32, input_shape=(1,feature_dim)))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(out_dim))
model.compile(loss='mse', optimizer='adam',run_eagerly=())
model.fit(trainX, trainY, epochs=5000, batch_size=60, verbose=2)





def CF_gen(model, trainX, periods=12):
    CF_data=[]
    N=len(trainX)
    
    forward=model.predict(trainX[(N-1):N,:])    
    CF_data.append(forward)
    
    for i in range(periods-1):
        dat=CF_data[i].reshape(1,1,15)
        forward=model.predict(dat)
        CF_data.append(forward)      
        
    return CF_data

CF_data=CF_gen(model, trainX)
CF_data=np.concatenate(CF_data)
CF_data=scaler.inverse_transform(CF_data)

plt.plot(CF_data[:,0])
plt.gcf().autofmt_xdate()
plt.show()


#Lets test it
trainPredict = model.predict(trainX)
trainPredict=scaler.inverse_transform(trainPredict)
trainX=scaler.inverse_transform(trainX.reshape(N,15))

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

trainX=trainX.reshape(127,1,15)

DE=shap.DeepExplainer(model, trainX)
shap_values = DE.shap_values(trainX) # X_validate is 3d numpy.ndarray

shap_values=np.array(shap_values).reshape(N,feature_dim)

shap.initjs()
shap.summary_plot(
    shap_values[0].reshape(N,feature_dim), 
    trainX.reshape(N,feature_dim),
    feature_names=col_names,
    max_display=10,
    plot_type='bar')



print('August Predicted Inflation:')
model.predict(trainX[126].reshape(1,1,feature_dim))



import  statsmodels
statsmodels.hpfilter(dataset['PRC_Imports'],100)



