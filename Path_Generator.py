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


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    
    df_fc = df_forecast.copy()
    columns = df_train.columns
    
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff

        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    
    return df_fc


#%%
############################################

#Multivariate time series data
dataset = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Household Data 1970.csv', 
                          usecols=range(1,9), engine='python')

#Interpolate and drop any nan rows
dataset=interpolator('NETEXP', dataset)
dataset=interpolator('M2V', dataset)

#Drop vars go here
drop_vars=['M2V']
dataset=dataset.drop(drop_vars, axis=1)

dataset=dataset.dropna()


#Colnames and indices for MP variables
col_names=list(dataset.columns)
FFR_idx=col_names.index('FFR')
#MSG_idx=col_names.index('M2SL_PCH')
CPI_idx=col_names.index('CPI')
URATE_idx=col_names.index('URATE')

#Forecast or validation?

FC=True

#Get number of months real quick
N=len(dataset)
fc_periods=12

if FC==True:
    df=dataset.iloc[0:N,:]
    fc_dates=['01/01/2023','02/01/2023','03/01/2023','04/01/2023','05/01/2023','06/01/2023',
              '07/01/2023','08/01/2023','09/01/2023','10/01/2023','11/01/2023','12/01/2023']
if FC==False:
    df=dataset.iloc[0:(N-fc_periods),:]
    df_test=dataset.iloc[(N-fc_periods):N,:]
    

FFR_idx=0

N=len(df)

#Split out our two outcome variables
CPI=dataset['CPI'].values
CPI=CPI[0:N].reshape(N,1)

Urate=dataset['URATE'].values
Urate=Urate[0:N].reshape(N,1)

#Pull corresponding dates
dates = pandas.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Household Data 1970.csv', usecols=[0], engine='python')
dates=dates.DATE[dataset.index.values]



#%% VAR

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

# ADF Test on each column
for name, column in df.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

#URATE, CPI, M2SL_PCH are stationary

df_differenced = df.diff().dropna()
stationary_cols=['M2SL_PCH']

df_differenced=df_differenced.drop(stationary_cols,axis=1)
for i in range(len(stationary_cols)): df_differenced[stationary_cols[i]]=df[stationary_cols[i]][1:]

# ADF Test on each column
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

model = VAR(df_differenced)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

x = model.select_order(maxlags=12)
print(x.summary())

model_fitted = model.fit(12)
model_fitted.summary()


#%%
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(col, ':', round(val, 2))

# Get the lag order
lag_order = model_fitted.k_ar

# Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
forecast_input

# Generate Forecast

#Lets use the actual FFR changes
#Manual update for observation 1
FFR_diff=df_test['FFR'].diff()
FFR_diff[0]=0

for i in range(fc_periods):
    
    fc = model_fitted.forecast(y = forecast_input, steps=1)
    fc[0,FFR_idx]=FFR_diff.iloc[i]
    
    if i==0: df_forecast = pd.DataFrame(fc, columns=df.columns+'_1d')
    else: df_forecast = df_forecast.append(pd.DataFrame(fc, columns=df.columns+'_1d'))
    
    forecast_input=np.delete(forecast_input, 1, 0)
    forecast_input = np.vstack([forecast_input, fc])


stationary_cols=['M2SL_PCH_forecast']

df_results_var = invert_transformation(df, df_forecast, second_diff=False)  
#df_results_var=df_results_var.drop(stationary_cols,axis=1)

#for i in range(len(stationary_cols)): df_results_var[stationary_cols[i]]=df_forecast[stationary_cols[i]+'_1d'][1:]

if FC==False:
    df_results_var['Date']=np.array(dates[-fc_periods:])
if FC==True:
    df_results_var['Date']=np.array(fc_dates)
    
df_results_var.set_index("Date", inplace=True)

df_test['Date']=np.array(dates[-fc_periods:])
df_test.set_index("Date", inplace=True)

df_results_var['CPI_forecast'].plot(legend=True);
df_test['CPI'].plot(legend=True)
plt.show()

df_results_var['URATE_forecast'].plot(legend=True);
df_test['URATE'].plot(legend=True)
plt.show()



#%% Fucking VECM
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
vec_rank1 = select_coint_rank(df, det_order = 1, k_ar_diff = 1, method = 'trace', signif=0.01)
print(vec_rank1.summary())


model = VECM(endog = df, k_ar_diff = 12, coint_rank = 3, deterministic = 'co')
res = model.fit()

X_pred = res.predict(steps=fc_periods)
df_results_vecm=pd.DataFrame(X_pred, columns=col_names)

if FC==False:
    df_results_vecm['Date']=np.array(dates[-fc_periods:])
if FC==True:
    df_results_vecm['Date']=np.array(fc_dates)

df_results_vecm.set_index("Date", inplace=True)


#%% Neuralnet estimation 

#Convert everything to arrays
#Reshape and convert to float32 (loaded in as object type)

differenced=False

if differenced==True:
    dataset=df_differenced.astype('float32')
    dataset=dataset.values
if differenced==False:
    dataset=df.values
    dataset=dataset.astype('float32')


#Grab last MP variables
last_FFR=dataset[N-1,FFR_idx]
#last_MSG=dataset[N-1,MSG_idx]

#Alright we want to predict the CPI inflation + URATE at t given all data before t
#Clean up dates, convert to usable list
import datetime as dt
dates=np.array(dates)
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

CPI_min=min(trainX[:,FFR_idx])
CPI_max=max(trainX[:,FFR_idx])

URATE_min=min(trainX[:,FFR_idx])
URATE_max=max(trainX[:,FFR_idx])

#MSG_min=min(trainX[:,MSG_idx])
#MSG_max=max(trainX[:,MSG_idx])

#Lets normalize the dataset real fast
#scaler = MinMaxScaler(feature_range=(0, 1))
#obj = scaler.fit(trainX)
#trainX = obj.transform(trainX)
#trainY= obj.transform(trainY)

scale_min=np.min(trainX,axis=0)
scale_max=np.max(trainX,axis=0)

trainX=(trainX-np.min(trainX,axis=0))/(np.max(trainX,axis=0)-np.min(trainX,axis=0))
trainY=(trainY-np.min(trainY,axis=0))/(np.max(trainY,axis=0)-np.min(trainY,axis=0))

#%%
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0] ,1, trainX.shape[1] ))

import random
random.seed(1337)

N=trainX.shape[0]
time_dim=trainX.shape[1]
feature_dim=trainX.shape[2]

out_dim = trainY.shape[1]

#Lets build a neural net!
model=Sequential()
model.add(LSTM(32, recurrent_dropout=0, input_shape=(1,feature_dim)))
model.add(Dense(256))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16))
model.add(Dense(8,activation='selu'))
model.add(Dense(out_dim))

#Compile and estimate
model.compile(loss='mse', optimizer='adam', run_eagerly=())
model.fit(trainX, trainY, epochs=1000, batch_size=250, verbose=2)

model.predict(trainX)

#Define QE scenarios
scenario='Mod' #Agressive, Lax, Observed

if scenario == 'Mod':
    periods=12
    end_period=12
    size=.75
    #rate=-.1

elif scenario == 'Lax':
    periods=12
    end_period=7
    size=.250
    #rate=.10

elif scenario == 'Mod':
    periods=12
    end_period=3
    size=.25
    #rate=.10

#Initialize counterfactual paths
CF_data=np.zeros([fc_periods, len(col_names)])
FFR_data=df_test['FFR']
K=len(trainX)-1

#Generate N+1 forecast
forward=model.predict(trainX)[K,:].reshape(1,feature_dim)

new_FFR=last_FFR+size
last_FFR=new_FFR

#new_MSG=rate

forward[0,FFR_idx] = (new_FFR - FFR_min) / (FFR_max - FFR_min)
#forward[0, MSG_idx] = (new_MSG - MSG_min) / (MSG_max - MSG_min)
CF_data[0,:]=(forward)
CF=True

for i in range(fc_periods-1):


    dat=np.vstack([trainX[1:N,:].reshape(K,feature_dim),forward])
    dat=dat.reshape(N,1,feature_dim)
    
    forward=model.predict(dat)[K,:].reshape(1,feature_dim)
    
    if CF==False:
        if differenced==True:
            new_FFR=FFR_diff[i]
        if differenced==False:
            new_FFR=df_test['FFR'].iloc[i]
            forward[0,FFR_idx] = (new_FFR - FFR_min) / (FFR_max - FFR_min)

    if CF==True:
        if i<end_period:
            new_FFR=last_FFR+size
            last_FFR=new_FFR
            
            forward[0,FFR_idx] = (new_FFR - FFR_min) / (FFR_max - FFR_min)
            #forward[0,MSG_idx] = (rate - MSG_min) / (MSG_max - MSG_min)
             
        else:
            forward[0,FFR_idx]=last_FFR
        
    CF_data[i+1]=(forward)      


CF_data

print(str(fc_periods)+' month URATE forecast')
print(CF_data[0:fc_periods,URATE_idx])

print(str(fc_periods)+' month CPI forecast')
print(CF_data[0:fc_periods,CPI_idx])

if differenced==True:
    df_results_lstm=pd.DataFrame(CF_data,columns=[x +"_1d" for x in col_names ])
    df_results_lstm=invert_transformation(df, df_results_lstm, second_diff=False) 
    
if differenced==False:
    df_results_lstm=pd.DataFrame(CF_data*(scale_max-scale_min)+scale_min,columns=col_names)

time_var=np.array(dates[-fc_periods:])
time_var=[dt.datetime.strptime(date, "%m/%d/%Y").date() for date in time_var]

if FC==True:
    df_results_lstm['Date']=np.array(fc_dates)    
    df_results_lstm.set_index("Date", inplace=True)
    
    df_results_var['Date']=np.array(fc_dates)
    df_results_var.set_index("Date", inplace=True)
    
    df_results_vecm['Date']=np.array(fc_dates)
    df_results_vecm.set_index("Date", inplace=True)
    
    df_test['Date']=np.array(fc_dates)
    df_test.set_index("Date", inplace=True)
    
if FC==False:
    df_results_lstm['Date']=np.array(time_var)
    df_results_lstm.set_index("Date", inplace=True)
    
    df_results_var['Date']=np.array(time_var)
    df_results_var.set_index("Date", inplace=True)
    
    df_results_vecm['Date']=np.array(time_var)
    df_results_vecm.set_index("Date", inplace=True)
    
    df_test['Date']=np.array(time_var)
    df_test.set_index("Date", inplace=True)



#Plot that shit
import matplotlib.patches as mpatches

df_results_lstm['CPI'].plot(legend=True,label='CPI LSTM Forecast');
df_results_var['CPI_forecast'].plot(legend=True,label='CPI VAR Forecast');
df_results_vecm['CPI'].plot(legend=True,label='CPI VECM Forecast');
#df_test['CPI'].plot(legend=True)

plt.xticks(rotation=45)
plt.title("CPI: Forecast vs Actuals")

plt.show()

df_results_lstm['URATE'].plot(legend=True,label='URATE LSTM Forecast');
df_results_var['URATE_forecast'].plot(legend=True,label='URATE VAR Forecast');
df_results_vecm['URATE'].plot(legend=True,label='URATE VECM Forecast');
#df_test['URATE'].plot(legend=True)

plt.xticks(rotation=45)
plt.title("URATE: Forecast vs Actuals")

plt.show()



#%% Save? Run this
out_array=np.array([df_test['CPI'], df_results_var['CPI_forecast'], df_results_lstm['CPI'] ,df_results_vecm['CPI'],
                    df_test['URATE'] ,df_results_var['URATE_forecast'] ,df_results_lstm['URATE'], df_results_vecm['URATE']])
out_array=np.transpose(out_array)
df_output=pd.DataFrame(out_array, columns=['CPI','CPI VAR','CPI LSTM', 'CPI VECM', 'URATE','URATE VAR','URATE LSTM','URATE VECM'])

df_output.to_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Good Paths\FC_results'+ str(fc_periods)+'forecast.csv')

#%% Ensemble Model

Ens_data=pandas.read_csv(r'C:/Users/mahon/Documents/Python Scripts/NN Practice/Ensemble_data.csv', engine='python')

#Now get VAR fits
VAR_fit=model_fitted.fittedvalues

#Generate fitted values for CPI
x = np.array(VAR_fit['CPI'])
z = np.insert(x,0,df['CPI'][155])
CPI_VAR_fit=np.cumsum(z)

#Generate fitted values for URATE
x = np.array(VAR_fit['URATE'])
z = np.insert(x,0,df['URATE'][155])
URATE_VAR_fit=np.cumsum(z)

K=len(URATE_VAR_fit)

#Lets start by getting fits from the LSTM model
LSTM_fit=model.predict(trainX)
LSTM_fit=LSTM_fit*(scale_max-scale_min)+scale_min

CPI_LSTM_fit=LSTM_fit[:,CPI_idx][-K:]
URATE_LSTM_fit=LSTM_fit[:,URATE_idx][-K:]

Urate=Urate.reshape(487)[-K:]
CPI=CPI.reshape(487)[-K:]

CPI=Ens_data['CPI']
URATE=Ens_data['URATE']

CPI_LSTM_fit=Ens_data['CPI LSTM']
CPI_VAR_fit=Ens_data['CPI VAR']
CPI_VECM_fit=Ens_data['CPI VECM']

URATE_LSTM_fit=Ens_data['URATE LSTM']
URATE_VAR_fit=Ens_data['URATE VAR']
URATE_VECM_fit=Ens_data['URATE VECM']


def CPI_objective(params):
    alpha, beta = params
    val=np.sum(((1-alpha-beta)*CPI_LSTM_fit + alpha*CPI_VAR_fit + beta*CPI_VECM_fit-CPI)**2) + np.sum(((1-alpha-beta)*URATE_LSTM_fit + alpha*URATE_VAR_fit+beta*URATE_VECM_fit - URATE)**2)
    return val

def URATE_objective(params):
    alpha, beta = params
    val= np.sum(((1-alpha-beta)*URATE_LSTM_fit + alpha*URATE_VAR_fit+beta*URATE_VECM_fit - URATE)**2)
    return val

guess=[.5,.5]

from scipy.optimize import minimize
bnds = ((0, 1), (0, 1))

results=minimize(CPI_objective, guess, bounds=bnds)
alpha=results.x[0]
beta=results.x[1]

comp_cpi=(1-alpha-beta)*df_results_lstm['CPI']+alpha*df_results_var['CPI_forecast']+beta*df_results_vecm['CPI']
comp_cpi=(1-alpha-beta)*df_results_lstm['URATE']+alpha*df_results_var['URATE_forecast']+beta*df_results_vecm['URATE']


#%%













#%%
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))

for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):

    df_results_var[col+'_forecast'].plot(legend=True, ax=ax)
    df_results_lstm[col].plot(legend=True, ax=ax);
    df_test[col].plot(legend=True, ax=ax);
    
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();

df_results_var[col+'_forecast'].iloc[:,1].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
np.array(df_results_var[col+'_forecast']).plot()
plt.plot()



#%%


import shap
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

DE=shap.DeepExplainer(model, trainX)
shap_values = DE.shap_values(trainX) # X_validate is 3d numpy.ndarray

shap_values=np.array(shap_values[0]).reshape(N,feature_dim)

shap.initjs()
shap.summary_plot(
    shap_values, 
    trainX.reshape(N,feature_dim),
    feature_names=col_names,
    max_display=10,
    plot_type='bar')


shap.plots.bar(shap_values)

x=np.concatenate( shap_values, axis=0 ).reshape(N,feature_dim)


#%%
from datetime import datetime
now=datetime.now()
now=str(now)[0:19]
now=now.replace(':',"-")
now=now.replace(' ',"-")

path=r"C:/Users/mahon/Documents/Python Scripts/NN Practice/CF_Paths/"+scenario+"-"+str(now)[0:19]+".csv"

df=pd.DataFrame(CF_data,columns=col_names)
df.to_csv(path)

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

plt.plot(CF_data[:,URATE_idx])
plt.gcf().autofmt_xdate()
plt.show()

plt.plot(CF_data[:,CPI_idx])
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



