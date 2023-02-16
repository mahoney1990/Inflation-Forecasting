# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:21:38 2023

@author: mahon
"""

import matplotlib.pyplot as plt
import pandas as pd

ag_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Good Paths\Aggressive-2023-02-07-11-09-13.csv")
lax_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Good Paths\Lax-2023-02-07-11-52-23.csv")
ob_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Good Paths\Observed.csv")

ag_data=ag_data[0:7]
lax_data=lax_data[0:7]

dates=['July', 'Aug', 'Oct','Nov','Dec',"Jan"]
var=['FFR','CPI','URATE']

for i in range(3):
      
    # to set the plot size
    plt.figure(figsize=(8, 8), dpi=250)
      
    # using plot method to plot open prices.
    # in plot method we set the label and color of the curve.
    ag_data[var[i]].plot(label='Agressive', color='red')
    lax_data[var[i]].plot(label='Lax', color='green')
    ob_data[var[i]].plot(label='Observed', color='blue')
      
      
    # adding Label to the x-axis
    plt.xlabel('Month')
      
    # adding legend to the curve
    plt.legend()
    

ob_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Model Output\Observed-2023-02-08-12-42-11.csv")
lax_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Model Output\Lax-2023-02-08-11-58-15.csv")
ag_data=pd.read_csv(r"C:\Users\mahon\Documents\Python Scripts\NN Practice\CF_Paths\Model Output\Aggressive-2023-02-08-13-00-30.csv")

ag_data['Cons']=ag_data['Cons']/ob_data['Cons'][0]
lax_data['Cons']=lax_data['Cons']/ob_data['Cons'][0]
ob_data['Cons']=ob_data['Cons']/ob_data['Cons'][0]



# to set the plot size
plt.figure(figsize=(10, 8), dpi=250)
 
# using plot method to plot open prices.
# in plot method we set the label and color of the curve.
ag_data['Cons'].plot(label='Agressive', color='red')
lax_data['Cons'].plot(label='Lax', color='green')
ob_data['Cons'].plot(label='Observed', color='blue')
 
 
# adding Label to the x-axis
plt.xlabel('Month')
 
# adding legend to the curve
plt.legend()


























