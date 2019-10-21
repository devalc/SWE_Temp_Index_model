# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 18:52:27 2018

@author: Chinmay Deval
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from scipy import optimize as op

#p0 = [tbase, train, tsnow, k]
#p0 = [0,3,0,2.29]
p0 = [0.2486748, 2.4000002, -0.8000008, 2.3418852] ### taken from R optim for now. TBD: implement optimization within this python script
filepath= 'data/snotel_623.csv'


data, P, T_avg, SWE_obs = process_data(filepath)

melt,Psnow,Prain = Snowmelt_DD_usace(p0, P,T_avg, Tsnow=p0[2], Train=p0[1],k=p0[3])

cumSWE, act_melt = simSWE(Psnow, melt)


data['cumSWE']= cumSWE[:]
data['melt']= melt[:]
data['Psnow']= Psnow[:]
data['Prain']= Prain[:]
data['act_melt']= act_melt[:]

md = md(SWE_obs, cumSWE)
rmse= rmse(SWE_obs, cumSWE)
nse= nse(SWE_obs, cumSWE)


plot_res(data)


## plots Precip and Psnow

fig= plt.figure()
plt.plot(data.iloc[:,0],data.iloc[:,6],'b-')
plt.plot(data.iloc[:,0],data.iloc[:,9],'r-')
#plt.plot(data.iloc[:,0],data.iloc[:,10],'g-')
plt.xlabel('Date')
plt.ylabel('Precip(mm)')
plt.legend(('Precip','Psnow', 'Prain'))
fig.savefig('plots/Precip_and_Psnow.png', dpi=400)   

## save dataframe after model calculations

data.to_excel("Output_dataframe/obs_mod_swe.xlsx")