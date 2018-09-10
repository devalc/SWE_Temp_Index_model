# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 18:52:27 2018

@author: Chinmay Deval
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#p0 = [tbase, train, tsnow,k]
p0 = [0,3,0,2.29]
filepath= 'D:/Chinmay/Hydro_modeling_erin_fall_2018/snotel_623_pre_mgmt.csv'
SNOF = 1.55
Tscrit = 2.9

def process_data(filepath):
    data = pd.read_csv(filepath, nrows = 6954,parse_dates=['Date'], skiprows=1,\
                       names= ['Date', 'SWE_obs','P_accum','T_max','T_min',\
                               'T_avg','P_incremental']).dropna()
    P = data['P_accum'].values
    T_avg = data['T_avg'].values
    SWE_obs= data['SWE_obs'].values
    return data, P, T_avg, SWE_obs

    

def SWE_usace(p0, T_avg):
    dt = T_avg.size
    Mf  = np.zeros(dt)
    Ps  = np.zeros(dt)
    SWE = np.zeros(dt)
    for t in np.arange(1,dt,1):
        if (SWE_obs[t-1] > 0.0) & (T_avg[t] > p0[0]):
            Mf[t] = np.minimum((p0[3]*(T_avg[t] - p0[0])), SWE_obs[t-1])
        if (P[t]>0.0):
            if (T_avg[t]-Tscrit):
                Ps[t] = SNOF*P[t]
            else:
                Ps[t] = 0.0
    
        SWE[t] = SWE[t-1] + Ps[t] - Mf[t]
        
    return SWE
            

def plot_res():
     fig= plt.figure(figsize=(14,24))
     plt.subplot(411)
     plt.plot(data['Date'],P,'r-')
     plt.ylabel('Precipitation [mm]')
     plt.subplot(412)
     plt.plot(data['Date'],T_avg,'r-')
     plt.ylabel('Temperature [${}^\circ$C]')
     plt.subplot(413)
     plt.plot(data['Date'],Mf,'b-')
     plt.ylabel('Melt flux [mm/day]')
     plt.subplot(414)
     plt.plot(data['Date'],SWE,'b-')
     plt.plot(data['Date'],SWE_obs,'ko')
     plt.xlabel('Date')
     plt.ylabel('SWE [mm]')
     plt.legend(('Modeled SWE','Observed SWE'))
     return fig
    
    
    
    
    data = datain
    data['SWE.melt'] = np.where((data['Air Temperature Average (degC)']-p0[0])*p0[3] <=0, \
        0,(data['Air Temperature Average (degC)']-p0[0])*p0[3])
    
    data['SWE.sim'] = np.where((data['Air Temperature Average (degC)']<p0[1] & data['Air Temperature Average (degC)'] > p0[2]),
                        data['P']*((data['Air Temperature Average (degC)']-p0[2])/(p0[1]-p0[2])),0)
    return data


data, P, T_avg, SWE_obs = process_data(filepath)

melt = SWE_usace(p0,data)

#    
#    if (T_avg - p0[0])*p0[3] <= 0:
#        swe_melt = 0
#    else:
#        swe_melt = (T_avg - p0[0])*p0[3]
#        return swe_melt
#        