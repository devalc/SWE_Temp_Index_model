# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:51:29 2018

@author: Chinmay Deval
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_data(filepath):
    data = pd.read_csv(filepath, nrows = 10214,parse_dates=['Date'], skiprows=1,\
                       names= ['Date', 'SWE_obs','P_accum','T_max','T_min',\
                               'T_avg','P_incremental']).dropna()
    P = data['P_incremental'].values
    T_avg = data['T_avg'].values
    SWE_obs= data['SWE_obs'].values
    return data, P, T_avg, SWE_obs

def Snowmelt_DD_usace(p0, P, T,  Tsnow,Train, k, Tbase=0.0):
    #melt
    melt= k*(T-Tbase)
    melt[melt<0.0]=0.0
    # Psnow and Prain
    Psnow= np.where(T < Train, np.where(T>Tsnow, np.multiply(P, np.divide(np.subtract(T,Tsnow),\
                                                                (Train-Tsnow))), P), 0.0)
    Prain = np.subtract(P, Psnow)
    
    return melt, Psnow, Prain


def simSWE(Psnow, meltflux):
    cumSWE = np.zeros(Psnow.shape, dtype=np.float32)
    act_melt = np.zeros(meltflux.shape, dtype=np.float32)
    for t in range(1,Psnow.shape[0]):
        swe_inc = cumSWE[t-1] + Psnow[t] - meltflux[t]
        
        if swe_inc > 0.0:
            cumSWE[t] = swe_inc
            act_melt[t] = meltflux[t]
        else:
            act_melt[t] = cumSWE[t-1]

    return cumSWE, act_melt


def md(sim, obs):
    """
    Mean difference 

    """
    return(np.mean(np.subtract(sim, obs)))

def nse(sim, obs):
    """
    Nash-Sutcliffe Efficiency

    """
    obs_mod2 = np.sum(np.square(np.subtract(obs, sim)))
    obs_mean2 = np.sum(np.square(np.subtract(obs, np.mean(obs))))
    nse = 1-(obs_mod2/obs_mean2)
    return nse

def rmse(sim, obs):
    """
    
    """
    return(np.nanmean(np.sqrt(np.square(np.subtract(sim, obs)))))


def plot_res(data):
    """
    takes in dataframe with date and calculated variables
    
    returns plots of P, T, Meltflux, and Observed and Predicted SWE
    """
    fig= plt.figure(figsize=(14,24))
    plt.subplot(411)
    plt.plot(data.iloc[:,0],data.iloc[:,6],'r-')
    plt.ylabel('Precipitation [mm]')
    plt.subplot(412)
    plt.plot(data.iloc[:,0], data.iloc[:,5],'r-')
    plt.ylabel('Temperature [${}^\circ$C]')
    plt.subplot(413)
    plt.plot(data.iloc[:,0],data.iloc[:,11],'b-')
    plt.ylabel('Melt flux [mm/day]')
    plt.subplot(414)
    plt.plot(data.iloc[:,0],data.iloc[:,7],'b-')
    plt.plot(data.iloc[:,0],data.iloc[:,1],'r-')
    plt.xlabel('Date')
    plt.ylabel('SWE [mm]')
    plt.legend(('Modeled SWE','Observed SWE'))
    
    return fig.savefig('plots/SWE_tmp_index.png', dpi=fig.dpi)

