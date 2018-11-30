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

#p0 = [tbase, train, tsnow,k]
p0 = [0,3,0,2.29]
filepath= 'snotel_623_pre_mgmt.csv'


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

op.minimize(rmse, p0[3])
