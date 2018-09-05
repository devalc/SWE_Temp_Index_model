# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 18:52:27 2018

@author: Chinmay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


data = pd.read_csv('D:/Chinmay/Hydro_modeling_erin_fall_2018/snotel_623.csv', nrows = 6954, header=0)
data = data.dropna()

Day = data.ix[:,0]
SWE_obs = data.ix[:,1]
P_accum = data.ix[:,2]
T_max = data.ix[:,3]
T_min = data.ix[:,4]
T_avg = data.ix[:,5]


#p0 = [tbase, train, tsnow,k]
p0 = [0,3,0,2.29]

def SWE_usace(p):
    if (T_avg - p0[0])*p0[3] > 0:
        melt = (T_avg - p0[0])*p0[3]
    else:
        melt = 0