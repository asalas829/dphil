# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:12:00 2015

@author: asalas
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the data
filepath = 'Data/gdx_kf_res.csv'
colnames = ['gdx', 'gld', 'y_hat', 'sd_hat', 'drift', 'slope']
data = pd.read_csv(filepath, names = colnames)
T = len(data)

GDX = data['gdx']
GLD = data['gld']
GDX_GLD = pd.DataFrame({'gdx': GDX, 'gld': GLD})

Y_hat = data['y_hat'] # measurement prediction
e = GLD - Y_hat # measurement prediction error

# Pairs-trading strategy
k = 0.25
longs_entry = e / data['sd_hat'] < -k
longs_exit = e / data['sd_hat'] > -k
shorts_entry = e / data['sd_hat'] > k
shorts_exit = e / data['sd_hat'] < k

num_units_long = np.empty(T)
num_units_short = np.empty(T)
num_units_long = pd.Series(num_units_long)
num_units_short = pd.Series(num_units_short)

num_units_long[0] = 0
num_units_long[1] = 0
num_units_long[longs_entry] = 1
num_units_long[longs_exit] = 0
    
num_units_short[0] = 0
num_units_long[1] = 0
num_units_short[shorts_entry] = -1
num_units_short[shorts_exit] = 0

num_units = num_units_long + num_units_short
hedge_ratios_sjrob = data['slope']
positions = pd.DataFrame({'gdx': -num_units * hedge_ratios_sjrob * GDX, 'gld': num_units * GLD}) # capital allocations to each ETF
pnl = (positions.shift() * GDX_GLD.pct_change()).sum(axis = 1)
ret = pnl / positions.shift().abs().sum(axis = 1)
ret = ret.fillna(0)

sjrob_cum_ret = (1 + ret).cumprod() - 1

# Cumulative Sharpe ratio
sjrob_cum_ret_mean = pd.expanding_mean(sjrob_cum_ret)
sjrob_cum_ret_std = pd.expanding_std(sjrob_cum_ret)
sjrob_cum_sharpe = sjrob_cum_ret_mean / sjrob_cum_ret_std
sjrob_cum_sharpe.plot()

# Performance metrics
bip = 20 # burn-in period
ret_bip = ret[bip:]
e_hat_bip = e[bip:]
V_bip = data['sd_hat'][bip:]
T_bip = len(e_hat_bip)

apr_sjrob = ((1 + ret).prod())**(252/len(ret)) - 1
apr_bip_sjrob = ((1 + ret_bip).prod())**(252/len(ret_bip)) - 1

sharpe_sjrob = np.sqrt(252) * ret.mean() / ret.std()
sharpe_bip_sjrob = np.sqrt(252) * ret_bip.mean() / ret_bip.std()

rmse_sjrob = np.sqrt(T**-1 * np.dot(e, e))
rmse_bip_sjrob = np.sqrt(T_bip**-1 * np.dot(e_hat_bip, e_hat_bip))

mad_sjrob = np.median(np.abs(e - np.median(e)))
mad_bip_sjrob = np.median(np.abs(e_hat_bip - np.median(e_hat_bip)))

mae_sjrob = np.abs(e).mean()
mae_bip_sjrob = np.abs(e_hat_bip).mean()

ll_sjrob = np.log(norm.pdf(e, loc = 0, scale = data['sd_hat'])).sum()
ll_bip_sjrob = np.log(norm.pdf(e_hat_bip, loc = 0, scale = V_bip)).sum()

print('apr_sjrob = ', apr_bip_sjrob)
print('sharpe_sjrob = ', sharpe_bip_sjrob)
print('rmse_sjrob = ', rmse_bip_sjrob)
print('mad_sjrob = ', mad_bip_sjrob)
print('mae_sjrob = ', mae_bip_sjrob)
print('ll_sjrob = ', ll_bip_sjrob)