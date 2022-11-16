# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:12:00 2015

@author: asalas
"""

from __future__ import division

import pandas as pd
import adabypass as aby
from scipy.stats import norm
from calculateMaxDD import calculateMaxDD


# Load the data
#filepath = 'Data/Archive/gdx_gld.csv'
filepath = 'gdx_gld.csv'
colnames = ['gdx', 'gld']
data = pd.read_csv(filepath, names = colnames, skiprows = 1)
T = len(data)

# Extract the inputs and outputs
Y = data['gld']
X = pd.DataFrame({'cst': [1.0] * T, 'gdx': data['gdx']})

# Initial hyperparameters
C = 1e-03 # aggressiveness parameter
a = C**-1
b = aby.ago.sqrt(a / 1000.0)
epsilon = 1.25
omega = {'a': a, 'b': b, 'epsilon': epsilon}
omega_min = {'a': 1e-08, 'b': 1e-08, 'epsilon': 1e-08}

# Initial mean variational parameters
alpha_init = a / b
beta_init = 500.0
mu_init = 0.0
varmu_init = epsilon**2 * (1 + epsilon / 3.0) / (1 + epsilon)
theta_init = aby.ago.np.array([alpha_init, beta_init, mu_init, varmu_init])

# Implementing ADA-BYPASS
adabypass_res = aby.adabypass(X, Y, omega, omega_min, theta_init, C)
m = adabypass_res['pred_mean']
e_hat = adabypass_res['pred_err']
V = adabypass_res['pred_var']
w = adabypass_res['weights']
pd.DataFrame(w, columns=X.columns).to_csv('gdx_gld_weights.csv')

# Pairs-trading strategy
k = 0.1
longs_entry = e_hat / aby.ago.np.sqrt(V) < -k
longs_exit = e_hat / aby.ago.np.sqrt(V) > -k
shorts_entry = e_hat / aby.ago.np.sqrt(V) > k
shorts_exit = e_hat / aby.ago.np.sqrt(V) < k

num_units_long = aby.ago.np.empty(T)
num_units_short = aby.ago.np.empty(T)

num_units_long[0] = 0
num_units_long[longs_entry] = 1
num_units_long[longs_exit] = 0

num_units_short[0] = 0
num_units_short[shorts_entry] = -1
num_units_short[shorts_exit] = 0

num_units = num_units_long + num_units_short
hedge_ratios = w[:, 1]
# capital allocations to each ETF
positions = pd.DataFrame({'gdx': -num_units * hedge_ratios * X['gdx'], 'gld': num_units * Y})
pnl = (positions.shift() * data.pct_change()).sum(axis = 1) # daily PnL
ret = pnl / positions.shift().abs().sum(axis = 1)
ret = ret.fillna(0)
cum_ret = (1 + ret).cumprod() - 1

# Cumulative Sharpe ratio
# cum_ret_mean = pd.expanding_mean(cum_ret)
# cum_ret_std = pd.expanding_std(cum_ret)
cum_ret_mean = cum_ret.expanding().mean()
cum_ret_std = cum_ret.expanding().std()
cum_sharpe = cum_ret_mean / cum_ret_std
cum_sharpe.plot()

# Performance metrics
bip = 20 # burn-in period
ret_bip = ret[bip:]
e_hat_bip = e_hat[bip:]
V_bip = V[bip:]
T_bip = len(e_hat_bip)

apr = ((1 + ret).prod())**(252/len(ret)) - 1
apr_bip = ((1 + ret_bip).prod())**(252/len(ret_bip)) - 1

sharpe = aby.ago.np.sqrt(252) * ret.mean() / ret.std()
sharpe_bip = aby.ago.np.sqrt(252) * ret_bip.mean() / ret_bip.std()

rmse = aby.ago.sqrt(T**-1 * aby.ago.np.dot(e_hat, e_hat))
rmse_bip = aby.ago.sqrt(T_bip**-1 * aby.ago.np.dot(e_hat_bip, e_hat_bip))

mad = aby.ago.np.median(aby.ago.np.abs(e_hat - aby.ago.np.median(e_hat)))
mad_bip = aby.ago.np.median(aby.ago.np.abs(e_hat_bip - aby.ago.np.median(e_hat_bip)))

mae = aby.ago.np.abs(e_hat).mean()
mae_bip = aby.ago.np.abs(e_hat_bip).mean()

ll = aby.ago.np.log(norm.pdf(e_hat, loc = 0, scale = aby.ago.np.sqrt(V))).sum()
ll_bip = aby.ago.np.log(norm.pdf(e_hat_bip, loc = 0, scale = aby.ago.np.sqrt(V_bip))).sum()

max_dd, max_dd_dur, _ = calculateMaxDD(cum_ret)

print('apr = ', round(apr_bip, 4))
print('sharpe = ', round(sharpe_bip, 2))
print('max_dd = ', round(max_dd, 4))
print('max_dd_dur = ', int(max_dd_dur))
print('rmse = ', round(rmse_bip, 2))
print('mad = ', round(mad_bip, 2))
print('mae = ', round(mae_bip, 2))
print('ll = ', round(ll_bip, 2))
# print('apr = ', apr)
# print('sharpe = ', sharpe)
# print('rmse = ', rmse)
# print('mad = ', mad)
# print('mae = ', mae)
# print('ll = ', ll)
