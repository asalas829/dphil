# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:12:00 2015

@author: asalas
"""

from __future__ import division

import pandas as pd
import adabypass as aby
from scipy.stats import norm

# Load the data
filepath = 'Data/fx_30min_data.csv'
data = pd.read_csv(filepath)

Y_raw = data['EURUSD']
mean = Y_raw.mean()
std =  Y_raw.std()
T = len(Y_raw)

# Inputs/outputs
Y = (Y_raw - mean) / std
X = pd.DataFrame({'cst': [1.0] * T, 'Y_lag': Y.shift()}).fillna(0) # AR(1)

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
m = mean + std * adabypass_res['pred_mean']
V = std**2 * adabypass_res['pred_var']

# Performance metrics
e_hat = Y_raw - m
bip = 20 # burn-in period
e_hat_bip = e_hat[bip:]
V_bip = V[bip:]
T_bip = len(e_hat_bip)

rmse = aby.ago.sqrt(T**-1 * aby.ago.np.dot(e_hat, e_hat))
rmse_bip = aby.ago.sqrt(T_bip**-1 * aby.ago.np.dot(e_hat_bip, e_hat_bip))

mad = aby.ago.np.median(aby.ago.np.abs(e_hat))
mad_bip = aby.ago.np.median(aby.ago.np.abs(e_hat_bip))

mae = aby.ago.np.abs(e_hat).mean()
mae_bip = aby.ago.np.abs(e_hat_bip).mean()

ll = aby.ago.np.log(norm.pdf(e_hat, loc = 0, scale = aby.ago.np.sqrt(V))).sum()
ll_bip = aby.ago.np.log(norm.pdf(e_hat_bip, loc = 0, scale = aby.ago.np.sqrt(V_bip))).sum()

print('rmse = ', rmse_bip)
print('mad = ', mad_bip)
print('mae = ', mae_bip)
print('ll = ', ll_bip)