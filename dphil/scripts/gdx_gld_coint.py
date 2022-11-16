# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:41:00 2015

@author: asalas
"""

from __future__ import division

from datetime import date
# import pandas.io.data as web
import adabypass as aby
import pandas as pd
import os

# # Load the data
# tickers = ['GLD', 'GDX']
# start = date(2006, 5, 22)
# end = date.today()
#
# all_data = {}
#
# for ticker in tickers:
#     all_data[ticker] = web.get_data_yahoo(ticker, start, end)
#
# adj_close = pd.DataFrame(dict(gld=all_data['GLD']['Adj Close'],
#                               gdx=all_data['GDX']['Adj Close']))

data_dir = '/home/autarkydotai/asalas/dphil/data'
start = '05-22-2006'
end = '04-22-2015'
symbols = ['GDX', 'GLD']

all_data = {symbol: pd.read_csv(os.path.join(data_dir, '{}.csv'.format(symbol)), index_col=0, parse_dates=True)
           for symbol in symbols}

adj_close = pd.DataFrame({symbol: data['Adj Close'] for symbol, data in all_data.items()})
adj_close.columns = [col.lower() for col in adj_close.columns]
adj_close = adj_close.loc[start:end]

# Extract the inputs and outputs
Y = aby.ago.np.log(adj_close['gld'])
X = aby.ago.np.log(adj_close['gdx'])
# print(X)
T = len(Y)

# Initial hyperparameters
#C = 1.0
#a = 1e-3
#b = 1e-3
#epsilon = 0.1
#omega = {'a': a, 'b': b, 'epsilon': epsilon}
#omega_min = omega
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
spreads = adabypass_res['pred_err']
V = adabypass_res['pred_var'].ravel()
hedge_ratios = adabypass_res['weights'].ravel()

# Pairs-trading strategy
k = 0.1

#delta = 0.025
#longs_entry = spreads <= -delta
#longs_exit = spreads > -delta
#shorts_entry = spreads >= delta
#shorts_exit = spreads < delta

long_signals = (spreads / aby.ago.np.sqrt(V) <= -k).astype(int)
short_signals = -(spreads / aby.ago.np.sqrt(V) >= k).astype(int)
signals = long_signals + short_signals

positions = pd.DataFrame(dict(gld=signals,
                              gdx=-signals * hedge_ratios),
                         index=adj_close.index)

pnl = (positions.shift() * adj_close.diff()).sum(axis = 1) # daily PnL
pnl[0] = 100
wealth = pnl.cumsum()
ret = wealth.pct_change()
ret.fillna(0, inplace=True)
cum_ret = (1 + ret).cumprod() - 1

cum_ret.plot()

# Performance metrics
apr = ((1 + ret).prod())**(252/len(ret)) - 1
sharpe = aby.ago.np.sqrt(252) * ret.mean() / ret.std()

print('apr = ', apr)
print('sharpe = ', sharpe)
