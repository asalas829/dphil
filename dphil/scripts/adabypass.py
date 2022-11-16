# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:12:00 2015

@author: asalas
"""

from __future__ import division

import adagovi as ago

def adabypass(X, Y, omega, omega_min, theta_init, C):
    T = len(Y) # number of examples
    try:
        I = len(X.columns) # number of features
    except AttributeError:
        I = 1

    m = ago.np.empty(T) # predictive means
    V = [] # predictive variances

    e_hat = ago.np.empty(T) # a priori errors

    theta = ago.np.empty((T, 4)) # mean variational parameters
    theta[0] = theta_init
    theta_old = theta[0]

    # Initialise weight mean/covariance and hyperparameter gradients
    w = ago.np.empty((T, I))
    # w[0] = 0.0
    # w[0] = 1.6766
    # w[0] = -2
    # w[0] = -10
    # w[0] = -1
    # w[0] = 1
    # w[0] = 2
    w[0] = 10
    Sigma_w0 = ago.np.zeros((I, I))
    moments_w_old = {'mean': w[0], 'cov': Sigma_w0}

    idmat_I = ago.np.eye(I)
    P = Sigma_w0 + theta_old[0]**-1 * idmat_I

    hgrads = {'mean': ago.np.zeros(I), 'cov': idmat_I}

    # ADA-BYPASS main
    for t in range(T):
        if t > 0:
            w[t] = w[t-1]
            theta_old = theta[t-1]
            P = Sigma_w + theta_old[0]**-1 * idmat_I
            moments_w_old = {'mean': w[t], 'cov': Sigma_w}

        # x = ago.np.array(X.ix[t]) # current input vector
        x = ago.np.array(X.iloc[t])
        m[t] = ago.np.dot(x, w[t]) + theta_old[2]
        V.append(ago.np.dot(ago.np.dot(x, P), x) + theta_old[1]**-1)

        # Observe y(t)
        y = Y[t]
        e_hat[t] = y - m[t] # a priori error

        # Update the hyperparameters
        tmp = ago.np.dot(x, hgrads['mean'])
        stdesc_dir = tmp * e_hat[t]

        for key in omega.keys():
            omega[key] = omega[key] + C * theta_old[1] * stdesc_dir
            omega[key] = max(omega[key], omega_min[key])

        # Update model params
        try:
            theta[t] = ago.theta_opt(theta_init, omega, x, y, moments_w_old)
        except:
            try:
                theta[t] = ago.theta_opt(theta_old, omega, x, y, moments_w_old)
            except:
                theta[t] = theta_old

        #theta[t, 1] = max(theta[t, 1], theta_init[1])

        # Update weight moments
        moments_w_new = ago.moments_w(x, y, moments_w_old, theta[t])
        w[t] = moments_w_new['mean']
        Sigma_w = moments_w_new['cov']

        # Update the gradients:
        mu_diff = theta[t, 2] - theta_old[2]
        adj = e_hat[t] - mu_diff
        hgrads = ago.hypergrads(moments_w_old['cov'], theta[t], x, adj, hgrads)

    return {'pred_mean': m, 'pred_var': ago.np.array(V), 'pred_err': e_hat, 'weights': w, 'theta': theta}
