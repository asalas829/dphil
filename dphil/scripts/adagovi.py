# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:42:39 2015

@author: Arnold
"""

from __future__ import division

from math import sqrt
import numpy as np
from scipy.special import k0, kn
from scipy.stats import truncnorm
from scipy.optimize import fixed_point

def moments_w(x, y, moments_w_prev, theta):
    """
    Arguments
    ----------
    theta: array of model parameters
           s.t. theta[0]: alpha, theta[1]: beta, theta[2]: mu, theta[3]: varmu
    
    """
    try:    
        I = len(x)
    except TypeError:
        I = 1
    
    idmat_I = np.eye(I) 
    
    P_w = moments_w_prev['cov'] + theta[0]**-1 * idmat_I
    V = np.dot(np.dot(x, P_w), x) + theta[1]**-1
    k_w = np.dot(P_w, x) / V
    
    m = np.dot(x, moments_w_prev['mean']) + theta[2]
    e = y - m
    moments_w_mean = moments_w_prev['mean'] + e * k_w
    
    tmp = idmat_I - np.outer(k_w, x)
    moments_w_cov = np.dot(np.dot(tmp, P_w), tmp.T) + theta[1]**-1 * np.outer(k_w, k_w) # Joseph's form
    moments_w_cov = .5 * (moments_w_cov + moments_w_cov.T) # to guarantee symmetry
    
    return {'mean': moments_w_mean, 'cov': moments_w_cov}

def theta_govi(theta, omega, x, y, moments_w_prev):
    """
    Returns mean variational parameters
    
    Arguments
    ---------
    psi: dictionary of hyperparameters
    
    """   
    moments_w_curr = moments_w(x, y, moments_w_prev, theta)
    mu_w_curr = moments_w_curr['mean']
    Sigma_w_curr = moments_w_curr['cov']
    eta_hat = y - np.dot(x, mu_w_curr)
    
    # Variational mean of ALPHA
    mean_delta_w = mu_w_curr - moments_w_prev['mean']
    cov_delta_w = Sigma_w_curr - moments_w_prev['cov']
    mean_euclid_w = np.dot(mean_delta_w, mean_delta_w) + sum(np.diag(cov_delta_w))
    
    alpha_shape = omega['a']
    alpha_rate = omega['b'] + .5 * mean_euclid_w
    alpha_govi = alpha_shape / alpha_rate
    
    # Variational mean of BETA
    rho = (eta_hat - theta[2])**2 + np.dot(np.dot(x, Sigma_w_curr), x) + theta[3]
    try:    
        rho_sqrt = sqrt(rho)
    except:
        rho_sqrt = 1e-08
    num = k0(rho_sqrt)
    denom = rho_sqrt * kn(1, rho_sqrt)
    beta_govi = num / denom
    
    # Variational mean and variance of MU
    try:    
        beta_sqrt = sqrt(theta[1])
    except:
        beta_sqrt = 1e-08
    l = -beta_sqrt * (omega['epsilon'] + eta_hat)
    u = beta_sqrt * (omega['epsilon'] - eta_hat)    
    mu_govi = truncnorm.mean(l, u, loc = eta_hat, scale = 1/beta_sqrt)
    varmu_govi = truncnorm.var(l, u, loc = eta_hat, scale = 1/beta_sqrt)
    
    # Function output
    return np.array([alpha_govi, beta_govi, mu_govi, varmu_govi])
    
def theta_opt(theta_prev, omega, x, y, moments_w_prev):
    ext = (omega, x, y, moments_w_prev)
    return fixed_point(theta_govi, theta_prev, args = ext)
    
def hypergrads(Sigma_w_prev, theta, x, post_err, hgrads_prev):
    try:    
        I = len(x)
    except TypeError:
        I = 1
    
    idmat_I = np.eye(I) 
    
    P_w = Sigma_w_prev + theta[0]**-1 * idmat_I
    V = np.dot(np.dot(x, P_w), x) + theta[1]**-1
    g = V**-1 * np.dot(P_w, x)
    tmp = idmat_I - np.outer(g, x)
    
    S_prev = hgrads_prev['cov']
    psi_prev = hgrads_prev['mean']
    
    S = np.dot(np.dot(tmp, S_prev), tmp.T)
    psi = np.dot(tmp, psi_prev) + theta[1] * post_err * np.dot(S, x)   
    
    return {'mean': psi, 'cov': S}