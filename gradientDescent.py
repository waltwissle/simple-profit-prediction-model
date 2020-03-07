# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 00:16:59 2020

@author: Walter Aburime
"""
import numpy as np
import costFunction

def gradientDescent(X, Y, theta, alpha, iters):
    m = len(Y)
    J_history = np.zeros((iters,1))
    
    for run in range(iters):
        h_theta = X @ theta
        theta = theta - ((alpha/m) * X.T @ (h_theta - Y))
        J = costFunction.costFunction(X, Y, theta)
        
        J_history[run] = costFunction.costFunction(X, Y, theta)
    
    return theta, J_history