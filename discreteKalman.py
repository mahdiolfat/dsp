#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 21:42:49 2017

@author: mahdiolfat
"""

# iterate n from 0 - n and build matrices 
# x(n|n-1) = A(n-1)x(n-1|n-1) 
# P(n|n-1) = A(n-1)P(n-1|n-1)A^H(n-1) + Q_w(n)
# K(n) = P(n|n-1)C^H(n)[C(n)P(n|n-1)C^H(n)+Q_v(n)]^-1     
# x(n|n) = x(n|n-1) + K(n)[y(n) - C9n)x(n|n-1)] 
# P(n|n) = [I-K(n)C(n)]P(n|n-1) 

#TODO: chaing the dot products

import sys
import numpy as np
import matplotlib.pyplot as plt
import math

SIGMA_V = 0.64
SIGMA_W = 1

a_1 = -0.1
a_2 = -0.09
a_3 = 0.648

N = 10

print('let\'s get it')

# TODO: function to calculate the Kalman filter

def kalmanGain(P, C, Q_v):    
    # TODO: assuming C = 1
    
    # TODO: check for sizes (bound check)

    #K_2_I = np.dot(np.dot(C, P), C.getH()) + SIGMA_V
    K_2_I = P + Q_v
    K_2 = K_2_I.getI()
    
    #K_1 = np.dot(P, C.getH())
    K_1 = P
    
    K = np.dot(K_1, K_2)
#==============================================================================
#     print("K_2_I = " + str(K_2_I))
#     print("K_1 = " + str(K_1))
#     print("K = " + str(K))
#==============================================================================
    
    return K;

A = np.matrix('-0.1 -0.09 0.648; 1 0 0; 0 1 0')
print(A)

Atest = np.matrix('0.8')
print(Atest)

print()

print()

C_list = [1]

# initialize recursion

# x_e(0) = E{x(0)}
# P(0) = E{|x(0)|^2}

# list to collect past covariance terms
P = []
X_est = []
Y = []
K = []

# initialize P(0|0) = E{|x(0)|^2}
#X_0 = np.matrix('1; 0; 0')
X_0_exp = np.matrix('1; 0; 0')
P_0 = np.dot(X_0_exp, X_0_exp.getH())
P.append(P_0)
print(P_0)
print()

X_est_0 = np.matrix('0; 0; 0')

# initialize estimator x_est(0|0) = E{x(0)}
X_est.append(X_est_0)

MODEL_ORDER = 3

A_H = A.getH()

# Generate the processes x(n) and y(n)
proc_X = []

w_n = np.random.normal(0, SIGMA_W, N)
v_n = np.random.normal(0, SIGMA_V, N)

print("Wn")
print(w_n)
print()

print("Vn")
print(v_n)
print()

def procX(n):
    if (n < 0):
        return 0
    
    return a_1*procX(n-1) + a_2*procX(n-2) + a_3*procX(n-3) + w_n[n]

for n in range(0, N):
    newX = procX(n)
    proc_X.append(newX)
    
pX = []
pX.append(w_n[0])
pX.append(a_1*pX[0] + w_n[1])
pX.append(a_1*pX[1] + a_2*pX[0] + w_n[2])

for n in range(3, N):
    pX.append(a_1*pX[n-1] + a_2*pX[n-2] + a_3*pX[n-3] + w_n[n])

print()
print(proc_X)

print()
print(pX)

pY = []
for n in range(0, N):
    pY.append(pX[n] + v_n[n])
    
print()
print(pY)

#print("P(n|n-1)\t\tK(n)\t\tP(n|n)")
for n in range(1, N):
    P_n_1 = np.dot(np.dot(A, P[n-1]), A_H) + SIGMA_W
    
    Kgain = kalmanGain(P_n_1, 1, SIGMA_V) 
    K.append(Kgain)
    
    P_n = np.dot((np.identity(MODEL_ORDER)-Kgain), P_n_1)
    P.append(P_n)
    
    X_n_1 = np.dot(A, X_est[n-1])
    X_n = X_n_1 + np.dot(Kgain, (pY[n] - X_n_1))
    
    X_est.append(X_n)

    print()
    #print(str(P_n_1) + '\t\t' + str(Kgain) + '\t\t' + str(P_n))    
#==============================================================================
#     print("P(n|n-1) = ")
#     print(P_n_1)
# 
#     print("K(n) = ")
#     print(Kgain)
# 
#     print("P(n|n-1) = ")
#     print(P_n)
#     
#     print()
#==============================================================================
estimator = []
# TODO: figure out what the first estimator is - cannot be 0?
estimator.append(1)
estimator.append(X_est[0].item((0, 0)) * estimator[0])
estimator.append(X_est[1].item((0, 0)) * estimator[1] + 
                 X_est[1].item((1, 0)) * estimator[0])

for n in range(3, N):
    estimator.append(X_est[n].item((0, 0)) * estimator[n-1] +
                     X_est[n].item((1, 0)) * estimator[n-2] +
                     X_est[n].item((2, 0)) * estimator[n-3])
    
print(estimator)
for est in X_est:
    print(est)
    print()
    
x_axis = np.arange(0, N, 1)
plt.plot(x_axis, pX) # blue
plt.plot(x_axis, pY) # orange
plt.plot(x_axis, estimator) # green
