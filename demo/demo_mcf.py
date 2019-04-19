# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:26:32 2015

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as plt

n0=10
s0=10

a=1.4
kn=0.181
ks=0.181*a
kex=1

dt=0.1
N = 1000
nn = np.zeros(N)
ss = np.zeros(N)

for i in range(N):
    n1 = n0 - kn*n0*dt - kex*(n0-s0)*dt
    s1 = s0 - ks*s0*dt + kex*(n0-s0)*dt
    n0 = n1
    s0 = s1
    nn[i] = n1
    ss[i] = s1

plt.clf()
plt.subplot(2,1,1)
plt.plot(np.arange(N)*dt,nn)
plt.plot(np.arange(N)*dt,ss)
plt.subplot(2,1,2)
plt.plot(np.arange(N)*dt,nn/ss)

    
