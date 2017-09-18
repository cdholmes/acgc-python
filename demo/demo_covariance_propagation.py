#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:11:48 2017

@author: cdholmes
"""

import numpy as np
import sympy as sym

n = 10000
x1 = np.random.randn(n)+10
x2 = np.random.randn(n)

y1 = x1
y2 = np.random.randn(n)+2

def f(x,y):
    a = 3; b = 5
    return a*x+b*y

def g(x,y):
    return x*y


from sympy.abc import w,x,y,z

fvals = dict(w=np.mean(x1),x=np.mean(x2))
gvals = dict(y=np.mean(y1),z=np.mean(y2))

# analytical expected covariance
cov = np.float64(0)
for var1 in [w,x]:
    for var2 in [y,z]:
        cov += sym.diff( f(w,x), var1 ).subs( fvals ) * \
            sym.diff( g(y,z), var2 ).subs( gvals ) * \
            np.cov( var1.subs(fvals), var2.subs(gvals) )
            
        

# numerical value of covariance
print(np.cov(f(x1,x2),g(y1,y2))[0,1])
print(3*np.mean(1/x1)*np.var(x1))
print(cov)