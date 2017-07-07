# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:24:09 2016

@author: cdholmes
"""

# Propagate errors/uncertainty through any user-defined formula
# User must specify values and standard errors of each input variable
# and the formula to combine them.
# Program will calculate value and uncertainty in the final result.
#
# Result will be exact for linear combinations.
# For non-linear combinations, the approximation is reasonable for small errors
# In all cases, variables x,y,z etc are assumed to be independent (i.e. covariance=0)

import sympy as sym

# Define all symbolic variables that you will use
x, y, z = sym.symbols('x y z')
syms = [x, y, z]

# Values of each variable
vals = dict(x=1.,  y=2., z=3.)

# Standard deviation of each variable
errs = dict(x=0.1, y=0.2, z=0.1)


# Define all formulas
y = z**2 + sym.sin(x)
# f is the variable whose error will be calculated
f = x*y

############################################
# No need to change code below

# variance of f is Sum_i (df/xi)**2 var(xi)
# Assumes cov(xi,xj) = 0
V = 0
for s in syms:
    V += sym.diff(f,s).subs(vals)**2 * s.subs(errs)**2

# Print results
print('f={:10.4f}+/-{:10.4f}'.format(f.subs(vals),sym.sqrt(V)))
