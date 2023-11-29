#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:23:38 2019

@author: cdholmes
"""
import numpy as np

def funcerror_uncorr(func,x0,xerr,h=None,**kwargs):
    '''Propagate uncertainties in function parameters to uncertainty in function value. 
    
    The uncertainty will be evaluated at f(x0). The method uses 
    a numerical estimate of the derivative of f(x0). For non-linear functions
    the method is appropriate for small relative errors (i.e. small xerr/x0).
    The method neglects possible error correlations between multiple parameters 
    in x0.
    The function variables must all be continuous variables 
    (no categorical or discrete variables).
    
    Example. To find the uncertainty in f(x,y) at for values x=1, y=2, where 
    the uncertainty (e.g. standard error) in x is 0.1 and the uncertainty in y 
    is 0.3...
    $ print( funcerr_uncorr( f, [1,2], [0.1,0.3] ) )
    
    Parameters
    ----------
    func : function handle
        function to be evaluated
    x0   : list or tuple
        parameter values for func
    xerr : list or tuple
        uncertainty in parameter values x0
    h    : float (default=1e-3)
        fractional perturbation to x0 used to estimate slope of func via finite difference 
    
    Returns
    -------
    ferr : float
        the uncertainty in f(x0) given xerr uncertainty in x0
    '''

    # Increment size for finite difference
    # Numerical Methods in C recommends h * x, where x is the variable and 
    # h = sqrt(machine precision). For a 64-bit (double) float, that implies 
    # h = 2e-8. For a 32-bit (single) float, that implies h = 3e-4.
    # We choose a larger value to reduce roundoff error. Note 
    # that very large h causes formula error where secant line is not parallel 
    # to instantaneous slope.
    if h is None:
        h = 1e-7
        h = 1e-3

    ferr = np.float64(0)

    for i,xi in enumerate(x0):
        xplus     = x0.copy()
        xminus    = x0.copy()
        xplus[i]  = xplus[i]  + h*xi
        xminus[i] = xminus[i] - h*xi

        # Centered finite difference estimate of derivative
        dfdx = ( func(*xplus,**kwargs) - func(*xminus,**kwargs) ) / ( xplus[i]-xminus[i] )

        ferr += dfdx**2 * xerr[i]**2

    ferr = np.sqrt(ferr)

    return ferr
