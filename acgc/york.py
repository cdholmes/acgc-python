#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:33:30 2019

@author: cdholmes
"""

from collections import namedtuple
import numpy as np
from scipy.stats import linregress

def york( x, y, sigx=1, sigy=1, rxy=0 ):
    '''York regression accounting for error in x and y
    Follows the notation and algorithm of York et al. (2004) Section III
    
    Parameters
    ----------
    x, y : ndarray
        dependent (x) and independent (y) variables for fitting
    sigx, sigy : ndarray (default=1)
        errors or uncertainty corresponding to x and y
    rxy : float (default=0)
        correlation coefficient for errors in x and y, 
        default to rxy=0 meaning that the errors in x are unrelated to errors in y
        sigx, sigy, and rxy can be constants or arrays of the same length as x and y
    
    Returns
    -------
    result : dict with the following keys
        b      = slope estimate
        a      = intercept estimate
        sigs   = standard error of slope estimate
        sigi   = standard error of intercept estimate
        params = [b,a] as array 
        sigma  = [sigs,sigi] as array 
    '''

    # relative error tolerance required for convergence
    rtol = 1e-15

    # Initial guess for slope, from ordinary least squares
    result = linregress( x, y )
    b = result[0]

    # Weights for x and y
    wx = 1 / sigx**2
    wy = 1 / sigy**2

    # Combined weights
    alpha = np.sqrt( wx * wy )

    # Iterate until solution converges, but not more 50 times
    maxiter=50
    for i in range(1,maxiter):

        # Weight for point i
        W = wx * wy / ( wx + b**2 * wy - 2 * b * rxy * alpha )
        Wsum = np.sum( W )

        # Weighted means        
        Xbar = np.sum( W * x ) / Wsum
        Ybar = np.sum( W * y ) / Wsum

        # Deviation from weighted means
        U = x - Xbar
        V = y - Ybar

        # parameter needed for slope
        beta = W * ( U / wy + b*V / wx - (b*U + V) * rxy / alpha )

        # Update slope estimate
        bnew = np.sum( W * beta * V ) / np.sum( W * beta * U )

        # Break from loop if new value is very close to old value
        if np.abs( (bnew-b)/b ) < rtol:
            break
        else:
            b = bnew

    if i==maxiter:
        raise ValueError( f'York regression failed to converge in {maxiter:d} iterations' )

    # Intercept
    a = Ybar - b * Xbar

    # least-squares adjusted points, expectation values of X and Y
    xa = Xbar + beta 
    ya = Ybar + b*beta

    # Mean of adjusted points
    xabar = np.sum( W * xa ) / Wsum
    yabar = np.sum( W * ya ) / Wsum

    # Devaiation of adjusted points from their means
    u = xa - xabar
    v = ya - yabar

    # Variance of slope and intercept estimates
    varb = 1 / np.sum( W * u**2 )
    vara = 1 / Wsum + xabar**2 * varb

    # Standard error of slope and intercept
    siga = np.sqrt( vara )
    sigb = np.sqrt( varb )

    # Define a named tuple type that will contain the results
    result = namedtuple( 'result', 'slope intercept sigs sigi params sigma' )

    # Return results as a named tuple, User can access as a regular tuple too
    return result( b, a, sigb, siga, [b,a], [sigb, siga] )
