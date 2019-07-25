#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:33:30 2019

@author: cdholmes
"""

import numpy as np
from scipy.stats import linregress
from collections import namedtuple

def york( x, y, sigx=1, sigy=1, rxy=0 ):
    """ York regression accounting for error in x and y
    Follows the notation and algorithm of York et al. (2004) Section III
    
    Arguments
    x, y       : arrays of points for fitting
    sigx, sigy : errors or uncertainty in x and y
    rxy        : correlation coefficient for errors in x and y 
        sigx, sigy, and rxy can be constants or arrays of the same length as x and y
    
    Returns
    result  : Dict containing
    """
    
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
    for i in range(50):

        # Weight for point i
        W = wx * wy / ( wx + b**2 * wy - 2 * b * rxy * alpha )

        # Weighted means        
        Xbar = np.sum( W * x ) / np.sum( W )
        Ybar = np.sum( W * y ) / np.sum( W )
        
        # Deviation from weighted means
        U = x - Xbar
        V = y - Ybar
        
        # 
        beta = W * ( U / wx + b*V / wx - (b*U + V) * rxy / alpha )
        
        # Update slope estimate
        bnew = np.sum( W * beta * V ) / np.sum( W * beta * U )
    
        # Break from loop if new value is  
        if (np.abs( (bnew-b)/b ) < rtol ):
            break
        else:
            b = bnew
    
    # Intercept
    a = Ybar - b * Xbar
    
    # least-squares adjusted points, expectaion values of X and Y
    xa = Xbar + beta 
    ya = Ybar + b*beta
    
    # Mean of adjusted points
    xabar = np.sum( W * xa ) / np.sum( W )
    
    # Devaiation of adjusted points from their means
    u = xa - xabar
    v = ya - yabar
    
    # Variance of slope and intercept estimates
    varb = 1 / np.sum( W * u**2 )
    vara = 1 / np.sum( W ) + xabar**2 * varb 
    
    # Standard error of slope and intercept
    siga = np.sqrt( vara )
    sigb = np.sqrt( varb )
    
    # Define a named tuple type that will contain the results
    result = namedtuple( 'result', 'slope intercept sigs sigi params sigma params sigma' )
    
    # Return results as a named tuple, User can access as a regular tuple too
    return result( b, a, sigb, siga, [b,a], [sigb, siga] )
