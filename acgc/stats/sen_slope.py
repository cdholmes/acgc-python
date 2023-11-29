# -*- coding: utf-8 -*-
"""Thiel-Sen slope estimate
Created on Thu Apr 23 11:39:27 2015
@author: cdholmes
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def sen_slope( x, y ):
    '''Estimate linear trend using the Thiel-Sen method
    
    This non-parametric method finds the median slope among all
    combinations of time points. 
    scipy.stats.theilslopes provides the same slope estimate, with  
    confidence intervals. However, this function is faster for 
    large datasets due to Numba 
    
    Parameters
    ----------
    x : array-like
        independent variable
    y : array-like 
        dependent variable
    
    Returns
    -------
    sen : float
        the median slope
    slopes :array
        all slope estimates from all combinations of x and y
    '''

    if len( x ) != len( y ):
        print('Inputs x and y must have same dimension')
        return np.nan

    # Find number of time points
    n = len( x )

    # Array to hold all slope estimates
    slopes = np.zeros(  np.ceil( n * ( n-1 ) / 2 ).astype('int') )
    slopes[:] = np.nan

    count = 0

    for i in range(n):
        for j in range(i+1, n):

            # Slope between elements i and j
            slopeij = ( y[j] - y[i] ) / ( x[j] - x[i] )

            slopes[count] = slopeij

            count += 1

    # Thiel-Sen estimate is the median slope, neglecting NaN
    sen = np.nanmedian( slopes )

    return sen, slopes
