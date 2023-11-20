# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:39:27 2015

PURPOSE:
Estimate the trend in a time series using Sen's method. 
This non-parametric method finds the median slope among all
combinations of time points.

See also scipy.stats.theilslopes, which provides confidence intervals.
However, this function is faster for large datasets due to Numba 

@author: cdholmes
"""

import numpy as np
from numba import jit

@jit
def sen_slope( X, Y ):

    if (len( X ) != len( Y )):
         print('Inputs X and Y must have same dimension')
         return np.nan

    # Find number of time points
    n = len( X )

    # Array to hold all slope estimates
    slopes = np.zeros(  np.ceil( n * ( n-1 ) / 2 ).astype('int') )
    slopes[:] = np.nan

    count = 0

    for i in range(n):
        for j in range(i+1, n):
      
            # Slope between elements i and j
            slopeij = ( Y[j] - Y[i] ) / ( X[j] - X[i] )

            slopes[count] = slopeij

            count += 1

    # Thiel-Sen estimate is the median slope, neglecting NaN
    sen = np.nanmedian( slopes )

    return sen, slopes


