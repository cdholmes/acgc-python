# -*- coding: utf-8 -*-
# NAME:
#        BOXCARPOLY
#
# PURPOSE:
#        Calculate a boxcar polynomial (i.e. running polynomial fit) of an
#        array. 
#
# CATEGORY:
#
# CALLING SEQUENCE:
#        RESULT = BOXCARPOLY( Array, Width [,keywords] )
#
# INPUTS:
#        Array - 1D array of values to average
#        Width - number of elements to include in the average. Can be
#                odd or even.
#
# KEYWORD PARAMETERS:
#        CENTER - If used, then the averaging window for each element
#                 of the output array will be centered on the
#                 respective element of the input array.
#        BACKWARD-Averaging window will include trailing elements.
#        FORWARD -Averaging window will include leading elements.
#
#        ORDER  - Choose the order of polynomial to fit within each boxcar window. A parabola is order=2.
#        QUANTILE-Use quantile regression rather than least squares in the polynomial fit.
# OUTPUTS:
#        RESULT - 1D array with averages. Length will be same as input
#                 ARRAY. Elements on either end may be NaN
#
# SUBROUTINES:
#
# REQUIREMENTS:
#
# NOTES: 
#         A centered boxcar average with even widths is traditionally
#         not allowed. This BOXCAR program allows even widths by
#         giving half weights to elements on either end of the
#         averaging window and full weights to all others. A centered
#         average with even width therefore uses uses WIDTH+1
#         elements in its averaging kernel.  
#
# EXAMPLE:
#
# MODIFICATION HISTORY:
#        cdh, 26 Feb 2015: VERSION 1.00, adapted from IDL
#        cdh, 17 Feb 2015: Added optional median filter
#        cdh, 21 May 2015: boxcarpoly adapted from boxcar
#
#
#-
# Copyright (C) 2011, Christopher Holmes, UC Irvine
# This software is provided as is without any warranty whatsoever.
# It may be freely used, copied or distributed for non-commercial
# purposes.  This copyright notice must be kept with any copy of
# this software. If this software shall be used commercially or
# sold as part of a larger package, please contact the author.
# Bugs and comments should be directed to cdholmes@post.harvard.edu
# with subject "IDL routine boxcar"
#-----------------------------------------------------------------------

import numpy as np
    
def boxcarpoly( array, width, order=2, center=True, backward=False, forward=False, quantile=False):

    # Resolve any input conflicts by priorty: center, forward, backward
    # If none are chosen, then default to center
    if (center):
        backward=False
        forward=False        
    elif (forward): 
        center=False
        backward=False
    elif (backward):
        center=False
        forward=False
    else:
        center=True
        backward=False
        forward=False        
    if (backward or forward):
        center=False


    N = array.size
    
    # Initialize smoothed array
    smootharray = np.empty_like(array)
    smootharray[:] = np.NaN

    # uniform averaging kernel        
    kernel = np.ones(width)         
    
    # Setup for backward boxcar
    if (backward):
        
        # Half widths before and after element I
        HW1 = width-1
        HW2 = width-HW1-1
         
    # Setup for forward boxcar
    if (forward):
        HW1 = 0
        HW2 = width-HW1-1
        
    # Setup for centered boxcar
    if (center):    
        # Separate treatments for odd and even widths
        if (np.mod(width,2)==1):
            # Odd widths
            HW1 = np.floor(width/2)
            HW2 = width-HW1-1
            
        else:
            # Even widths
            HW1 = width/2
            HW2 = width-HW1
            
            # Uniform kernel in middle
            kernel = np.ones(width+1)
            
            # Half weight kernel for ends
            kernel[0] = 0.5
            kernel[width] = 0.5
            
            # Normalize the kernel
            kernel=kernel/kernel.sum()
           
    # Do boxcar
    for i in range(int(HW1),int(N-HW2-1)):
        
        # Sub array that we operate on
        sub = array[i-HW1:i+HW2+1]

        # Local x coordinate
        x = np.arange(-HW1,HW2+1)
    
        # Fit with polynomial of specified order
        # Avoid NaNs
        idx = np.isfinite(sub)

        # number of valid points (non NaN)        
        nvalid = np.sum(idx)
        
        if (nvalid == 0):
            # smoothed value is NaN when there are no valid points
            smootharray[i] = np.nan
        else:
            # A polynomial fit requires at least (order+1) points.
            # When there are fewer points, use the highest possible order.
            ordmax = np.max([np.min([order,np.sum(idx)-1]),0])
        
            p = np.polyfit(x[idx],sub[idx],ordmax,w=kernel[idx])
            
            # The fitted value at local x=0 is just the constant term
            smootharray[i] = p[order]
            
    return smootharray
