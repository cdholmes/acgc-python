# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    "boxcar",
    "boxcarpoly"
]

def boxcarpoly( array, width, order=2, align='center'):
    '''Calculate a boxcar polynomial (i.e. running polynomial fit) of an array. 
    
    See boxcar for parameter definitions
    '''
    return boxcar( array, width, order=order, align=align, method='polynomial')

def boxcar( array, width, align='center', method='mean', order=2 ):
    '''Calculate a boxcar average (i.e. running mean, median, or polynomial fit) of an array. 
    
    Elements of input array are assumed to be equally spaced along an (unspecified) 
    coordinate dimension.

    A centered boxcar average with even widths is traditionally
    not allowed. This BOXCAR program allows even widths by
    giving half weights to elements on either end of the
    averaging window and full weights to all others. A centered
    average with even width therefore uses uses WIDTH+1
    elements in its averaging kernel.  

    Parameters
    ----------
    array : array of floats
        1D array of values to average
    width : int
        number of elements to include in the average. 
    align : str
        specifies how the averaging window for each element of the output array
        aligns with the input array. 
        Values: center (default), forward (trailing elements), backward (leading elements)
    method : str
        specifies the averaging kernel function: "mean", "median", "polynomial"
    order : int
        specifies the polynomial order to fit within each boxcar window. A parabola is order=2.
        order=0 is equivalent to method="mean"
            
    Returns
    -------
    result : array of floats
        1D array with averages with same size as input array. Elements on either end may be NaN
    '''

    N = array.size

    # Initialize smoothed array
    smootharray = np.empty_like(array)
    smootharray[:] = np.NaN

    # uniform averaging kernel
    kernel = np.ones(width)

    # Setup for backward boxcar
    if align=='backward':
        # Half widths before and after element I

        HW1 = width-1
        HW2 = width-HW1-1

    # Setup for forward boxcar
    elif align=='forward':
        HW1 = 0
        HW2 = width-HW1-1

    # Setup for centered boxcar
    elif align=='center':    
        # Separate treatments for odd and even widths
        if np.mod(width,2)==1:
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
    else:
        raise ValueError( f'align={align} not implemented. '
                         + 'Value should be "center", "forward" or "backward".' )

    # Convert to integer type
    HW1 = int(HW1)
    HW2 = int(HW2)

    # Do boxcar
    for i in range(HW1,N-HW2-1):

        # Sub array that we operate on
        sub = array[i-HW1:i+HW2+1]

        if method=='median':
            # Running median
            smootharray[i] = np.nanmedian(sub)

        elif method=='mean':
            # Running mean
            # Kernel average, only over finite elements
            #(NaNs removed from numerator and denominator
            smootharray[i] = np.nansum( sub*kernel ) / np.sum(kernel[np.where(np.isfinite(sub))])

        elif method=='polynomial':

            # Local x coordinate
            x = np.arange(-HW1,HW2+1)

            # Fit with polynomial of specified order
            # Avoid NaNs
            idx = np.isfinite(sub)

            # number of valid points (non NaN)
            nvalid = np.sum(idx)

            if nvalid==0:
                # smoothed value is NaN when there are no valid points
                smootharray[i] = np.nan

            else:
                # A polynomial fit requires at least (order+1) points.
                # When there are fewer points, use the highest possible order.
                ordmax = np.max([np.min([order,np.sum(idx)-1]),0])

                p = np.polyfit(x[idx],sub[idx],ordmax,w=kernel[idx])

                # The fitted value at local x=0 is just the constant term
                smootharray[i] = p[order]

        else:
            raise ValueError( f'method={method} not implemented. '
                             + 'Value should be "mean", "median" or "polynomial"')

    return smootharray
