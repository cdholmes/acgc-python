#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Weighted variance, covariance, correlation, median, and quantiles

The weighted correlation coefficient here may be usedful to construct a 
a "weighted R2" or "weighted coefficient of determination" from a 
Weighted Least Squares regression: R2w = wcorr(x,y,w)**2. The resulting R2w is
equivalent to equation 4 from Willett and Singer (1988, American Statistician) 
and their caveats apply. 

Craig Milligan provides the following advice in an online forum:
"What is the meaning of this (corrected) weighted r-squared? 
Willett and Singer interpret it as: "the coefficient of determination in the 
transformed [weighted] dataset. It is a measure of the proportion of the 
variation in weighted Y that can be accounted for by weighted X, and is the 
quantity that is output as R2 by the major statistical computer packages when a 
WLS regression is performed".

Is it meaningful as a measure of goodness of fit? 
This depends on how it is presented and interpreted. Willett and Singer caution 
that it is typically quite a bit higher than the r-squared obtained in ordinary 
least squares regression, and the high value encourages prominent display... 
but this display may be deceptive IF it is interpreted in the conventional 
sense of r-squared (as the proportion of unweighted variation explained by a 
model). Willett and Singer propose that a less 'deceptive' alternative is 
their equation 7 [usual unweighted R2, called pseudor2wls by Willett and Singer]. 
In general, Willett and Singer also caution that it is 
not good to rely on any r2 (even their pseudor2wls) as a sole measure of 
goodness of fit. Despite these cautions, the whole premise of robust regression 
is that some cases are judged 'not as good' and don't count as much in the model 
fitting, and it may be good to reflect this in part of the model assessment 
process. The weighted r-squared described, can be one good measure of goodness 
of fit - as long as the correct interpretation is clearly given in the 
presentation and it is not relied on as the sole assessment of goodness of fit."



Willett, J. B. and Singer, J. D.: Another Cautionary Note About R2: 
    Its Use in Weighted Least-Squares Regression-Analysis, American Statistician, 
    42(3), 236–238, 1988.

Created on Mon Apr 24 17:39:51 2017

@author: cdholmes
"""

import numpy as np
from sklearn.covariance import MinCovDet
from scipy.interpolate import interp1d

def wmean(x,w=None,robust=False):
    '''Weighted mean 
    
    Calculate the mean of x using weights w.
    
    Args:
        x : array of values to be averaged
        w      : array of weights for each element of x; can be ommitted if robust=True
        robust : (boolean) robust weights will be internally calculated using FastMCD;
                 only used if robust=True and w is empty
        
    Returns:
        scalar : weighted mean    
    '''
    if (w!=None):
        assert len(w) == len(x), 'w must be the same length as x'

    # Use FastMCD to calculate weights; Another method could be used here
    if (robust and w==None):
        w = MinCovDet().fit( np.array([x,x]).T ).support_
    
    if (len(w) == 0): raise SystemExit('must specify weights w or select robust=True')
    assert len(w) == len(x), 'w must be the same length as x'

    return np.sum( x * w ) / np.sum(w)

def wstd(x,w=None,ddof=1,robust=False):
    '''Weighted standard deviation
    
    Calculate the standard deviation of x using weights w. If ddof=1 (default),
    then the result is the unbiased (sample) standard deviation when w=1.
    
    Args:
        x    : array of values 
        w      : array of weights for each element of x; can be ommitted if robust=True
        ddof   : scalar differential degrees of freedom (Default ddof=1)
        robust : (boolean) robust weights will be internally calculated using FastMCD;
                 only used if robust=True and w is empty
        
    Returns:
        scalar : weighted variance   
    '''
    if (w!=None):
        assert len(w) == len(x), 'w must be the same length as x'
        
    return np.sqrt( wcov(x,x,w,ddof,robust) )

def wvar(x,w=None,ddof=1,robust=False):
    '''Weighted variance 
    
    Calculate the variance of x using weights w. If ddof=1 (default),
    then the result is the unbiased (sample) variance when w=1.
    
    Args:
        x    : array of values 
        w      : array of weights for each element of x; can be ommitted if robust=True
        ddof   : scalar differential degrees of freedom (Default ddof=1)
        robust : (boolean) robust weights will be internally calculated using FastMCD;
                 only used if robust=True and w is empty
        
    Returns:
        scalar : weighted variance   
    '''
    if (w!=None):
        assert len(w) == len(x), 'w must be the same length as x'
        
    return wcov(x,x,w,ddof,robust)


def wcov(x,y,w=None,ddof=1,robust=False):
    '''Weighted covariance 
    
    Calculate the covariance of x and y using weights w. If ddof=1 (default),
    then the result is the unbiased (sample) covariance when w=1.
    
    Implements weighted covariance as defined by NIST Dataplot (https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weighvar.pdf)
    
    Args:
        x,y    : array of values 
        w      : array of weights for each element of x; can be ommitted if robust=True
        ddof   : scalar differential degrees of freedom (Default ddof=1)
        robust : (boolean) robust weights will be internally calculated using FastMCD;
                 only used if robust=True and w is empty
        
    Returns:
        scalar : weighted covariance   
    '''
    n = len(x)   
    assert len(y) == n, 'y must be the same length as x'

    # Use FastMCD to calculate weights; Another method could be used here
    if (robust and w==None):
        w = MinCovDet().fit( np.array([x,y]).T ).support_
    
    if (len(w) == 0): raise SystemExit('must specify weights w or select robust=True')
    assert len(w) == n, 'w must be the same length as x and y'

    w = wscale(w)
    nw = np.count_nonzero(w)

    return np.sum( ( x - wmean(x,w) ) * ( y - wmean(y,w) ) * w ) / \
        ( np.sum(w) / nw * (nw - ddof) )
    
def wcorr(x,y,w=None,robust=False):
    '''Weighted correlation coeffient
    
    Calculate the Pearson linear correlation coefficient of x and y using weights w. 
    This is derived from the weighted covariance and weighted variance.
    
    Args:
        x,y    : array of values 
        w      : array of weights for each element of x
        robust : (boolean) robust weights will be internally calculated using FastMCD;
                 only used if robust=True and w is empty
        
    Returns:
        scalar : weighted covariance   
    '''

    n = len(x)   
    assert len(y) == n, 'y must be the same length as x'

    # Use FastMCD to calculate weights; Another method could be used here
    if (w==None or robust==True):
        w = MinCovDet().fit( np.array([x,y]).T ).support_
    
    if (len(w) == 0): raise SystemExit('must specify weights w or select robust=True')
    assert len(w) == n, 'w must be the same length as x and y'
    w = wscale(w)
    return wcov(x,y,w) / np.sqrt( wvar(x,w) * wvar(y,w) )

# Alias for wcorr
wcorrcoef = wcorr

def wscale(w):
    '''Scale array to a maximum value of 1

    Rescale array to a maximum value of 1. 
    In weighted averaging, we will assume that a weight of 1 means 1 degree of 
    freedom in the observations.
    
    Args:
        w : array 
        
    Returns:
        array : input array rescaled to a maximum value of 1
    '''
    return w / np.max( w )

def wmedian(x,w,**kwargs):
    '''Weighted median

    See documentation for wquantile
    '''
    return wquantile(x,0.5,w,**kwargs)

def wquantile(x,q,w,interpolation='partition'):
    '''Weighted quantile 
    
    Calculate the quantile q from data array x using data weights w.
    If weights reflect the relative frequency of the elements of x in a large population,
    then the weighted quantile result is equivalent to numpy.quantile or numpy.percentile 
    operating on an array of that larger population (containing many repeated elements).

    For uniform weights and interpolation=partition or partition0, this function 
    differs from numpy.percentile. This behavior is expected and desirable.
    The numpy.percentile behaviro can be reproduced here by using 
    interpolation = linear, lower, higher, or nearest.
    Note that numpy.quantile is equivalent to numpy.percentile(interpolation='linear')
    
    This naive algorithm is O(n) and may be slow for large samples (x).
    Consider using Robustats or another optimized package.       
    
    Args:
        x      : array of values to compute quantiles
        q      : quantile or list of quantiles to calculate, in range 0-1
        w      : array of weights for each element of x (e.g. representing the frequency of elements x in a large population)
        interpolation : {'partition' [default], 'partition0', 'linear', 'nearest', 'lower', 'upper'}
            This parameter specifies the interpolation method to use when the desired quantile 
            lies bewteen elements i < j in x.
            'partition': [default] choose the element of x that partitions the 
                         sum of weights on either side to q and (1-q)
                         When two elements both satisfy partition, then average them.
                         This is the Edgeworth method (https://en.wikipedia.org/wiki/Weighted_median)
            'partition0': Same as partition, but result is always an element of x (no averaging). 
                         Instead return the element of x that partitions weights most closely to q and (1-q) 
                         or, if there is still a tie, then the smaller element.
            'linear'  : i + (j-1) * fraction. replicates behavior of numpy.quantile when all 
                        weights are equal
            'nearest' : i or j element that most closely divides data at the q quantile
            'lower'   : i, the largest element <= the q quantile
            'higher'  : j, the smallest element >= the q quantile
                         
    Returns:
        scalar or array : weighted quantile
    '''

    # Ensure arguments are arrays
    x = np.asarray( x )
    w = np.asarray( w )

    # Number of elements
    n = len(x)

    # Ensure weights are same length as x
    if (len(w) != n ):
        raise ValueError( 'weights w must be the same length as array x')

    # Ensure inputs are all finite, no NaN or Inf
    if np.any( ~np.isfinite(x) ):
        raise ValueError( 'Array x contains non-finite elements')
    if np.any( ~np.isfinite(w) ):
        raise ValueError( 'Weights w contains non-finite elements')
    
    # Sort x from smallest to largest
    idx = np.argsort( x )


    if interpolation in ['partition','partition2']:

        # To calculate multiple quantiles, call function iteratively for each quantile requested
        if isinstance(q, (list, tuple, np.ndarray)):
            return [wquantile(x,qi,w,interpolation) for qi in q]
    
        # Cumulative sum of weights, divided by sum
        wsum = np.cumsum( w[idx] ) / np.sum( w )
        # Reverse cumulative sum (cumulative sum of elements in reverse order)
        wsumr = np.cumsum( w[idx][::-1] )[::-1] / np.sum( w )
    
        # Lower bound for quantile; il is an index into the sorted array
        if q <= wsum[0]:
            il = 0
        else:
            il   = np.flatnonzero( wsum < q )[-1] + 1
        
        # Upper bound for quantile; iu is an index into the sorted array
        if (1-q) <= wsumr[-1]:
            iu = n-1
        else:
            iu   = np.flatnonzero( wsumr < (1-q) )[0] - 1
            
        if il == iu:
            # Upper and lower bounds are the same; we're done
            xq = x[idx[il]]
    
        else:
            # Several methods for reconciling different upper and lower bounds
    
            if interpolation == 'partition':
                # Average the upper and lower bounds
                # This creates an element not found in the input array, which may be inappropriate in some cases 
                xq = np.mean( x[idx[[il,iu]]] )

            else:
                # Choose the element with the smaller weight
                # This guarantees that the value is an element of the input array
                if w[idx[il]] <= w[idx[iu]]:
                    iq = il
                else:
                    iq = iu
                xq = x[idx[iq]]

    else:

        # These methods give the same results as numpy.percentile when using 
        # the same interpolation method and uniform weights.

                        
        # Define the quantile for each element in x
        w         = w.astype(np.float32)
        qx        = w[idx] / 2 - w[idx][0] / 2
        qx[1:]   += np.cumsum( w[idx] )[:-1] 
        qx       /= qx[-1]

        # Interpolate to get quantile value
        if interpolation == 'linear':
            f = interp1d(qx,x[idx],kind='linear')
        elif interpolation == 'nearest':
            f = interp1d(qx,x[idx],kind='nearest')
        elif interpolation == 'lower':
            f = interp1d(qx,x[idx],kind='previous')
        elif interpolation == 'higher':
            f = interp1d(qx,x[idx],kind='next')
        elif interpolation == 'midpoint':
            # Average the lower and higher values
            f = lambda q: ( interp1d(qx,x[idx],kind='previous')(q) +
                            interp1d(qx,x[idx],kind='next')(q)     ) / 2
        else:
            raise ValueError('Unrecognized value for interpolation: ' + interpolation)
            
        xq = f(q)
    
    return xq
