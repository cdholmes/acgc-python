#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Weighted variance, covariance, and correlation

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

def wmean(x,w):
    '''Weighted mean 
    
    Calculate the mean of x using weights w.
    
    Args:
        x : array of values to be averaged
        w : array of weights for each element of x
        
    Returns:
        scalar : weighted mean    
    '''
    n = len(x)
    assert len(w) == n, 'w must be the same length as x'
    return np.sum( x * w ) / np.sum(w)


def wvar(x,w,ddof=1):
    '''Weighted variance 
    
    Calculate the variance of x using weights w. If ddof=1 (default),
    then the result is the unbiased (sample) variance when w=1.
    
    Args:
        x : array of values 
        w : array of weights for each element of x
        ddof: scalar differential degrees of freedom (Default ddof=1)
        
    Returns:
        scalar : weighted variance   
    '''
    n = len(x)   
    assert len(w) == n, 'w must be the same length as x'

    w = wscale(w)
    return np.sum( ( x - wmean(x,w) )**2 * w ) / (np.sum(w) - ddof)



def wcov(x,y,w,ddof=1):
    '''Weighted covariance 
    
    Calculate the covariance of x and y using weights w. If ddof=1 (default),
    then the result is the unbiased (sample) covariance when w=1.
    
    Args:
        x,y: array of values 
        w : array of weights for each element of x
        ddof: scalar differential degrees of freedom (Default ddof=1)
        
    Returns:
        scalar : weighted covariance   
    '''
    n = len(x)   
    assert len(y) == n, 'y must be the same length as x'
    assert len(w) == n, 'w must be the same length as x and y'
    w = wscale(w)
    return np.sum( ( x - wmean(x,w) ) * ( y - wmean(y,w) ) * w ) / (np.sum(w) - ddof)
    
def wcorr(x,y,w):
    '''Weighted correlation coeffient
    
    Calculate the Pearson linear correlation coefficient of x and y using weights w. 
    This is derived from the weighted covariance and weighted variance.
    
    Args:
        x,y: array of values 
        w : array of weights for each element of x
        
    Returns:
        scalar : weighted covariance   
    '''
    n = len(x)   
    assert len(y) == n, 'y must be the same length as x'
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

    