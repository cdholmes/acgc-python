#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Statistical measures of bias between two populations
Includes: 
Normalized Mean Bias Factor (NMBF) and Normalized Mean Absolute Error Factor (NMAEF)

Yu, S., Eder, B., Dennis, R., Chu, S.-H., & Schwartz, S. E. (2006). 
    New unbiased symmetric metrics for evaluation of air quality models.
    Atmospheric Science Letters, 7(1), 26–34. https://doi.org/10.1002/asl.125

Created on June 4 2019

@author: cdholmes
"""

import numpy as np

def nmb( xObs, xMod=None ):
    '''Compute Normalized Mean Bias (NMB)
    nmb = (<xMod> - <xObs>) / <xObs>

    xObs : Observed values
    xMod : Model values
    '''

    # Ensure that xObs has the same length as xMod
    if (xMod is not None):
        assert (len(xMod) == len(xObs)), "xMod must have the same length as xObs"

    # Mean values
    oMean = np.mean(xObs)
    mMean = np.mean(xMod)

    # Metric value
    nmb = mMean / oMean - 1
 
    return nmb

def nmae( xObs, xMod ):
    '''Compute Normalized Mean Absolute Error (NMAE)
    '''

     # Mean values
    oMean = np.mean(xObs)

    # Mean absolute difference
    aDiff = np.mean( np.abs(xMod - xObs) )

    # Metric value
    nmae = aDiff / np.abs( oMean ) 

    return nmae

def nmbf( xObs, xMod=None ):
    '''Compute Normalized Mean Bias Factor (NMBF)
    Definition from Yu et al. (2006, Atmos. Sci. Lett.)

    xObs : Observed values
    xMod : Model values
    '''

    # Ensure that xObs has the same length as xMod
    if (xMod is not None):
        assert (len(xMod) == len(xObs)), "xMod must have the same length as xObs"

    # Mean values
    oMean = np.mean(xObs)
    mMean = np.mean(xMod)

    # Metric value
    if (mMean >= oMean):
        nmbf = mMean / oMean - 1
    else:
        nmbf = 1 - oMean / mMean
    # Equivalent (faster?) implementation
    #S = (mMean - oMean) / np.abs(mMean - oMean)
    #nmbf = S * ( np.exp( np.abs( mMean / oMean )) - 1 )

    return nmbf

def nmaef( xObs, xMod ):
    '''Compute Normalized Mean Absolute Error Factor (NMAE)
    Definition from Yu et al. (2006, Atmos. Sci. Lett.)
    
    xObs : Observed values
    xMod : Model values
    '''
     # Mean values
    oMean = np.mean(xObs)
    mMean = np.mean(xMod)

    # Mean absolute difference
    aDiff = np.mean( np.abs(xMod - xObs))

    # Metric value
    if (mMean >= oMean):
        nmaef = aDiff / oMean 
    else:
        nmaef = aDiff / mMean
    # Equivalent (faster?) implementation
    #S = (mMean - oMean) / np.abs(mMean - oMean)
    #nmaef = aDiff / ( oMean**((1+S)/2) * mMean**((1-S)/2) )

    return nmaef