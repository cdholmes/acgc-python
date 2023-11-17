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
import xarray as xr

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


class bivariate_statistics(object):
    
    def __init__(self,x,y,w=None,**kwargs):

        from scipy import stats
        
        #Ensure that x and y have same length
        if len(x) != len(y):
            raise ValueError( 'Arguments x and y must have the same length' )

        diff = y - x
        absdiff = np.abs( y - x )
        ratio = y/x

        # Means, medians, and standard deviations
        self.xmean = np.mean(x)
        self.ymean = np.mean(y)
        self.xmedian = np.median(x)
        self.ymedian = np.median(y)
        self.xstd   = np.std(x)
        self.ystd   = np.std(y)

        self._x = x
        self._y = y
        self._w = w

        # Mean and mean absolute differences
        self.mean_difference            = self.md   = self.ymean - self.xmean
        self.mean_absolute_difference   = self.mad  = np.mean( absdiff )

        # Relative and standardized differences
        self.relative_mean_difference           = self.rmd  = self.mean_difference / self.xmean
        self.relative_mean_absolute_difference  = self.rmad = self.mean_absolute_difference / self.xmean
        self.standardized_mean_difference       = self.smd  = self.mean_difference / self.xstd
        self.standardized_mean_absolute_difference  = self.smad = self.mean_absolute_difference / self.xstd

        # Mean and median relative differences
        self.mean_relative_difference   = self.mrd  = np.mean( ratio - 1 )
        self.median_relative_difference = self.medianrd= np.median( ratio - 1 )

        # Median and median absolute differences
        self.median_difference            = np.median( diff ) 
        self.median_absolute_difference   = np.median( absdiff )

        # Relative median differences
        self.relative_median_difference   = self.rmd  = self.median_difference / self.xmedian
        self.relative_median_absolute_difference      = self.median_absolute_difference / self.xmedian

        self.normalized_mean_bias_factor            = self.nmbf  = nmbf(x,y)
        self.normalized_mean_absolute_error_factor  = self.nmaef = nmaef(x,y)

        # RMS difference
        self.root_mean_square_difference    = self.rmsd     = np.sqrt( np.mean( np.power( diff, 2) ) )

        # Covariance, correlation
        self.covariance = np.cov(x,y)[0][1]
        self.correlation = self.correlation_pearson = self.R = self.r = np.corrcoef(x,y)[0][1]
        self.correlation_spearman = stats.spearmanr(x,y).statistic
        self.R2 = self.r2 = self.R**2
      
    def __getitem__(self,key):
        return getattr(self,key)

    def fitline(self,method='sma',intercept=True,**kwargs):
        '''Compute bivariate line fit
        
        Arguments
        ---------
        method : str
            line fitting method: sma (default) or old
        intercept : bool
            defines whether non-zero intercept should be fitted
        **kwargs passed to smafit (e.g. robust=True)

        Returns
        -------
        Result : dict containing keys
            slope : slope of fitted line
            intercept : intercept of fitted line
            fittedvalues : values on fit line
            residuals : residual from fit line
        '''  

        from smafit import smafit

        if method.lower()=='sma':
            sma = smafit(self._x,
                        self._y,
                        self._w,
                        intercept=intercept,
                        **kwargs) 
            slope = sma['slope']
            intercept= sma['intercept']
        
        elif method.lower()=='ols':
            if intercept:
                ols = np.linalg.lstsq( np.vstack([self._x,np.ones(len(self._x))]).T, self._y, rcond=None )
            else:
                ols = np.linalg.lstsq( np.vstack([self._x]).T, self._y, rcond=None )
            slope = ols[0][0]
            intercept = ols[0][1]
        else:
            raise ValueError('Undefined method '+method)

        line = dict( slope          = slope,
                     intercept      = intercept,
                     fittedvalues   = slope * self._x + intercept,
                     residuals      = self._y - ( slope * self._x + intercept ) )

        return line

    def slope(self,method='sma',intercept=True,**kwargs):
        '''Compute slope of bivariate line fit
        
        Arguments
        ---------
        method : str
            line fitting method: sma (default) or old
        intercept : bool
            defines whether non-zero intercept should be fitted
        **kwargs passed to fitline()

        Returns
        -------
        slope : float
            value of y intercept
        '''        
        return self.fitline(method,intercept,**kwargs)['slope']
    
    def intercept(self,method='sma',intercept=True,**kwargs):
        '''Compute intercept of bivariate line fit
        
        Arguments
        ---------
        method : str
            line fitting method: sma (default) or old
        intercept : bool
            defines whether non-zero intercept should be fitted
        **kwargs passed to fitline()

        Returns
        -------
        intercept : float
            value of y intercept
        '''
        return self.fitline(method,intercept,**kwargs)['intercept']

    def summary(self, variables=None, floatformat='{:10f}', stringlength=None, 
                fitline_kw=None ):
        '''Summarize bivariate statistics

        Arguments
        ---------
        vars : list
            names of variables to include in summary
        floatformat : str
            format specifier for floating point values
        stringlength : int
            length of the variables on output
        
        Returns
        -------
        summary : str
            names and values of variables'''

        if variables is None:
            variables=['R','RMD']

        if stringlength is None:
            stringlength = np.max([len(v) for v in variables])+1
        stringformat = '{:'+str(stringlength)+'s}'

        if fitline_kw is None:
            fitline_kw = {'method':'sma',
                          'intercept':True}

        # Extract length of the float numbers from floatformat
        # import re
        # floatlength = np.floor( float( re.findall("[-+]?(?:\d*\.*\d+)", floatformat )[0] ) ).astype(int)

        # summary = (stringformat+'{:>10s}').format('Variable','Value')
        summary = ''
        for v in variables:
            if v in ['slope','intercept']:
                # These variables are object methods
                func = getattr(self,v)
                value = func(**fitline_kw)
            else:
                # Retrieve values
                value = getattr(self,v.lower())
            
            summary += (stringformat+'='+floatformat+'\n').format(v,value)
        return summary
    
