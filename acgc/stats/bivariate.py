#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Bivariate statistics

Statistical measures of relationships between two populations
"""

import numpy as np
from scipy import stats
from .bivariate_lines import sen, sma, bivariate_line_equation
# import xarray as xr

__all__ = [
    "BivariateStatistics",
    "nmb",
    "nmae",
    "nmbf",
    "nmaef"
]

def nmb( x0, x1 ):
    '''Compute Normalized Mean Bias (NMB)

    NMB = ( mean(x1) - mean(x0) ) / mean(x0)

    Parameters
    ----------
    x0 : array_like
        reference values
    x1 : array_like
        experiment values
    '''

    assert (len(x0) == len(x1)), \
        "Parameters x0 and x1 must have the same length"

    # Mean values
    x0_mean = np.mean(x0)
    x1_mean = np.mean(x1)

    # Metric value
    return x1_mean / x0_mean - 1

def nmae( x0, x1 ):
    '''Compute Normalized Mean Absolute Error (NMAE)

    NMAE = mean(abs(x1 - x0)) / abs(mean(x0))

    Parameters
    ---------
    x0 : array_like
        reference values
    x1 : array_like
        experiment values
    '''

     # Mean values
    x0_mean = np.mean(x0)

    # Mean absolute difference
    abs_diff = np.mean( np.abs(x1 - x0) )

    # Metric value
    return abs_diff / np.abs( x0_mean )


def nmbf( x0, x1 ):
    '''Compute Normalized Mean Bias Factor (NMBF)

    Definition from Yu et al. (2006) https://doi.org/10.1002/asl.125

    Parameters
    ----------
    x0 : array_like
        reference values
    x1 : array_like
        experiment values
    '''

    # Ensure that arguments have the same length
    assert (len(x0) == len(x1)), \
        "Parameters x0 and x1 must have the same length"

    # Mean values
    x0_mean = np.mean(x0)
    x1_mean = np.mean(x1)

    # Metric value
    if x1_mean >= x0_mean:
        result = x1_mean / x0_mean - 1
    else:
        result= 1 - x0_mean / x1_mean
    # Equivalent (faster?) implementation
    #S = (mMean - oMean) / np.abs(mMean - oMean)
    #result = S * ( np.exp( np.abs( mMean / oMean )) - 1 )

    return result

def nmaef( x0, x1 ):
    '''Compute Normalized Mean Absolute Error Factor (NMAEF)

    Definition from Yu et al. (2006) https://doi.org/10.1002/asl.125
    
    Parameters
    ----------
    x0 : array_like
        reference values
    x1 : array_like
        experiment values
    '''

    # Ensure that arguments have the same length
    assert (len(x0) == len(x1)), \
        "Parameters x0 and x1 must have the same length"

    # Mean values
    x0_mean = np.mean(x0)
    x1_mean = np.mean(x1)

    # Mean absolute difference
    abs_diff = np.mean( np.abs(x1 - x0))

    # Metric value
    if x1_mean >= x0_mean:
        result = abs_diff / x0_mean 
    else:
        result = abs_diff / x1_mean
    # Equivalent (faster?) implementation
    #S = (exp_mean - ref_mean) / np.abs(exp_mean - ref_mean)
    #result = abs_diff / ( oMean**((1+S)/2) * mMean**((1-S)/2) )

    return result

def _texify_name(name):
    '''Return a LaTex formatted string for some variables
    
    Parameters
    ----------
    name : str
    
    Returns
    -------
    pretty_name : str
    '''
    if name.lower()=='n':
        pretty_name = r'$n$'
    elif name=='R2':
        pretty_name = f'$R^2$'
    elif name=='r2':
        pretty_name = f'$r^2$'
    elif name.lower()=='y_ols':
        pretty_name = r'$y_{\rm OLS}$'
    elif name.lower()=='y_sma':
        pretty_name = r'$y_{\rm SMA}$'
    elif name.lower()=='y_sen':
        pretty_name = r'$y_{\rm Sen}$'
    else:
        pretty_name = name
    return pretty_name

def _number2str(value,
                intformat='{:d}',
                floatformat='{:.4f}'):
    '''Format number as string using integer and float format specifiers
    
    Parameters
    ----------
    value : numeric, str
        value to be converted
    intformat : str, default='{:d}'
        format specifier for integer types
    floatformat : str, default='{:.4f}'
        format specifier for float types

    Returns
    -------
    str
    '''
    if isinstance(value,str):
        pass
    elif isinstance(value,(int,np.integer)):
        value = intformat.format(value)
    else:
        value = floatformat.format(value)
    return value

class BivariateStatistics:
    '''A suite of common statistics to quantify bivariate relationships

    Class method 'summary' provides a formatted summary of these statistics
    
    Attributes
    ----------
    count, n : int
        number of valid (not NaN) data value pairs
    xmean, ymean : float
        mean of x and y variables
    xmedian, ymedian :float
        median of x and y variables
    xstd, ystd : float
        standard deviation of x and y variables
    mean_difference, md : float
        ymean - xmean
    std_difference, stdd : float
        std( y - x )
    mean_absolute_difference, mad : float
        mean( |y-x| )
    relative_mean_difference, rmd : float
        md / xmean
    relative_mean_absolute_difference, rmad :float
        mad / xmean
    standardized_mean_difference, smd : float
        md / xstd
    standardized_mean_absolute_difference, smad : float
        mad /xstd
    mean_relative_difference, mrd : float
        mean(y/x) - 1
    mean_log10_ratio, mlr : float
        mean( log10(y/x) )
    std_log10_ratio, stdlr : float
        std( log10(y/x) )
    mean_absolute_log10_ratio, malr : float
        mean( abs( log10(y/x) ) )
    median_difference, medd : float
        median(y-x)
    median_absolute_difference, medad : float
        median(|y-x|)
    relative_median_difference, rmedd : float
        median(y-x) / xmedian
    relative_median_absolute_difference, rmedad : float
        median(|y-x|) / xmedian
    median_relative_difference, medianrd, medrd : float
        median(y/x)-1
    median_log10_ratio, medlr : float
        median( log10(y/x) )
    median_absolute_log10_ratio, medalr : float
        median( abs( log10(y/x) ) )
    normalized_mean_bias_factor, nmbf : float
        see `nmbf` 
    normalized_mean_absolute_error_factor, nmaef : float
        see `nmaef`
    root_mean_square_difference, rmsd : float
        $\\sqrt{ \\langle (y - x)^2 \\rangle }$
    root_mean_square_log10_ratio, rmslr : float
        $\\sqrt{ \\langle log10(y/x)^2 \\rangle }$
    covariance : float
        cov(x,y)
    correlation_pearson, correlation, pearsonr, R, r : float
        Pearson linear correlation coefficient 
    correlation_spearman, spearmanr : float
        Spearman, non-parametric rank correlation coefficient
    R2, r2 : float
        Linear coefficient of determination, $R^2$
    '''

    def __init__(self,x,y,w=None,dropna=False,data=None):
        '''Compute suite of bivariate statistics during initialization
        
        Statistic values are saved in attributes.
        CAUTION: Weights w are ignored except in SMA fit

        Parameters
        ----------
        x : ndarray or str
            independent variable values
        y : ndarray or str
            dependent variable values, same size as x
        w : ndarray or str, optional
            weights for points (x,y), same size as x and y
        dropna : bool, optional (default=False)
            drops NaN values from x, y, and w
        data : dict-like, optional
            if x, y, or w are str, then they should be keys in data
        '''

        # Get values from data if needed
        if data is None and (isinstance(x,str) or isinstance(y,str) or isinstance(w,str)):
            raise ValueError( 'Data argument must be used if x, y, or w is a string')
        if isinstance(x,str):
            x = data[x]
        if isinstance(y,str):
            y = data[y]
        if isinstance(w,str):
            w = data[w]

        #Ensure that x and y have same length
        if len(x) != len(y):
            raise ValueError( 'Arguments x and y must have the same length' )
        if w is None:
            w = np.ones_like(x)
        if len(w) != len(x):
            raise ValueError( 'Argument w (if present) must have the same length as x' )

        # Drop NaN values
        if dropna:
            isna = np.isnan(x*y*w)
            x = x[~isna]
            y = y[~isna]
            w = w[~isna]

        # Differences and ratios used repeatedly
        diff = y - x
        absdiff = np.abs( y - x )
        # Ignore divide by zero and 0/0 while dividing
        old_settings = np.seterr(divide='ignore',invalid='ignore')
        ratio = y/x
        log10ratio = np.log10(ratio)
        np.seterr(**old_settings)

        # Number of data points
        self.count = self.n = len(x)

        # Means, medians, and standard deviations
        self.xmean = np.mean(x)
        self.ymean = np.mean(y)
        self.xmedian = np.median(x)
        self.ymedian = np.median(y)
        self.xstd   = np.std(x)
        self.ystd   = np.std(y)

        # Save values for use later
        self._x = x
        self._y = y
        self._w = w

        # Mean and mean absolute differences
        self.mean_difference            = self.md   = self.ymean - self.xmean
        self.mean_absolute_difference   = self.mad  = np.mean( absdiff )
        self.std_difference             = self.stdd = np.std( diff )

        # Relative and standardized differences
        self.relative_mean_difference           = self.rmd  = self.mean_difference / self.xmean
        self.relative_mean_absolute_difference  = self.rmad = self.mean_absolute_difference / self.xmean
        self.standardized_mean_difference       = self.smd  = self.mean_difference / self.xstd
        self.standardized_mean_absolute_difference  = self.smad = self.mean_absolute_difference / self.xstd

        # Mean and median relative differences
        self.mean_relative_difference   = self.mrd  = np.mean( ratio - 1 )
        self.median_relative_difference = self.medianrd = self.medrd = np.median( ratio - 1 )

        # Median and median absolute differences
        self.median_difference          = self.medd  = np.median( diff )
        self.median_absolute_difference = self.medad = np.median( absdiff )

        # Relative median differences
        self.relative_median_difference          = self.rmedd  = self.median_difference / self.xmedian
        self.relative_median_absolute_difference = self.rmedad = self.median_absolute_difference / self.xmedian

        self.normalized_mean_bias_factor            = self.nmbf  = nmbf(x,y)
        self.normalized_mean_absolute_error_factor  = self.nmaef = nmaef(x,y)

        # Mean and mean absolute log ratio
        self.mean_log10_ratio          = self.mlr  = np.mean( log10ratio )
        self.mean_absolute_log10_ratio = self.malr = np.mean( np.abs( log10ratio ) )
        self.std_log10_ratio           = self.stdlr= np.std( log10ratio )

        # Median and median absolute log ratio
        self.median_log10_ratio          = self.medlr  = np.median( log10ratio )
        self.median_absolute_log10_ratio = self.medalr = np.median( np.abs( log10ratio ) )

        # RMS difference
        self.root_mean_square_difference = self.rmsd   = np.sqrt( np.mean( np.power( diff, 2) ) )
        # RMS log ratio
        self.root_mean_square_log10_ratio = self.rmslr = np.sqrt( np.mean( np.power( log10ratio, 2 )))

        # Covariance, correlation
        self.covariance = np.cov(x,y)[0][1]
        self.correlation = self.correlation_pearson = self.R = self.r = self.pearsonr = \
            np.corrcoef(x,y)[0][1]
        self.correlation_spearman = self.spearmanr = stats.spearmanr(x,y).statistic
        self.R2 = self.r2 = self.R**2

    def __getitem__(self,key):
        '''Accesses attribute values via object['key']'''
        return getattr(self,key)

    def fitline(self,method='sma',intercept=True,**kwargs):
        '''Compute bivariate line fit
        
        Parameters
        ----------
        method : str
            line fitting method: sma (default), ols, wls, York, sen, siegel
        intercept : bool
            defines whether non-zero intercept should be fitted
        **kwargs 
            passed to `acgc.stats.sma` (e.g. robust=True)

        Returns
        -------
        result : dict
            dictionary with keys:
            - slope (float)
                slope of fitted line
            - intercept (float)
                intercept of fitted line
            - fittedvalues (array (N,))
                values on fit line
            - residuals (array (N,))
                residual from fit line
        '''

        fitintercept = intercept

        if method.lower()=='sma':
            fit = sma(  self._x,
                        self._y,
                        self._w,
                        intercept=fitintercept,
                        **kwargs)
            slope = fit['slope']
            intercept= fit['intercept']

        elif method.lower()=='ols':
            if fitintercept:
                ols = np.linalg.lstsq( np.vstack([self._x,np.ones(len(self._x))]).T,
                                      self._y, rcond=None )
            else:
                ols = np.linalg.lstsq( np.vstack([self._x]).T, self._y, rcond=None )
            slope = ols[0][0]
            intercept = ols[0][1]

        elif method.lower() in ['theil','sen','theilsen']:
            fitintercept = True
            fit = sen( self._x,
                       self._y,
                       **kwargs)
            slope = fit.slope
            intercept = fit.intercept

        elif method.lower()=='siegel':
            fitintercept = True
            siegel = stats.siegelslopes( self._x,
                                         self._y )
            slope = siegel.slope
            intercept = siegel.intercept

        elif method.lower()=='wls':
            raise NotImplementedError('WLS regression not implemented yet')

        elif method.lower()=='york':
            raise NotImplementedError('York regression not implemented yet')

        else:
            raise ValueError('Undefined method '+method)

        line = dict( slope          = slope,
                     intercept      = intercept,
                     fittedvalues   = slope * self._x + intercept,
                     residuals      = self._y - ( slope * self._x + intercept ),
                     method         = method,
                     fitintercept   = fitintercept )

        return line

    def slope(self,method='sma',intercept=True,**kwargs):
        '''Compute slope of bivariate line fit
        
        Parameters
        ----------
        method : str
            line fitting method: sma (default), ols, wls
        intercept : bool
            defines whether non-zero intercept should be fitted
        **kwargs 
            passed to `fitline`

        Returns
        -------
        slope : float
            value of y intercept
        '''
        return self.fitline(method,intercept,**kwargs)['slope']

    def intercept(self,method='sma',intercept=True,**kwargs):
        '''Compute intercept of bivariate line fit
        
        Parameters
        ----------
        method : str
            line fitting method: sma (default) or ols
        intercept : bool
            defines whether non-zero intercept should be fitted
        **kwargs 
            passed to `fitline`

        Returns
        -------
        intercept : float
            value of y intercept
        '''
        return self.fitline(method,intercept,**kwargs)['intercept']

    def _expand_variables(self,variables):
        '''Expand special strings into a list of variables
        
        Parameter
        ---------
        variables : list or str, default='common'
            Special strings ("all","common") will be expanded to a list of variables
            list arguments will not be modified

        Returns
        -------
        list 
            variable names
        '''
        if variables is None:
            variables='common'
        if variables=='all':
            variables=['MD','MAD','RMD','RMAD','MRD','SMD','SMAD',
                       'MLR','MALR',
                       'MedD','MedAD','RMedD','RMedAD','MedRD',
                       'MedLR','MedALR',
                       'NMBF','NMAEF','RMSD','cov',
                       'R','R2','spearmanr','slope','intercept',
                       'fitline','n']
        elif variables=='common':
            variables=['MD','MAD','RMD','RMAD','MRD','R2','slope','n']
        if not isinstance(variables,list):
            raise ValueError(
                'variables must be a list, None, or one of these strings: "all","common"')

        return variables

    def summary_dict(self, variables=None, fitline_kw=None, floatformat_fiteqn='{:.3f}'):
        '''Summarize bivariate statistics into a dict

        Parameters
        ----------
        vars : list or str, default='common'
            names of attribute variables to include in summary
            names are case insensitive            
            The following strings are also accepted in place of a list 
                "all" (displays all variables)
                "common" (displays all measures of mean difference)
        fitline_kw : dict, default=None
            keywords passed to `fitline`
        floatformat_fiteqn : str, default=floatformat
            format specifier for slope and intercept (a,b) in y = a x + b
        
        Returns
        -------
        summary : dict
            names and values of variables
        '''

        # List of variables
        variables = self._expand_variables(variables)

        if fitline_kw is None:
            fitline_kw = {'method':'sma',
                          'intercept':True}

        # Construct the dict
        summary = {}
        for v in variables:
            if v in ['slope','intercept']:
                # These variables are object methods
                func = getattr(self,v)
                value = func(**fitline_kw)
            elif v == 'fitline':
                line = self.fitline(**fitline_kw)
                v,value = bivariate_line_equation(line,floatformat_fiteqn,ystring='separate')
            else:
                # Retrieve values
                value = getattr(self,v.lower())

            # summary += (stringformat+'='+floatformat+'\n').format(v,value)
            summary[v] = value

        return summary

    def summary(self, variables=None, fitline_kw=None,
                intformat='{:d}', floatformat='{:.4f}', floatformat_fiteqn=None,
                stringlength=None ):
        '''Summarize bivariate statistics

        Parameters
        ----------
        vars : list or str, default='common'
            names of attribute variables to include in summary
            names are case insensitive            
            The following strings are also accepted in place of a list 
                "all" (displays all variables)
                "common" (displays all measures of mean difference)
        fitline_kw : dict, default=None
            keywords passed to `fitline`
        intformat : str, default='{:d}'
            format specifier for integer values
        floatformat : str, default='{:.4f}'
            format specifier for floating point values
        floatformat_fiteqn : str, default=floatformat
            format specifier for slope and intercept (a,b) in y = a x + b
        stringlength : int, default=None
            length of the variables on output
            default (None) is to use the length of the longest variable name
        
        Returns
        -------
        summary : str
            names and values of variables
        '''
        # List of variables
        variables = self._expand_variables(variables)

        if floatformat_fiteqn is None:
            floatformat_fiteqn = floatformat
        if stringlength is None:
            stringlength = np.max([len(v) for v in variables])
        stringformat = '{:'+str(stringlength)+'s}'

        # Get a dict containing the needed variables
        summarydict = self.summary_dict( variables, fitline_kw, floatformat_fiteqn )

        # Extract length of the float numbers from floatformat
        # import re
        # floatlength = np.floor( float( re.findall("[-+]?(?:\d*\.*\d+)",
        #       floatformat )[0] ) ).astype(int)

        # summary = (stringformat+'{:>10s}').format('Variable','Value')
        summarytext = ''
        for k,v in summarydict.items():
            vstr = _number2str(v,intformat,floatformat)
            summarytext += (stringformat+' = {:s}\n').format(k,vstr)

        return summarytext

    def summary_fig_inset(self, ax, variables=None, fitline_kw=None,
                          intformat='{:d}', floatformat='{:.3f}', floatformat_fiteqn=None,
                          loc=None, loc_units='axes',
                          **kwargs):
        '''Display bivariate statistics as a table inset on a plot axis

        Parameters
        ----------
        ax : matplotlib.Figure.Axis 
            axis where the table will be displayed
        variables : list or str, default='common'
            names of attribute variables to include in summary
            names are case insensitive            
            The following strings are also accepted in place of a list 
                "all" (displays all variables)
                "common" (displays all measures of mean difference)
        fitline_kw : dict, default=None
            keywords passed to `fitline`
        intformat : str, default='{:d}'
            format specifier for integer values
        floatformat : str, default='{:.3f}'
            format specifier for floating point values
        floatformat_fiteqn : str, default=floatformat
            format specifier for slope and intercept (a,b) in y = a x + b
        loc : tuple (x0,y0), default=(0.85, 0.05)
            location on the axis where the table will be drawn
            can be in data units or axes units [0-1]
        loc_units : {'axes' (default), 'data'}
            specifies whether loc has 'data' units or 'axes' units [0-1]
                    
        Returns
        -------
        text1, text2 : matplotlib text object
            Artist for the two text boxes        
        '''
        # List of variables
        variables = self._expand_variables(variables)

        if floatformat_fiteqn is None:
            floatformat_fiteqn = floatformat

        # Default location in lower right corner
        if loc is None:
            loc = (0.8,0.05)

        # Coordinates for loc
        if loc_units.lower()=='data':
            coord=ax.transData
        elif loc_units.lower() in ['axes','axis']:
            coord=ax.transAxes
        else:
            raise ValueError('Display units should be "Data" or "Axes"')

        # Get a dict containing the needed variables
        summarydict = self.summary_dict( variables, fitline_kw, floatformat_fiteqn )

        # Column of label text
        label_text = '\n'.join([_texify_name(key)
                                for key in summarydict])
        # Column of value text
        value_text = '\n'.join([_number2str(v,intformat,floatformat)
                                for v in summarydict.values()])

        # Check if horizontal alignment keyword is used
        ha=''
        try:
            ha = kwargs['ha']
        except KeyError:
            pass
        try:
            ha = kwargs['horizontalalignment']
        except KeyError:
            pass

        # For right alignment, align on values first
        # Otherwise, align on labels
        if ha=='right':
            first_text = value_text
            second_text = label_text
            sign = -1
        else:
            first_text = label_text
            second_text = value_text
            sign = +1

        # Add first column of text
        t1=ax.text(loc[0],loc[1],
                first_text,
                transform=coord,
                **kwargs
                )

        # Get width of first text column
        bbox = t1.get_window_extent().transformed(coord.inverted())
        width = bbox.x1-bbox.x0

        # Add second column of text
        t2 = ax.text(loc[0]+width*sign,loc[1],
                     second_text,
                     transform=coord,
                     **kwargs
                     )

        ##################################
        # Early version of this function using matplotlib.table.table()

        # if isinstance(loc,(tuple,list)):
        #     # Create an inset axis to contain the table
        #     tableaxis = ax.inset_axes(loc)
        #     table_width=1
        # else:
        #     tableaxis = ax

        # # Display the table on the axis
        # return mtable.table(
        #     tableaxis,
        #     cellText=[[floatformat.format(value)] for value in summarydict.values()],
        #     rowLabels=[texify_name(key) for key in summarydict],
        #     colWidths=[table_width/2]*2,
        #     edges=edges,
        #     loc=loc, bbox=bbox
        #     )

        return [t1,t2]
