#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Bivariate statistics

Statistical measures of relationships between two populations


Created on June 4 2019

@author: cdholmes
"""

import numpy as np
from scipy import stats
from .bivariate_lines import sma
# import xarray as xr

__all__ = [
    "BivariateStatistics",
    "nmb",
    "nmae",
    "nmbf",
    "nmaef"
]

def nmb( ref_values, exp_values ):
    '''Compute Normalized Mean Bias (NMB)
    nmb = (<exp> - <ref>) / <ref>

    Parameters
    ----------
    ref_values : reference values
    exp_values : experiment values
    '''

    assert (len(exp_values) == len(ref_values)), \
        "exp_values must have the same length as ref_values"

    # Mean values
    ref_mean = np.mean(ref_values)
    exp_mean = np.mean(exp_values)

    # Metric value
    return exp_mean / ref_mean - 1

def nmae( ref_values, exp_values ):
    '''Compute Normalized Mean Absolute Error (NMAE)
    < |exp - ref| > / |<ref>|

    Parameters
    ---------
    ref_values : reference values
    exp_values : experiment values
    '''

     # Mean values
    ref_mean = np.mean(ref_values)

    # Mean absolute difference
    abs_diff = np.mean( np.abs(exp_values - ref_values) )

    # Metric value
    return abs_diff / np.abs( ref_mean )


def nmbf( ref_values, exp_values ):
    '''Compute Normalized Mean Bias Factor (NMBF)
    Definition from Yu et al. (2006) https://doi.org/10.1002/asl.125

    Parameters
    ----------
    ref_values : reference values
    exp_values : experiment values
    '''

    # Ensure that arguments have the same length
    assert (len(exp_values) == len(ref_values)), \
        "exp_values must have the same length as ref_values"

    # Mean values
    ref_mean = np.mean(ref_values)
    exp_mean = np.mean(exp_values)

    # Metric value
    if exp_mean >= ref_mean:
        result = exp_mean / ref_mean - 1
    else:
        result= 1 - ref_mean / exp_mean
    # Equivalent (faster?) implementation
    #S = (mMean - oMean) / np.abs(mMean - oMean)
    #result = S * ( np.exp( np.abs( mMean / oMean )) - 1 )

    return result

def nmaef( ref_values, exp_values ):
    '''Compute Normalized Mean Absolute Error Factor (NMAEF)
    Definition from Yu et al. (2006) https://doi.org/10.1002/asl.125
    
    Parameters
    ----------
    ref_values : reference values
    exp_values : experiment values
    '''

    # Ensure that arguments have the same length
    assert (len(exp_values) == len(ref_values)), \
        "exp_values must have the same length as ref_values"

    # Mean values
    ref_mean = np.mean(ref_values)
    exp_mean = np.mean(exp_values)

    # Mean absolute difference
    abs_diff = np.mean( np.abs(exp_values - ref_values))

    # Metric value
    if exp_mean >= ref_mean:
        result = abs_diff / ref_mean 
    else:
        result = abs_diff / exp_mean
    # Equivalent (faster?) implementation
    #S = (exp_mean - ref_mean) / np.abs(exp_mean - ref_mean)
    #result = abs_diff / ( oMean**((1+S)/2) * mMean**((1-S)/2) )

    return result

def _texify_name(name):
    '''Return a LaTex formatted string for some variables
    
    Parameter
    ---------
    name : str
    
    Returns
    -------
    pretty_name : str
    '''
    if name=='R2':
        pretty_name = f'$R^2$'
    elif name=='r2':
        pretty_name = f'$r^2$'
    else:
        pretty_name = name
    return pretty_name

class BivariateStatistics:
    '''A suite of common statistics to quantify bivariate relationships

    Class method 'summary' provides a formatted summary of these statistics
    
    Attributes:
        xmean, ymean (float) : mean of x and y variables
        xmedian, ymedian (float) : median of x and y variables
        xstd, ystd (float): standard deviation of x and y variables

        mean_difference, md (float) : ymean - xmean
        mean_absolute_difference, mad (float) : <|y-x|>
        relative_mean_difference, rmd (float) : md / xmean
        relative_mean_absolute_difference, rmad (float) : mad / xmean
        standardized_mean_difference, smd (float) : md / xstd
        standardized_mean_absolute_difference, smad (float) : mad /xstd
        mean_relative_difference, mrd (float) : <y/x> - 1
        
        median_difference, medd (float) : median(y-x)
        median_absolute_difference, medad (float) : median(|y-x|)
        relative_median_difference, rmedd (float) : median(y-x) / xmedian
        relative_median_absolute_difference, rmedad (float) : median(|y-x|) / xmedian
        median_relative_difference, medianrd, medrd (float) : median(y/x)-1
        
        normalized_mean_bias_factor, nmbf (float) : see nmbf function
        normalized_mean_absolute_error_factor, nmaef (float) : see nmaef

        root_mean_square_difference, rmsd (float) :
        covariance (float) : cov(x,y)
        correlation_pearson, correlation, pearsonr, R, r (float) : 
        correlation_spearman, spearmanr (float) :
        R2, r2 (float) : correlation_pearson**2
        '''

    def __init__(self,x,y,w=None):
        '''Compute suite of bivariate statistics during initialization
        
        Statistic values are save in attributes.
        CAUTION: Weights w are ignored except in SMA fit

        Parameters
        ----------
        x : ndarray
            independent variable values
        y : ndarray
            dependent variable values, same size as x
        w : ndarray (optional)
            weights for points (x,y), same size as x and y
        '''

        #Ensure that x and y have same length
        if len(x) != len(y):
            raise ValueError( 'Arguments x and y must have the same length' )
        if (w is not None) and (len(w) != len(x)):
            raise ValueError( 'Argument w (if present) must have the same length as x' )

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
        self.median_relative_difference = self.medianrd = self.medrd = np.median( ratio - 1 )

        # Median and median absolute differences
        self.median_difference          = self.medd  = np.median( diff )
        self.median_absolute_difference = self.medad = np.median( absdiff )

        # Relative median differences
        self.relative_median_difference          = self.rmedd  = self.median_difference / self.xmedian
        self.relative_median_absolute_difference = self.rmedad = self.median_absolute_difference / self.xmedian

        self.normalized_mean_bias_factor            = self.nmbf  = nmbf(x,y)
        self.normalized_mean_absolute_error_factor  = self.nmaef = nmaef(x,y)

        # RMS difference
        self.root_mean_square_difference    = self.rmsd     = np.sqrt( np.mean( np.power( diff, 2) ) )

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
        **kwargs passed to sma (e.g. robust=True)

        Returns
        -------
        Result : dict containing keys
            slope : slope of fitted line
            intercept : intercept of fitted line
            fittedvalues : values on fit line
            residuals : residual from fit line
        '''

        if method.lower()=='sma':
            fit = sma(  self._x,
                        self._y,
                        self._w,
                        intercept=intercept,
                        **kwargs)
            slope = fit['slope']
            intercept= fit['intercept']

        elif method.lower()=='ols':
            if intercept:
                ols = np.linalg.lstsq( np.vstack([self._x,np.ones(len(self._x))]).T, 
                                      self._y, rcond=None )
            else:
                ols = np.linalg.lstsq( np.vstack([self._x]).T, self._y, rcond=None )
            slope = ols[0][0]
            intercept = ols[0][1]

        elif method.lower() in ['theil','sen','theilsen']:
            sen = stats.theilslopes( self._y,
                                     self._x )
            slope = sen.slope
            intercept = sen.intercept

        elif method.lower()=='siegel':
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
                     residuals      = self._y - ( slope * self._x + intercept ) )

        return line

    def slope(self,method='sma',intercept=True,**kwargs):
        '''Compute slope of bivariate line fit
        
        Parameters
        ----------
        method : str
            line fitting method: sma (default), ols, wls
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
        
        Parameters
        ----------
        method : str
            line fitting method: sma (default) or ols
        intercept : bool
            defines whether non-zero intercept should be fitted
        **kwargs passed to fitline()

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
        variables : list, str or None (default='common')
            Special strings ("all","common") will be expanded to a list of variables
            list arguments will not be modified

        Returns
        -------
        list of variable names
        '''
        if variables is None:
            variables='common'
        if variables=='all':
            variables=['MD','MAD','RMD','RMAD','MRD','SMD','SMAD',
                       'MedD','MedAD','RMedD','RMedAD','MedRD',
                       'NMBF','NMAEF','RMSD',
                       'R','R2','spearmanr','slope','intercept']
        elif variables=='common':
            variables=['MD','MAD','RMD','RMAD','MRD','R2','slope']
        if not isinstance(variables,list):
            raise ValueError(
                'variables must be a list, None, or one of these strings: "all","common"')

        return variables

    def summary_dict(self, variables=None, fitline_kw=None ):
        '''Summarize bivariate statistics into a dict

        Parameters
        ----------
        vars : list, None, or str (default='common')
            names of attribute variables to include in summary
            names are case insensitive            
            The following strings are also accepted in place of a list 
                "all" (displays all variables)
                "common" (displays all measures of mean difference)
        fitline_kw : dict (default=None)
            keywords passed to self.fitline()
        
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
            else:
                # Retrieve values
                value = getattr(self,v.lower())

            # summary += (stringformat+'='+floatformat+'\n').format(v,value)
            summary[v] = value

        return summary

    def summary(self, variables=None, fitline_kw=None, 
                floatformat='{:.4f}', stringlength=None ):
        '''Summarize bivariate statistics

        Parameters
        ----------
        vars : list, None, or str (default='common')
            names of attribute variables to include in summary
            names are case insensitive            
            The following strings are also accepted in place of a list 
                "all" (displays all variables)
                "common" (displays all measures of mean difference)
        floatformat : str (default='{:.4f}')
            format specifier for floating point values
        stringlength : int (default=None)
            length of the variables on output
            default (None) is to use the length of the longest variable name
        fitline_kw : dict (default=None)
            keywords passed to self.fitline()
        
        Returns
        -------
        summary : str
            names and values of variables
        '''
        # List of variables
        variables = self._expand_variables(variables)

        if stringlength is None:
            stringlength = np.max([len(v) for v in variables])
        stringformat = '{:'+str(stringlength)+'s}'

        # Get a dict containing the needed variables
        summarydict = self.summary_dict( variables, fitline_kw )

        # Extract length of the float numbers from floatformat
        # import re
        # floatlength = np.floor( float( re.findall("[-+]?(?:\d*\.*\d+)",
        #       floatformat )[0] ) ).astype(int)

        # summary = (stringformat+'{:>10s}').format('Variable','Value')
        summarytext = ''
        for k,v in summarydict.items():
            summarytext += (stringformat+' = '+floatformat+'\n').format(k,v)

        return summarytext

    def summary_fig_table(self, ax, variables=None, fitline_kw=None,
                          floatformat='{:.3f}',
                          loc=None, loc_units='data',
                          **kwargs):
        '''Display bivariate statistics as a table on a plot axis

        Parameters
        ----------
        ax : matplotlib.Figure.Axis 
            axis where the table will be displayed
        variables : list, None, or str (default='common')
            names of attribute variables to include in summary
            names are case insensitive            
            The following strings are also accepted in place of a list 
                "all" (displays all variables)
                "common" (displays all measures of mean difference)
        fitline_kw : dict (default=None)
            keywords passed to self.fitline()
        floatformat : str (default='{:.3f}')
            format specifier for floating point values
        loc : tuple (x0,y0) (default=(0.85, 0.05))
            location on the axis where the table will be drawn 
        loc_units : str (default='data')
            specifies whether loc has 'data' units or 'axes' units [0-1]
                    
        Returns
        -------
        Table 
            Artist for the created table        
        '''
        # List of variables
        variables = self._expand_variables(variables)

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
        summarydict = self.summary_dict( variables, fitline_kw )

        # Column of label text
        label_text = '\n'.join([_texify_name(key) for key in summarydict])
        # Column of value text
        value_text = '\n'.join([floatformat.format(value) for value in summarydict.values()])

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
