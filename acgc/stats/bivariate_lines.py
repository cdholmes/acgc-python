# -*- coding: utf-8 -*-
"""Specialized methods of bivariate line fitting

* Standard major axis (SMA) also called reduced major axis (RMA)
* York regression, for data with errors in x and y
* Theil-Sen, non-parametric slope estimation (use numba to accelerate the function in this module)
"""
# from collections import namedtuple
import warnings
import numpy as np
import scipy.stats as stats
from sklearn.covariance import MinCovDet
import statsmodels.formula.api as smf
import statsmodels.robust.norms as norms
from scipy.stats import theilslopes
#from numba import jit

__all__ = [
    "bivariate_line_equation",
    "sma",
    "smafit",
    "sen",
    "sen_slope",
    "sen_numba",
    "york"
]
# Aliases
def sen_slope(*args,**kwargs):
    '''Alias for `sen`'''
    return sen(*args,**kwargs)
def smafit(*args,**kwargs):
    '''Alias for `sma`'''
    return sma(*args,**kwargs)

def bivariate_line_equation(fitresult,
                    floatformat='{:.3f}',
                    ystring='include',
                    include_error=False ):
    '''Write equation for the fitted line as a string
    
    Parameters
    ----------
    fitresult : dict
        results of the line fit
    floatformat : str
        format string for the numerical values (default='{:.3f}')
    ystring : {'include' (default), 'separate', 'none'}
        specifies whether "y =" should be included in result, a separate item in tuple, or none
    include_error : bool
        specifies whether uncertainty terms should be included in the equation
    
    Returns
    -------
    fitline_string : str
        equation for the the fitted line, in the form "y = a x + b" or "y = a x"
        If uncertainty terms are included, then "y = (a ± c) x + (b ± d)" or "y = (a ± c) x"
    '''

    # Left-hand side
    lhs = "y_"+fitresult['method']

    # Right-hand side
    if fitresult['fitintercept']:
        if include_error:
            rhs = f'({floatformat:s} ± {floatformat:s}) x + ({floatformat:s} ± {floatformat:s})'.\
                    format( fitresult['slope'], fitresult['slope_ste'], fitresult['intercept'], fitresult['intercept_ste'] )
        else:
            rhs = f'{floatformat:s} x + {floatformat:s}'.\
                    format( fitresult['slope'], fitresult['intercept'] )
    else:
        if include_error:
            rhs = f'({floatformat:s} ± {floatformat:s}) x'.\
                    format( fitresult['slope'], fitresult['slope_ste'] )
        else:
            rhs = f'{floatformat:s} x'.\
                    format( fitresult['slope'] )

    # Combine right and left-hand sides
    if ystring=='include':
        equation = f'{lhs:s} = {rhs:s}'
    elif ystring=='separate':
        equation = (lhs,rhs)
    elif ystring=='none':
        equation = rhs
    else:
        raise ValueError('Unrecognized value of ystring: '+ystring)

    return equation

def sma(X,Y,W=None,
           data=None,
           alpha=0.95,
           intercept=True,
           robust=False,robust_method='FastMCD'):
    '''Standard Major-Axis (SMA) line fitting
    
    Calculate standard major axis, aka reduced major axis, fit to 
    data X and Y. The main advantage of this over ordinary least squares is 
    that the best fit of Y to X will be the same as the best fit of X to Y.
    
    The fit equations and confidence intervals are implemented following 
    Warton et al. (2006). Robust fits use the FastMCD covariance estimate 
    from Rousseeuw and Van Driessen (1999). While there are many alternative 
    robust covariance estimators (e.g. other papers by D.I. Warton using M-estimators), 
    the FastMCD algorithm is default in Matlab. When the standard error or 
    uncertainty of each point is known, then weighted SMA may be preferrable to 
    robust SMA. The conventional choice of weights for each point i is 
    W_i = 1 / ( var(X_i) + var(Y_i) ), where var() is the variance 
    (squared standard error).
    
    References 
    Warton, D. I., Wright, I. J., Falster, D. S. and Westoby, M.: 
        Bivariate line-fitting methods for allometry, Biol. Rev., 81(02), 259, 
        doi:10.1017/S1464793106007007, 2006.
    Rousseeuw, P. J. and Van Driessen, K.: A Fast Algorithm for the Minimum 
        Covariance Determinant Estimator, Technometrics, 41(3), 1999.

    Parameters
    ----------
    X, Y : array_like or str
        Input values, Must have same length.
    W    : array_like or str, optional
        array of weights for each X-Y point, typically W_i = 1/(var(X_i)+var(Y_i)) 
    data : dict_like, optional
        data structure containing variables. Used when X, Y, or W are str.
    alpha : float (default = 0.95)
        Desired confidence level [0,1] for output. 
    intercept : bool, default=True
        Specify if the fitted model should include a non-zero intercept.
        The model will be forced through the origin (0,0) if intercept=False.
    robust : bool, default=False
        Use statistical methods that are robust to the presence of outliers
    robust_method: {'FastMCD' (default), 'Huber', 'Biweight'}
        Method for calculating robust variance and covariance. Options:
        - 'MCD' or 'FastMCD' for Fast MCD
        - 'Huber' for Huber's T: reduce, not eliminate, influence of outliers
        - 'Biweight' for Tukey's Biweight: reduces then eliminates influence of outliers

        
    Returns
    -------
    fitresult : dict 
        Contains the following keys:
        - slope (float)
            Slope or Gradient of Y vs. X
        - intercept (float)
            Y intercept.
        - slope_ste (float)
            Standard error of slope estimate
        - intercept_ste (float)
            standard error of intercept estimate
        - slope_interval ([float, float])
            confidence interval for gradient at confidence level alpha
        - intercept_interval ([float, float])
            confidence interval for intercept at confidence level alpha
        - alpha (float)
            confidence level [0,1] for slope and intercept intervals
        - df_model (float)
            degrees of freedom for model
        - df_resid (float)
            degrees of freedom for residuals
        - params ([float,float])
            array of fitted parameters
        - fittedvalues (ndarray)
            array of fitted values
        - resid (ndarray)
            array of residual values
        - method (str)
            name of the fit method
    '''

    def str2var( v, data ):
        '''Extract variable named v from Dataframe named data'''
        try:
            return data[v]
        except Exception as exc:
            raise ValueError( 'Argument data must be provided with a key named '+v ) from exc

    # If variables are provided as strings, get values from the data structure
    if isinstance( X, str ):
        X = str2var( X, data )
    if isinstance( Y, str ):
        Y = str2var( Y, data )
    if isinstance( W, str ):
        W = str2var( W, data )

    # Make sure arrays have the same length
    assert ( len(X) == len(Y) ), 'Arrays X and Y must have the same length'
    if W is None:
        W = np.zeros_like(X) + 1
    else:
        assert ( len(W) == len(X) ), 'Array W must have the same length as X and Y'

    # Make sure alpha is within the range 0-1
    assert (alpha < 1), 'alpha must be less than 1'
    assert (alpha > 0), 'alpha must be greater than 0'

    # Drop any NaN elements of X, Y, or W
    # Infinite values are allowed but will make the result undefined
    # idx = ~np.logical_or( np.isnan(X0), np.isnan(Y0) )
    idx = ~np.isnan(X) * ~np.isnan(Y) * ~np.isnan(W)

    X0 = X[idx]
    Y0 = Y[idx]
    W0 = W[idx]

    # Number of observations
    N = len(X0)

    include_intercept = intercept

    # Degrees of freedom for the model
    if include_intercept:
        dfmod = 2
    else:
        dfmod = 1

    method = 'SMA'

    # Choose whether to use methods robust to outliers
    if robust:

        method = 'rSMA'

        # Choose the robust method
        if ((robust_method.lower() =='mcd') or (robust_method.lower() == 'fastmcd') ):
            # FAST MCD

            if not include_intercept:
                # intercept=False could possibly be supported by calculating
                # using mcd.support_ as weights in an explicit variance/covariance calculation
                raise NotImplementedError('FastMCD method only supports SMA with intercept')

            # Fit robust model of mean and covariance
            mcd = MinCovDet().fit( np.array([X0,Y0]).T )

            # Robust mean
            Xmean = mcd.location_[0]
            Ymean = mcd.location_[1]

            # Robust variance of X, Y
            Vx    = mcd.covariance_[0,0]
            Vy    = mcd.covariance_[1,1]

            # Robust covariance
            Vxy   = mcd.covariance_[0,1]

            # Number of observations used in mean and covariance estimate
            # excludes observations marked as outliers
            N = mcd.support_.sum()

        elif ((robust_method.lower() =='biweight') or (robust_method.lower() == 'huber') ):

            # Tukey's Biweight and Huber's T
            if robust_method.lower()=='biweight':
                norm = norms.TukeyBiweight()
            else:
                norm = norms.HuberT()

            # Get weights for downweighting outliers
            # Fitting a linear model the easiest way to get these
            # Options include "TukeyBiweight" (totally removes large deviates)
            # "HuberT" (linear, not squared weighting of large deviates)
            rweights = smf.rlm('y~x+1',{'x':X0,'y':Y0},M=norm).fit().weights

            # Sum of weight and weights squared, for convienience
            rsum  = np.sum( rweights )
            rsum2 = np.sum( rweights**2 )

            # Mean
            Xmean = np.sum( X0 * rweights ) / rsum
            Ymean = np.sum( Y0 * rweights ) / rsum

            # Force intercept through zero, if requested
            if not include_intercept:
                Xmean = 0
                Ymean = 0

            # Variance & Covariance
            Vx    = np.sum( (X0-Xmean)**2 * rweights**2 ) / rsum2
            Vy    = np.sum( (Y0-Ymean)**2 * rweights**2 ) / rsum2
            Vxy   = np.sum( (X0-Xmean) * (Y0-Ymean) * rweights**2 ) / rsum2

            # Effective number of observations
            N = rsum

        else:

            raise NotImplementedError("sma hasn't implemented robust_method={:%s}".\
                                      format(robust_method))
    else:

        if include_intercept:

            wsum = np.sum(W)

            # Average values
            Xmean = np.sum(X0 * W0) / wsum
            Ymean = np.sum(Y0 * W0) / wsum

            # Covariance matrix
            cov = np.cov( X0, Y0, ddof=1, aweights=W0**2 )

            # Variance
            Vx = cov[0,0]
            Vy = cov[1,1]

            # Covariance
            Vxy = cov[0,1]

        else:

            # Force the line to pass through origin by setting means to zero
            Xmean = 0
            Ymean = 0

            wsum = np.sum(W0)

            # Sum of squares in place of variance and covariance
            Vx = np.sum( X0**2 * W0 ) / wsum
            Vy = np.sum( Y0**2 * W0 ) / wsum
            Vxy= np.sum( X0*Y0 * W0 ) / wsum

    # Standard deviation
    Sx = np.sqrt( Vx )
    Sy = np.sqrt( Vy )

    # Correlation coefficient (equivalent to np.corrcoef()[1,0] for non-robust cases)
    R = Vxy / np.sqrt( Vx * Vy )

    #############
    # SLOPE

    Slope  = np.sign(R) * Sy / Sx

    # Standard error of slope estimate
    ste_slope = np.sqrt( 1/(N-dfmod) * Sy**2 / Sx**2 * (1-R**2) )

    # Confidence interval for Slope
    B = (1-R**2)/(N-dfmod) * stats.f.isf(1-alpha, 1, N-dfmod)
    ci_grad = Slope * ( np.sqrt( B+1 ) + np.sqrt(B)*np.array([-1,+1]) )

    #############
    # INTERCEPT

    if include_intercept:
        Intercept = Ymean - Slope * Xmean

        # Standard deviation of residuals
        # New Method: Formula from smatr R package (Warton)
        # This formula avoids large residuals of outliers when using robust=True
        Sr = np.sqrt((Vy - 2 * Slope * Vxy + Slope**2 *  Vx ) * (N-1) / (N-dfmod) )

        # OLD METHOD
        # Standard deviation of residuals
        #resid = Y0 - (Intercept + Slope * X0 )
        # Population standard deviation of the residuals
        #Sr = np.std( resid, ddof=0 )

        # Standard error of the intercept estimate
        ste_int = np.sqrt( Sr**2/N + Xmean**2 * ste_slope**2  )

        # If Intercept and ste_int are DataArrays, convert to float
        if hasattr(Intercept,'values'):
            Intercept = Intercept.values
        if hasattr(ste_int,'values'):
            ste_int = ste_int.values

        # Confidence interval for Intercept
        tcrit = stats.t.isf((1-alpha)/2,N-dfmod)
        ci_int = Intercept + ste_int * np.array([-tcrit,tcrit])

    else:

        # Set Intercept quantities to zero
        Intercept = 0
        ste_int   = 0
        ci_int    = np.array([0,0])

    result = dict( method           = method,
                   fitintercept     = include_intercept,
                   slope            = Slope,
                   intercept        = Intercept,
                   slope_ste        = ste_slope,
                   intercept_ste    = ste_int,
                   slope_interval   = ci_grad,
                   intercept_interval = ci_int,
                   alpha            = alpha,
                   df_model         = dfmod,
                   df_resid         = N-dfmod,
                   params           = np.array([Slope,Intercept]),
                   nobs             = N,
                   fittedvalues     = Intercept + Slope * X0,
                   resid            = Intercept + Slope * X0 - Y0 )

    # return Slope, Intercept, ste_slope, ste_int, ci_grad, ci_int
    return result

def york( x, y, err_x=1, err_y=1, rerr_xy=0 ):
    '''York regression accounting for error in x and y
    Follows the notation and algorithm of York et al. (2004) Section III
    
    Parameters
    ----------
    x, y : ndarray
        dependent (x) and independent (y) variables for fitting
    err_x, err_y : ndarray (default=1)
        standard deviation of errors/uncertainty in x and y
    rerr_xy : float (default=0)
        correlation coefficient for errors in x and y, 
        default to rerr_xy=0 meaning that the errors in x are unrelated to errors in y
        err_x, err_y, and rerr_xy can be constants or arrays of the same length as x and y
    
    Returns
    -------
    fitresult : dict 
        Contains the following keys:
        - slope (float)
            Slope or Gradient of Y vs. X
        - intercept (float)
            Y intercept.
        - slope_ste (float)
            Standard error of slope estimate
        - intercept_ste (float)
            standard error of intercept estimate
        - slope_interval ([float, float])
            confidence interval for gradient at confidence level alpha
        - intercept_interval ([float, float])
            confidence interval for intercept at confidence level alpha
        - alpha (float)
            confidence level [0,1] for slope and intercept intervals
        - df_model (float)
            degrees of freedom for model
        - df_resid (float)
            degrees of freedom for residuals
        - params ([float,float])
            array of fitted parameters
        - fittedvalues (ndarray)
            array of fitted values
        - resid (ndarray)
            array of residual values
    '''

    # relative error tolerance required for convergence
    rtol = 1e-15

    # Initial guess for slope, from ordinary least squares
    result = stats.linregress( x, y )
    b = result[0]

    # Weights for x and y
    wx = 1 / err_x**2
    wy = 1 / err_y**2

    # Combined weights
    alpha = np.sqrt( wx * wy )

    # Iterate until solution converges, but not more 50 times
    maxiter=50
    for i in range(1,maxiter):

        # Weight for point i
        W = wx * wy / ( wx + b**2 * wy - 2 * b * rerr_xy * alpha )
        Wsum = np.sum( W )

        # Weighted means
        Xbar = np.sum( W * x ) / Wsum
        Ybar = np.sum( W * y ) / Wsum

        # Deviation from weighted means
        U = x - Xbar
        V = y - Ybar

        # parameter needed for slope
        beta = W * ( U / wy + b*V / wx - (b*U + V) * rerr_xy / alpha )

        # Update slope estimate
        bnew = np.sum( W * beta * V ) / np.sum( W * beta * U )

        # Break from loop if new value is very close to old value
        if np.abs( (bnew-b)/b ) < rtol:
            break
        else:
            b = bnew

    if i==maxiter:
        raise ValueError( f'York regression failed to converge in {maxiter:d} iterations' )

    # Intercept
    a = Ybar - b * Xbar

    # least-squares adjusted points, expectation values of X and Y
    xa = Xbar + beta
    ya = Ybar + b*beta

    # Mean of adjusted points
    xabar = np.sum( W * xa ) / Wsum
    yabar = np.sum( W * ya ) / Wsum

    # Devaiation of adjusted points from their means
    u = xa - xabar
    v = ya - yabar

    # Variance of slope and intercept estimates
    varb = 1 / np.sum( W * u**2 )
    vara = 1 / Wsum + xabar**2 * varb

    # Standard error of slope and intercept
    siga = np.sqrt( vara )
    sigb = np.sqrt( varb )

    # Define a named tuple type that will contain the results
    # result = namedtuple( 'result', 'slope intercept sigs sigi params sigma' )

    # Return results as a named tuple, User can access as a regular tuple too
    # return result( b, a, sigb, siga, [b,a], [sigb, siga] )

    dfmod = 2
    N = np.sum( ~np.isnan(x) * ~np.isnan(y) )

    result = dict( method        = 'York',
                fitintercept     = True,
                slope            = b,
                intercept        = a,
                slope_ste        = sigb,
                intercept_ste    = siga,
                slope_interval   = [None,None],
                intercept_interval = [None,None],
                alpha            = alpha,
                df_model         = dfmod,
                df_resid         = N-dfmod,
                params           = np.array([b,a]),
                nobs             = N,
                fittedvalues     = a + b * x,
                resid            = a + b * x - y )

    return result

def sen( x, y, alpha=0.95, method='separate' ):
    ''''Theil-Sen slope estimate
    
    This function wraps `scipy.stats.theilslopes` and provides
    results in the same dict format as the other line fitting methods 
    in this module
    
    Parameters
    ----------
    x, y : ndarray
        dependent (x) and independent (y) variables for fitting
    alpha : float (default = 0.95)
        Desired confidence level [0,1] for output. 
    method : {'separate' (default), 'joint'}
        Method for estimating intercept. 
        - 'separate' uses np.median(y) - slope * np.median(x)
        - 'joint' uses np.median( y - slope * x )
            
    Returns
    -------
    fitresult : dict 
        Contains the following keys:
        - slope (float)
            Slope or Gradient of Y vs. X
        - intercept (float)
            Y intercept.
        - slope_ste (float)
            Standard error of slope estimate
        - intercept_ste (float)
            standard error of intercept estimate
        - slope_interval ([float, float])
            confidence interval for gradient at confidence level alpha
        - intercept_interval ([float, float])
            confidence interval for intercept at confidence level alpha
        - alpha (float)
            confidence level [0,1] for slope and intercept intervals
        - df_model (float)
            degrees of freedom for model
        - df_resid (float)
            degrees of freedom for residuals
        - params ([float,float])
            array of fitted parameters
        - fittedvalues (ndarray)
            array of fitted values
        - resid (ndarray)
            array of residual values
    '''

    slope, intercept, low_slope, high_slope = theilslopes(y,x,alpha,method)

    dfmod = 2
    N = np.sum( ~np.isnan(x) * ~np.isnan(y) )

    result = dict( method        = 'Theil-Sen',
                fitintercept     = True,
                slope            = slope,
                intercept        = intercept,
                slope_ste        = None,
                intercept_ste    = None,
                slope_interval   = [low_slope,high_slope],
                intercept_interval = [None,None],
                alpha            = alpha,
                df_model         = dfmod,
                df_resid         = N-dfmod,
                params           = np.array([slope,intercept]),
                nobs             = N,
                fittedvalues     = intercept + slope * x,
                resid            = intercept + slope * x - y )

    return result

#@jit(nopython=True)
def sen_numba( x, y ):
    '''Estimate linear trend using the Thiel-Sen method
    
    This non-parametric method finds the median slope among all
    combinations of time points. 
    scipy.stats.theilslopes provides the same slope estimate, with  
    confidence intervals. However, this function is faster for 
    large datasets due to Numba 
    
    Parameters
    ----------
    x : array_like (N,)
        independent variable
    y : array_like (N,)
        dependent variable
    
    Returns
    -------
    sen : float
        the median slope
    slopes : array (N*N,)
        all slope estimates from all combinations of x and y
    '''

    with warnings.catch_warnings():
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f'Sen function is slow unless numba.jit is used. Use scipy.stats.theilslopes instead.',
                    DeprecationWarning, stacklevel=2)
        
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
