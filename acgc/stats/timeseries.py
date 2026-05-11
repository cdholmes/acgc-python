#!/usr/bin/env python3
'''timeseries analysis functions'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.nonparametric.smoothers_lowess import lowess

__all__ = ['STL_harmonic',
           'STL_smoother',
           'DecomposeResult_to_DataFrame',
           'boxcar',
           'boxcarpoly']

def STL_smoother( data, period=12, seasonal=7, seasonal_smooth=4, robust=False ):
    '''STL decomposition of time series into trend, seasonal, and residual components
    with additional smoothing of the seasonal component
    
    Parameters
    ----------
    data : array_like
        time series data to be decomposed
    period : float, optional
        number of observations per cycle, e.g. 12 for monthly data with yearly seasonality
    seasonal : float, optional
        length of the seasonal smoother, must be odd, default is 7
    seasonal_smooth : int, optional
        length of the seasonal smooth smoother, default is 4
    robust : bool, optional
        whether to use robust fitting to reduce the influence of outliers, default is False

    Returns
    -------
    result : DecomposeResult
        object containing the original data, seasonal component, trend component, residuals, and weights (if robust=True)
    '''

    result = STL(data, period=period, seasonal=seasonal, robust=robust).fit()
    
    #Additional smoothing of the seasonal component
    seasonal_smooth = lowess(result.seasonal, 
                             result.seasonal.index, 
                             frac=seasonal_smooth/len(data), 
                             return_sorted=False)
    seasonal_smooth = pd.Series(seasonal_smooth, 
                                index=result.seasonal.index,
                                name='seasonal')

    resid_adjusted = result.resid + (result.seasonal - seasonal_smooth)
    resid_adjusted = resid_adjusted.rename('resid')
    

    return DecomposeResult(data, 
                            seasonal_smooth, 
                            result.trend, 
                            resid_adjusted, 
                            weights=result.weights)

def STL_harmonic( y, x=None, data=None, 
         period=None, trend_ratio=1.5,
         nharmonics=4, 
         maxiter=4, rtol=1e-2, robust=False, verbose=False,
         **kwargs):
    '''Decompose time series into seasonal, trend, and residual components 
    using harmonic regression for the seasonal component and lowess smoothing/filtering to separate the trend and residual components.
    
    This is similar STL decomposition with the following differences:
    * The seasonal component is modeled using harmonic regression, which guarantees a smooth seasonal cycle. 
    * The seasonal cycle is fixed in this decomposition, while it can vary in STL. 
    * STL requires regularly spaced data, while this decomposition can handle irregularly spaced data, 
    as long as the time variable `x` is provided. Nevertheless, the quality is likely to be better if the time series has no large gaps.

    Like STL, the decomposition is iterative. The seasonal cycle is fitted to detredended data,
    then the seasonal cycle is subtracted from the data. Lowess is applied to the deseasonalized data to estimate
    the trend. The updated trend is subtracted from the original data and an updated seasonal cycle
    is fitted to the updated detrended data. This process is repeated until the seasonal cycle converges 
    or the maximum number of iterations is reached.

    Parameters
    ----------
    y : array-like, pandas.Series or str
        The time series to decompose. If a string, it is treated as a column name in `data`.
    x : array-like or str, optional
        The time variable corresponding to `y`. If a string, it is treated as a column name in `data`. 
        If None (default), the index of `y` is used if it has one, otherwise a range from 0 to len(y)-1 is used.
    data : DataFrame, optional
        A DataFrame containing the data. Required if `y` or `x` are strings referring to column names.
    period :  float or np.timedelta64 or pd.Timedelta
        The length of the seasonal cycle in the same units as `x`. 
        If `x` is datetime-like, `period` should be a np.timedelta64 or pandas.Timedelta, e.g. `pd.Timedelta(days=365.25)` for annual seasonal cycles.
        If `x` is in months, `period` should be 12, and if `x` is in days, then `period` should be 365.25.
    trend_ratio : float, optional
        The ratio of the trend smoother span to the seasonal cycle length (period). Default is 1.5, which means the trend smoother span will be 1.5 times the period.
    nharmonics : int, optional
        The number of harmonics to include in the seasonal model. Default is 4, which includes the annual, semiannual, triannual, and quadriannual harmonics for yearly seasonality.
    maxiter : int, optional
        The maximum number of iterations for the decomposition. Default is 4.
    rtol : float, optional
        The relative tolerance for convergence of the seasonal component. Default is 1e-2.
    robust : bool or int, optional
        Specifies whether robust lowess fitting should be used for trend smoothing.
        If False (default), no robust fitting is done. If True, 3 iterations of robust fitting are done within the lowess filter. 
        If an integer >= 1, lowess uses that many iterations of robust fitting.
    verbose : bool, optional
        If True, prints convergence information. Default is False.
    **kwargs :
        Additional keyword arguments to pass to the lowess function for trend smoothing. For example, `
    
    Returns
    -------
    DecomposeResult
        An object containing the original data, seasonal component, trend component, and residuals.
    '''

    if period is None:
        raise ValueError('Keyword `period` must be specified')

    # If y is a string, treat it as a column name in data
    if isinstance(y, str):
        try:
            y = data[y]
        except KeyError:
            raise ValueError(f'y must be a 1-D array or a column name in data. {y} not found in data.')

    # If x is a string, treat it as a column name in data.
    if isinstance(x, str):
        try:
            x = data[x]
        except KeyError:
            raise ValueError(f'x must be a 1-D array or a column name in data. {x} not found in data.')
    # If x is None, use the index of y or a range if y has no index.
    elif x is None:
        try:
            x = y.index
        except AttributeError:
            x = np.arange(len(y))

    # Ensure nharmonics is a positive integer
    if not isinstance(nharmonics, int) or nharmonics < 1:
        raise ValueError('nharmonics must be a positive integer')

    # Robust fitting iterations
    if robust is False:
        robust_iterations=1
    elif robust is True:
        robust_iterations=3
    elif isinstance(robust, int) and robust >= 1:
        robust_iterations=robust
        robust = True
    else:
        raise ValueError('Invalid value for robust parameter. Must be True, False, or an integer >= 1.')

    # Normalize x, so that it is the number of periods since the start of the time series
    # xnorm = ( x - np.min(x) ) / period
    xnorm = ( x - x[0] ) / period

    # Initialize trend and seasonal_old
    trend = np.ones_like(y)*np.mean(y)
    seasonal_old = np.zeros_like(y)
    converged = False

    for i in range(1,maxiter+1):
          
        # Detrend
        detrended = y - trend

        # Fit seasonal cycle with nharmonics harmonics (annual, semiannual, triannual, etc.)
        X = np.column_stack([func(2*i*np.pi*xnorm) for i in range(1, nharmonics+1) for func in (np.sin, np.cos)])
        if robust:
            res = sm.RLM(detrended, X, M=sm.robust.norms.TukeyBiweight()).fit()
        else:
            res = sm.OLS(detrended, X).fit()
        seasonal = res.predict(X)

        # Update trend by fitting lowess to data - seasonal
        trend = lowess(y-seasonal, xnorm, 
                       frac = trend_ratio/np.max(xnorm), it=robust_iterations,
                       #    frac=ratio*period/len(y), it=robust_iterations,
                       return_sorted=False, **kwargs)
        
        # Exit if seasonal cycle has converged
        if np.max(np.abs(seasonal - seasonal_old)) < rtol*np.mean(np.abs(seasonal)):
            converged = True
            break

        # Update seasonal_old for next iteration
        seasonal_old = seasonal    

    # Print convergence message if verbose
    if verbose:
        if converged:
            print(f'STL_harmonic converged in {i} iterations')
        else:
            print(f'STL_harmonic did not converge in {maxiter} iterations')

    # Compute residual
    resid = y - trend - seasonal

    # Weights are not computed in this implementation, so set to NaN
    weights = np.full_like(resid, np.nan)

    if isinstance(y, pd.Series):
        seasonal = pd.Series(seasonal, index=y.index, name='seasonal')
        trend = pd.Series(trend, index=y.index, name='trend')
        resid = pd.Series(resid, index=y.index, name='resid')
        weights = pd.Series(weights, index=y.index, name='weights')

    # return trend, seasonal, resid
    return DecomposeResult(y,
                           seasonal, 
                           trend, 
                           resid,
                           weights)

def DecomposeResult_to_DataFrame(result):
    '''Convert a DecomposeResult object to a pandas DataFrame.
    
    Parameters
    ----------
    result : DecomposeResult
    
    Returns
    -------
    DataFrame
        A DataFrame with columns 'original', 'seasonal', 'trend', 'residual', and 'weights'.
    '''
    try:
        index = result.observed.index
    except AttributeError:
        index = np.arange(len(result.data))
    return pd.DataFrame({'original': result.observed, 
                         'seasonal': result.seasonal, 
                         'trend': result.trend, 
                         'residual': result.resid, 
                         'weights': result.weights}, 
                         index=index)


def boxcarpoly( array, width, order=2, align='center'):
    '''Calculate a boxcar polynomial (i.e. running polynomial fit) of an array. 
    
    See `boxcar` for parameter definitions
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
    array : array of float (N,)
        1D array of values to average
    width : int
        number of elements to include in the average. 
    align : str
        specifies how the averaging window for each element of the output array
        aligns with the input array. 
        Values: center (default), forward (trailing elements), backward (leading elements)
    method : {'mean' (default), 'median', 'polynomial'}
        specifies the averaging kernel function
    order : int, default=2
        specifies the polynomial order to fit within each boxcar window.
        This has no effect unless `method`='polynomial'
        A parabola is order=2; order=0 is equivalent to method="mean"
            
    Returns
    -------
    result : array of float (N,)
        1D array with averages with same size as input array. 
        Elements on either end may be NaN
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
