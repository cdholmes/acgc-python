# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:38:50 2015

@author: cdholmes
"""
# Call loess function from R
# requires that rpy2 and R are installed
# rpy2 for Anaconda can be obtained via 'conda install -c https://conda.binstar.org/r rpy2'

def loess(x,y,span=0.5,degree=2,family="symmetric"):
    # Call the loess function from R
    
    import numpy as np
    import os
    os.environ['R_HOME'] = '/Users/cdholmes/anaconda/bin'
    import rpy2.robjects as robjects
    
    def fillnan(values, idxisnan):
        # Add NaNs to the array "values" at the same locations given in "idxisnan"
        # Output will be an array with the same shape as idxisnan, containing "values"

        # Set up output array with the same number of elements as idxisnan and same type as values
        out = np.empty(idxisnan.shape, dtype=values[0].dtype)
        out[:] = np.nan
        # Fill the values that are not NaN
        out[~idxisnan] = values
        
        return out

    # R functions
    rloess = robjects.r['loess']
    rpredict = robjects.r['predict']

    # R versions of arrays
    rx = robjects.FloatVector(x)
    ry = robjects.FloatVector(y)
    df = robjects.DataFrame({"x":rx, "y":ry})
    
    # Do the fit and create fit standard errors
    fit = rloess('y~x',data=df,span=span,degree=degree,family=family)
    pred = rpredict(fit,se=True)

    # Convert the fit to numpy arrays
    # fitted value
    yfit = fillnan( np.array(fit.rx2('fitted')), np.isnan(y) )
    # residuals
    yresid = fillnan( np.array(fit.rx2('residuals')), np.isnan(y) )
    # standard error of fit
    yste = fillnan( np.array(pred.rx2('se.fit')), np.isnan(y) )    
    
    # standard deviation of points around fit (i.e. pointwise prediction interval)
    # calculate from a loess fit to resid**2, then sqrt
    df2 = robjects.DataFrame({"x":rx, "y":robjects.FloatVector(yresid**2)})
    fit2 = rloess('y~x',data=df2,span=span,degree=degree,family=family)
    # var must be >= 0
    yvar = np.fmax(0,np.array(fit2.rx2('fitted')))
    ystd = fillnan( np.sqrt(yvar), np.isnan(y) )
    
    return yfit, ystd, yste