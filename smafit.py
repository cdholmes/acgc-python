# -*- coding: utf-8 -*-
"""Standard Major Axis (SMA) line fitting

Created on Fri May 20 19:13:26 2016

@author: cdholmes
"""

def smafit(X0,Y0,cl=0.95,intercept=True,robust=False,rmethod='FastMCD'):
    """Standard Major-Axis (SMA) line fitting
    
    Calculate standard major axis, aka reduced major axis, fit to 
    data X and Y. The main advantage of this over ordinary least squares is 
    that the best fit of Y to X will be the same as the best fit of X to Y.
    
    The fit equations and confidence intervals are implemented following 
    Warton et al. (2006). Robust fits use the FastMCD covariance estimate 
    from Rousseeuw and Van Driessen (1999). While there are many alternative 
    robust covariance estimators (e.g. other papers by D.I. Warton using M-estimators), 
    the FastMCD algorithm is default in Matlab. 
    
    References 
    Warton, D. I., Wright, I. J., Falster, D. S. and Westoby, M.: 
        Bivariate line-fitting methods for allometry, Biol. Rev., 81(02), 259, 
        doi:10.1017/S1464793106007007, 2006.
    Rousseeuw, P. J. and Van Driessen, K.: A Fast Algorithm for the Minimum 
        Covariance Determinant Estimator, Technometrics, 41(3), 1999.

    Parameters
    ----------
    X, Y : array_like
        Input values, Must have same length.
    cl   : float (default = 0.95)
        Desired confidence level for output. 
    intercept : boolean (default=True)
        Specify if the fitted model should include a non-zero intercept.
        The model will be forced through the origin (0,0) if intercept=False.
    robust : boolean (default=False)
        Use statistical methods that are robust to the presence of outliers
    rmethod: string (default='FastMCD')
        Method for calculating robust variance and covariance. Options:
        'MCD' or 'FastMCD' for Fast MCD
        'Huber' for Huber's T: reduce, not eliminate, influence of outliers
        'Biweight' for Tukey's Biweight: reduces then eliminates influence of outliers
        
    Returns
    -------
    Slope     : float
        Slope or Gradient of Y vs. X
    Intercept : float
        Y intercept.
    ste_grad : float
        Standard error of gradient estimate
    ste_int : float
        standard error of intercept estimate
    ci_grad : [float, float]
        confidence interval for gradient at confidence level cl
    ci_int : [float, float]
        confidence interval for intercept at confidence level cl
    """

    import numpy as np
    import scipy.stats as stats
    from sklearn.covariance import MinCovDet
    import statsmodels.formula.api as smf
    import statsmodels.robust.norms as norms
        
    # Make sure arrays have the same length
    assert ( len(X0) == len(Y0) ), 'Arrays X and Y must have the same length'

    # Make sure cl is within the range 0-1
    assert (cl < 1), 'cl must be less than 1'
    assert (cl > 0), 'cl must be greater than 0'    
    
    
    # Drop any NaN elements of X or Y    
    # Infinite values are allowed but will make the result undefined
    idx = ~np.logical_or( np.isnan(X0), np.isnan(Y0) ) 
    
    X = X0[idx]
    Y = Y0[idx]
    
    # Number of observations
    N = len(X)
    
    # Degrees of freedom for the model
    if (intercept):
        dfmod = 2
    else:
        dfmod = 1
    
    # Choose whether to use methods robust to outliers
    if (robust):
        
        # Choose the robust method
        if ((rmethod.lower() =='mcd') or (rmethod.lower() == 'fastmcd') ):
            # FAST MCD    
        
            if (not intercept):
                # intercept=False could possibly be supported by calculating
                # using mcd.support_ as weights in an explicit variance/covariance calculation
                raise NotImplemented('FastMCD method only supports SMA with intercept')
            
            # Fit robust model of mean and covariance
            mcd = MinCovDet().fit( np.array([X,Y]).T )
        
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

        elif ((rmethod.lower() =='biweight') or (rmethod.lower() == 'huber') ):

            # Tukey's Biweight and Huber's T
            if ( rmethod.lower() =='biweight'):
                norm = norms.TukeyBiweight()
            else:
                norm = norms.HuberT()
        
            # Get weights for downweighting outliers
            # Fitting a linear model the easiest way to get these             
            # Options include "TukeyBiweight" (totally removes large deviates) 
            # "HuberT" (linear, not squared weighting of large deviates)
            rweights = smf.rlm('y~x+1',{'x':X,'y':Y},M=norm).fit().weights

            # Sum of weight and weights squared, for convienience
            rsum  = np.sum( rweights ) 
            rsum2 = np.sum( rweights**2 ) 
        
            # Mean
            Xmean = np.sum( X * rweights ) / rsum
            Ymean = np.sum( Y * rweights ) / rsum
        
            # Force intercept through zero, if requested
            if (not intercept):
                Xmean = 0
                Ymean = 0
        
            # Variance & Covariance
            Vx    = np.sum( (X-Xmean)**2 * rweights**2 ) / rsum2
            Vy    = np.sum( (Y-Ymean)**2 * rweights**2 ) / rsum2
            Vxy   = np.sum( (X-Xmean) * (Y-Ymean) * rweights**2 ) / rsum2   

            # Effective number of observations
            N = rsum  

        else:

            raise NotImplemented("smafit.py hasn't implemented rmethod={:%s}".format(rmethod))
    else:
    
        if (intercept):
            # Average values
            Xmean = np.mean(X)
            Ymean = np.mean(Y)
        
            # Covariance matrix
            cov = np.cov( X, Y, ddof=1 )
  
            # Variance
            Vx = cov[0,0]
            Vy = cov[1,1]
        
            # Covariance
            Vxy = cov[0,1]

        else:
            
            # Force the line to pass through origin by setting means to zero
            Xmean = 0
            Ymean = 0
            
            # Sum of squares in place of variance and covariance
            Vx = np.sum( X**2 ) / (N-1)
            Vy = np.sum( Y**2 ) / (N-1)
            Vxy= np.sum( X*Y  ) / (N-1)
        
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
    B = (1-R**2)/(N-dfmod) * stats.f.isf(1-cl,1,N-dfmod)
    ci_grad = Slope * ( np.sqrt( B+1 ) + np.sqrt(B)*np.array([-1,+1]) )

    #############
    # INTERCEPT

    if (intercept):
        Intercept = Ymean - Slope * Xmean
       
        # Standard deviation of residuals
        # New Method: Formula from smatr R package (Warton)
        # This formula avoids large residuals of outliers when using robust=True
        Sr = np.sqrt((Vy - 2 * Slope * Vxy + Slope**2 *  Vx ) * (N-1) / (N-dfmod) )

        # OLD METHOD
        # Standard deviation of residuals
        #resid = Y - (Intercept + Slope * X )    
        # Population standard deviation of the residuals
        #Sr = np.std( resid, ddof=0 )      
    
        # Standard error of the intercept estimate
        ste_int = np.sqrt( Sr**2/N + Xmean**2 * ste_slope**2  )
        
        # Confidence interval for Intercept
        tcrit = stats.t.isf((1-cl)/2,N-dfmod)
        ci_int = Intercept + ste_int * np.array([-tcrit,tcrit])
    
    else:     
        
        # Set Intercept quantities to zero
        Intercept = 0
        ste_int   = 0
        ci_int    = np.array([0,0])
        
    
    return Slope, Intercept, ste_slope, ste_int, ci_grad, ci_int
