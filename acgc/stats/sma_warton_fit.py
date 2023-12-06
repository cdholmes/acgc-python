# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:13:26 2016

@author: cdholmes
"""
import numpy as np
import scipy.linalg as la 
import scipy.stats as stats
    
#from numba import jit

__all__ = ['sma_warton_fit']

def sma_warton_fit(X0,Y0,alpha=0.05,intercept=True,robust=False,wartonrobust=False):

    '''
    Calculate standard major axis, aka reduced major axis, fit to 
    data X and Y The main advantage of this is that the best fit
    of Y to X will be the same as the best fit of X to Y.
    More details in Warton et al. Biology Review 2006 
    
    Parameters
    ----------
    X, Y : array_like
        Input values, Must have same length.
    alpha : float
        Desired confidence level for output.
        
    Returns
    -------
    Gradient : float
        Gradient or Slope of Y vs. X
    Intercept : floats
        Y intercept.
    stdgrad : float
        Standard error of gradient estimate
    stdint : float
        standard error of intercept estimate
    ci_grad : [float, float]
        confidence interval for gradient at confidence level alpha
    ci_int : [float, float]
        confidence interval for intercept at confidence level alpha'''

    # Make sure arrays have the same length
    assert ( len(X0) == len(Y0) ), 'Arrays X and Y must have the same length'

    # Make sure alpha is within the range 0-1
    assert (alpha < 1), 'alpha must be less than 1'
    assert (alpha > 0), 'alpha must be greater than 0'    
    
    
    # Drop any NaN elements of X or Y    
    # Infinite values are allowed but will make the result undefined
    idx = ~np.isnan(X0 * Y0 )
    X = X0[idx]
    Y = Y0[idx]
    
    # Number of finite elements
    N = len(X)
    
    # Degrees of freedom for model
    if (intercept):
        dfmod = 2
    else:
        dfmod = 1

    # Robust factors, to be changed later
    rfac1 = 1
    rfac2 = 1
    
    if (intercept):
        # Average values
        Xmean = np.mean(X)
        Ymean = np.mean(Y)
    else:
        Xmean = 0
        Ymean = 0            
         
    # Sample variance of X, Y
    Vx  = np.sum( (X - Xmean)**2 ) / (N-1)
    Vy  = np.sum( (Y - Ymean)**2 ) / (N-1)
    
    # Sample covariance of X, Y
    Vxy = np.sum( (X - Xmean) * (Y - Ymean) ) / (N-1)

    if (robust):

        from sklearn.covariance import MinCovDet

        rcov = MinCovDet().fit( np.array([X,Y]).T )
        
        Xmean = rcov.location_[0]
        Ymean = rcov.location_[1]
        Vx    = rcov.covariance_[0,0]
        Vy    = rcov.covariance_[1,1]
        Vxy   = rcov.covariance_[0,1]

        # Number of observations used in covariance estimate
        N = rcov.support_.sum()
        
    if (wartonrobust):
        
        c=np.sqrt(3)
        
        # Use alternate methods to calculate robust means and covariance
        q = stats.chi2.cdf(c,2)
        
        # Huber M to get robust mean and covariance
        rm, rcov = huber_cov( np.array([X,Y]), c=c )
        
        Xmean = rm[0]
        Ymean = rm[1]
        Vx    = rcov[0,0]
        Vy    = rcov[1,1]
        Vxy   = rcov[0,1]
                
        # Robust factors scale the degrees of freedom
        rfac1, rfac2 = robust_factor( np.array([X,Y]), rm, rcov, q )
                
    
    # Correlation Coefficient
    R = Vxy / np.sqrt( Vx * Vy )

    Sx = np.sqrt( Vx )
    Sy = np.sqrt( Vy )
    
    # Slope
    slope = np.sign(R) * Sy / Sx

    # Standard error of slope estimate
    ste_slope = np.sqrt( rfac2 / (N-dfmod) * Sy**2 / Sx**2 * (1-R**2) )

    # Confidence interval for slope
    B = (1-R**2) / (N-dfmod) * rfac1 * stats.f.isf(alpha,1,(N-dfmod))
    ci_slope = slope * ( np.sqrt( B+1 ) + np.sqrt(B) * np.array([-1,+1]) )

    # If slope is negative, flip the order for first element is most negative
    if (slope < 0):
        ci_slope = np.flipud( ci_slope )
 
    if (intercept):
        # Intercept    
        Intercept = Ymean - slope * Xmean
        
        # Residuals
        resid = Y - (Intercept + slope * X )    

        # Sample standard deviation of the residuals
        Sr = np.std( resid, ddof=dfmod )    

        # Another method, may be faster, but less obvious
        # This method is better for robust fitting, because the simple 
        Sr = np.sqrt((Vy - 2 * slope * Vxy + slope**2 *  Vx ) * (N-1) / (N-dfmod) )
        #print(Sr,Sr2)

        # Standard error of the intercept estimate
        ste_int = np.sqrt( Sr**2/N * rfac2 + Xmean**2 * ste_slope**2  )

        # Confidence interval for Intercept
        tcrit = stats.t.isf(alpha/2,N-dfmod)
        ci_int = Intercept + ste_int * np.array([-tcrit,tcrit])
        
    else:
        # Intercept is zero by definition
        Intercept = 0
        ste_int   = 0
        ci_int    = np.array([0,0])

   
        
    return slope, Intercept, ste_slope, ste_int, ci_slope, ci_int

#@jit
def robust_factor( X, rm, rcov, c=0.777 ):

    rfac1=1
    rfac2=1

    N = X.shape[1] # number of pointsn
    k = X.shape[0] # number of variables

    # inverse square root of covariance matrix
    U, S, V   = la.svd( rcov )
    rinvsq = U @ np.diag(np.sqrt(1/S)) @ V

    # Means in matrix form
    Xm = np.tile(rm,[N,1]).T

    # Centered data
    Xc = X-Xm
    
    # z-score
    z = la.norm( rinvsq @ Xc, axis=0 )
    
    q  = stats.chi2.cdf(3,k)
    ###CHECK THE C VALUE
    rfac1 = np.mean( alpha_fun(z, k, q )**2 ) / 8
    rfac2 = np.mean( gamma_fun(z, k, q )**2 ) / 2
    
    return rfac1, rfac2

def alpha_fun( r, k, q ):
    
    c = stats.chi2.ppf(q,k)
    sig = stats.chi2.cdf(c,k+2) + (c/k) * (1-q)
    c = np.sqrt( c )
    
    #c2 = c**2
    #q  = stats.chi2.cdf(c2,k)
    #s2 = stats.chi2.cdf(c2,k+2) + (c2/k) * (1-q) 
        
    eta = r**2    / (2 * sig**2)
    eta[r>c] = c**2 / (4 * sig**2)
    eta = np.mean( eta )
    
    alpha = r**2    / (eta * sig**2)
    alpha[r>c] = c**2 / (eta * sig**2)
    
    return alpha

def gamma_fun( r, k, q ):
    
    c = np.sqrt( stats.chi2.ppf(q,k) )
    #sig = stats.chi2.cdf(c,k+2) + (c/k) * (1-q)
    c = np.sqrt( c )
    
    eta = c * (k-1) / ( r*k )
    eta[r<=c] = 1
    eta = np.mean(eta)
    
    gamma = r / eta
    gamma[r>c] = c/eta    
    
    return gamma

#@jit
def huber_cov( X, c=1.73):
    ''' Mean and covariance of x and y, using Huber's M estimator
     Method is taken from Taskinen and Warton, 2013 '''
    
    N = X.shape[1] # number of points
    k = X.shape[0] # number of variables
    
    # first guess is normal covariance
    rcov = np.cov( X )
    
    # first guess is normal means
    rm = X.mean(axis=1)

    # Alternate values
    #c=3
    #c=1.345
    c2 = c**2
    q  = stats.chi2.cdf(c2,k)
    s2 = stats.chi2.cdf(c2,k+2) + (c2/k) * (1-q) 

    # These parameters are used by huber.M, but seem suspicious
    c=3
    q  = stats.chi2.cdf(c,k)
    c2=stats.chi2.cdf(c,k+2) + (c/k) * (1-q)
    
    R = la.cholesky( la.inv( rcov ) )
    
    #q = 0.777, c = 3
    
    it = 0
    d1 = 1
    d2 = 1
    eps = 1e-6 
    maxit = 100
    while ((it<maxit) and (d1>eps) and (d2>eps) ):    

        
        # Means in matrix form
        Xm = np.tile(rm,[N,1]).T

        # Centered X
        Xc = X - Xm
        
        # z scores
        s = np.diag( Xc.T @ (R.T @ R ) @ Xc )
        
        # Weighting
        u       = (c/c2) / s
        u[s<=c] = 1/c2
        
        # updated weighted inverse covariance
        C = R @ (Xc @ np.diag(u) @ Xc.T) @ R.T / N
        
        R0 = la.cholesky( la.inv( C ) )
        
        R = R0 @ R

        d1 = np.max( np.abs( R0 - np.identity(k)).sum(axis=1) )
    
        # updated z scores
        s = np.diag( Xc.T @ (R.T @ R) @ Xc )
        
        # Weighting
        v       = np.sqrt(c / s )
        v[s<=c] = 1
    
        # Update increment for means
        h = ( Xc @ np.diag(v) ).mean(axis=1) / v.mean()
    
        # Updated robust mean
        rm = rm + h
        
        d2 = np.sqrt( (h-rm).T @ (h-rm) )
    
        it += 1
        
    if (it>=maxit):
        raise SystemExit('huber_cov did not converge')
    
    # Robust covariance
    rcov = la.inv( R.T @ R )
    
    return rm, rcov

def huber_cov_old( X, c=1.73):
    ''' Mean and covariance of x and y, using Huber's M estimator
     Method is taken from Taskinen and Warton, 2013 '''
    
    N = X.shape[1] # number of points
    k = X.shape[0] # number of variables
    
    # first guess is normal covariance
    rcov = np.cov( X )
    
    # first guess is normal means
    rm = X.mean(axis=1)

    # Alternate values
    #c=3
    #c=1.345
    c2 = c**2
    q  = stats.chi2.cdf(c2,k)
    s2 = stats.chi2.cdf(c2,k+2) + (c2/k) * (1-q) 
    
    #q = 0.777, c = 3
    
    it = 0
    d1 = 1
    d2 = 1
    eps = 1e-6 
    while ((it<100) and (d1>eps) and (d2>eps) ):    

        # inverse square root of covariance matrix
        U, S, V   = la.svd( rcov )
        rinvsq = U @ np.diag(np.sqrt(1/S)) @ V
        #rinvsq = la.sqrtm( la.inv( rcov ) ) 
    
        # z-scores
        z = la.norm( rinvsq @ ( X - np.tile(rm,[N,1]).T ) , axis=0 )

        # Weights for mean
        w1 = np.minimum( 1, c/z )
        #w1 = np.ones(N)

        # Weights for covariance, add scaling later
        w2 = w1**2 / s2
        
        # New estimate of robust mean
        rm2 = np.sum( X * np.tile(w1,[2,1]), axis=1 ) / np.sum(w1)
    
        # Means in matrix form
        Xm = np.tile(rm2,[N,1]).T

        # New estimate of robust covariance
        rcov2 = 1/(N-1) * ( (X-Xm) @ ( (X-Xm) * np.tile(w2,[2,1]) ).T )
    
        it += 1
        d1 = np.max(np.abs(rm2-rm))
        d2 = np.max(np.abs(rcov2-rcov))
        rcov = rcov2
        rm   = rm2
    
    if (it>100):
        raise SystemExit('huber_cov did not converge')
    
    return rm, rcov
