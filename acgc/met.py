#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for computing boundary layer meteorological properties

"""

import numpy as np

def pbl_bulk_richardson( z, thetav, ws, crit=0.25):
    '''Compute PBL mixing height from critical value of bulk Richardson number
    
    Arguments
    ---------
    z : array, float
        altitude, m
    thetav  : array, float
        virtual potential temperature, K
    ws : array, float
        wind speed, m/s
    crit : float
        critical value defining the top of the PBL mixing (default=0.25)

    Returns
    -------
    zpbl : float
        altitude of PBL top, m
    L : float
        level number within z array corresponding to altitude zpbl
    '''

    #    if (max(z) > 100):
    #        print('pbl_bulk_richardson: z must by in km')
    #        raise SystemExit()

    g0 = 9.81

    # Convert km -> m
    #zm = z * 1e3
    # Input is already in meters
    zm = z

    # Bulk Richardson number
    Ri = g0 * zm * ( thetav - thetav[0] ) / thetav[0] / ws**2

    # mixing height is the lowest level where Ri > 0.25
    L=0
    while Ri[L] < crit:
        L+=1
    zpbl1 = z[L]

    # Use a 2nd order polynomial fit to smooth the profile and
    # refine the height of the critical value
    pf = np.polyfit( z[L-1:L+2], Ri[L-1:L+2]-crit, 2 )

    # Root from quadratic formula (positive root because Ri is increasing in z)
    zpbl = (-pf[1] + np.sqrt( pf[1]**2 - 4*pf[0]*pf[2] ) ) / (2 * pf[0])

    # Add a fraction to L indicating where in the interval the pbl height is
    if zpbl >= zpbl1:
        L = L + (zpbl-zpbl1) / ( z[L+1] - z[L] )
    else:
        L = L + (zpbl-zpbl1) / ( z[L]   - z[L-1])

    return zpbl, L

def pbl_parcel(z, thetav, delta=0.5 ):
    '''Compute PBL mixing height with parcel method

    The mixing height is defined as the altitude where a rising parcel of air
    from the base of the profile becomes neutrally buoyant. More formally, where
    the virtual potential temperature exceeds the value at the base plus delta.
    
    Arguments
    ---------
    z : array, float
        altitude, m
    thetav  : array, float
        virtual potential temperature, K or C
    delta : float
        incremental threshold of virtual potential temperature, K or C

    Returns
    -------
    zpbl : float
        altitude of PBL top, m
    L : float
        level number within z array corresponding to altitude zpbl
    '''

    # mixing height is the highest level where thetav < thetav(0) + delta
    L=0
    while thetav[L+1] < ( thetav[0] + delta ):
        L += 1
    zpbl1 = z[L]

    if L == 0:
        # Linear interpolation for lowest layer
        zpbl = np.interp(thetav[0]+delta, thetav[0:2], z[0:2])
        #zpbl = ( zpbl[0] + zpbl[1] ) / 2

    else:
        # Use a 2nd order polynomial fit to smooth the profile and
        # refine the height of the critical value
        pf = np.polyfit( z[L-1:L+2], thetav[L-1:L+2]-(thetav[0]+delta), 2 )

        # Root from quadratic formula (positive root because theta_v is increasing in z)
        zpbl = (-pf[1] + np.sqrt( pf[1]**2 - 4*pf[0]*pf[2] ) ) / (2 * pf[0])

        # Add a fraction to L indicating where in the interval the pbl height is
        if zpbl >= zpbl1:
            L = L + (zpbl-zpbl1) / ( z[L+1] - z[L] )
        else:
            L = L + (zpbl-zpbl1) / ( z[L]   - z[L-1])

    return zpbl, L

def mo_wind(surfaceP, q2m, temp2m, ustar, shf, lhf, z0, d, z):
    '''Predict wind speed at altitude z using Monin-Obukhov similarity theory

    Arguments
    ---------
    surfaceP : float
        surface pressure, hPa
    q2m : float
        specific humidity at 2m, kg/kg
    temp2m : float
        temperature at 2m, K
    ustar : float
        friction velocity, m/s
    shf, lhf : float
        sensible and latent heat fluxes, W/m2
    z0 : float
        roughness length, m
    d : float
        displacement height, m
    z : float
        height at which wind speed will be calculated, m
        
    Returns
    -------
    ws : float
        wind speed at altitude z, m/s
    '''
    # Bring in SHF, LHF, ustar, z0 from A1 files

    p = surfaceP*100    # calculations require pascals
    mmair = .02897      # kg/mol
    mmwater = 0.01801   # kg/mol
    Rgas = 8.3145       # Ideal gas constant (J/K*mol)
    g = 9.81            # gravity
    k = 0.4             # von Karman's constant
    Cp = 1004           # Specific Heat capacity of air (J/kg*K)
    rdcp = 0.286        # Rd/Cp constant

    Lv = 2.501e06 - 2370*(temp2m-273.15)        # units: J/kg
    rho = (p*mmair)/(Rgas*temp2m)               # units: kg/m^3
    theta = temp2m*(1000.0/(p/100.0))**rdcp     # units are Kelvin

    # Vapor pressure, Pa
    e2m = q2m / (1-q2m) * (mmair/mmwater) * surfaceP
    #OLD FORMULA: (P/10 appears to be an error)
    #e2m = q2m*(mmair/mmwater)*(surfaceP/10)

    # Virtual potential temperature, K
    theta_e = theta*(1.0 + e2m/p * 0.61 )
    #OLD METHOD: theta_e = theta/(1.0-e2m/p*(1.0-0.622))

    # Obukhov length, m
    # numerator and denominator, for safe division in case den=0
    num = -(ustar**3 * rho * theta_e)
    den =  (k*g*(shf*(1+0.61*q2m)/Cp+0.61*theta*(lhf/Lv)))
    if np.abs(den) > 0:
        L = num/den
    else:
        L = np.inf

    # Parameters within similarity functions
    zeta = (z - d)/L
    zeta0 = z0/L
    eta0 = (1-(15*zeta0))**(1/4)
    etaR = (1-(15*zeta))**(1/4)


    if L == 0 or np.absolute(L) > 10**5:
        # Neutral
        ws = (ustar/k) * np.log( (z-d) / z0 )
    elif L < 10**5 and L > 0:
        # Stable
        ws = (ustar/k) * np.log( (z-d) / z0 ) + 4.7 * (zeta - zeta0)
    elif L > -10**5 and L < 0:
        # Unstable
        ws = (ustar/k) * np.log( (z-d) / z0 ) + \
             np.log( ( (eta0**2+1) * (eta0+1)**2) /
                     ( (etaR**2+1) * (etaR+1)**2) ) + \
             2*(np.arctan(etaR) - np.arctan(eta0))
    else:
        ws = np.NaN

    return ws
