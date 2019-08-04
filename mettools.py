#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:41:04 2017

@author: kgraham
"""

### Program to standardize PBLH extraction and input to csv for all O-buoy files


import numpy as np
import netCDF4 as ncdf
import pandas as pd
import glob
import datetime as dt
from scipy.interpolate import interp1d
import gctools as gct
import RHtoQ
import wind2m
from numba import jit


@jit
def pbl_bulk_richardson( z, thetav, ws, crit=0.25):
# Bulk Richardson definition of PBL height
#
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
    while (Ri[L] < crit ):
        L+=1
    zpbl1 = z[L]

    # Use a 2nd order polynomial fit to smooth the profile and
    # refine the height of the critical value
    pf = np.polyfit( z[L-1:L+2], Ri[L-1:L+2]-crit, 2 )

    # Root from quadratic formula (positive root because Ri is increasing in z)
    zpbl = (-pf[1] + np.sqrt( pf[1]**2 - 4*pf[0]*pf[2] ) ) / (2 * pf[0])

    # Add a fraction to L indicating where in the interval the pbl height is
    if (zpbl >= zpbl1):
        L = L + (zpbl-zpbl1) / ( z[L+1] - z[L] )
    else:
        L = L + (zpbl-zpbl1) / ( z[L]   - z[L-1])

    return zpbl, L

@jit
def pbl_parcel(z, thetav, delta=0.5 ):
# Parcel method definition of PBL height

    # mixing height is the highest level where thetav < thetav(0) + delta
    L=0
    while (thetav[L+1] < ( thetav[0] + delta ) ):
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
        if (zpbl >= zpbl1):
            L = L + (zpbl-zpbl1) / ( z[L+1] - z[L] )
        else:
            L = L + (zpbl-zpbl1) / ( z[L]   - z[L-1])

    return zpbl, L

def logwind(surfaceP, q2m, temp2m, ustar, shf, lhf, z0, z):
    # Bring in SHF, LHF, ustar, z0 from A1 files

    p = surfaceP*100    # calculations require pascals
    #t = temp+273.15    # calculations require kelvin, except for Lv
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
    theta_e = theta+(1.0 + e2m/p * 0.61 )
    #OLD METHOD: theta_e = theta/(1.0-e2m/p*(1.0-0.622))

    # Obukhov length, m
    # numerator and denominator, for safe division in case den=0
    num = -(ustar**3 * rho * theta_e)
    den =  (k*g*(shf*(1+0.61*q2m)/Cp+0.61*theta*(lhf/Lv)))
    if (np.abs(den) > 0):
        L = num/den
    else:
        L = np.inf

    z = z           #height that you are calculating
    d = 0           #displacement height assumed zero over ice
    z0 = z0         #roughness length
    zeta = (z - d)/L
    zeta0 = z0/L
    eta0 = (1-(15*zeta0))**(1/4)
    etaR = (1-(15*zeta))**(1/4)


    if L == 0 or np.absolute(L) > 10**5:
        # Neutral
        logwind = (ustar/k) * np.log( (z-d) / z0 )
    elif L < 10**5 and L > 0:
        # Stable
        logwind = (ustar/k) * np.log( (z-d) / z0 ) + 4.7 * (zeta - zeta0)
    elif L > -10**5 and L < 0:
        # Unstable
        logwind = (ustar/k) * np.log( (z-d) / z0 ) + \
             np.log( ( (eta0**2+1) * (eta0+1)**2) /
                     ( (etaR**2+1) * (etaR+1)**2) ) + \
             2*(np.arctan(etaR) - np.arctan(eta0))
    else:
        logwind = np.NaN

    return logwind
