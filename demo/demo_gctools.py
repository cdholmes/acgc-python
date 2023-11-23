#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:00:41 2017

@author: cdholmes
"""

import acgc.gctools as gc
import acgc.nctools as nct
import matplotlib.pyplot as plt
import numpy as np

def pbl_thetav_infl( z, thetav ):

    # Locate the mixing layer depth, 
    # identified as the inflection point of the virtual potential temperature
    # and restricted to the lowest 4 km above the surface.  
    #
    # INPUTS:
    #   z       : altitude in km
    #   thetav  : virtual potential temperature, any units
    #
    # OUTPUT:
    #   zpbl    : altitude of the mixed layer top, km
    #   L       : index location of zpbl as integer + fraction
    #    
    #
    # This method seems to fail when the mixed layer top is within the first
    # model layer so that the 2nd derivative cannot be calculated in that layer
    
    import numpy as np
    
    if (max(z) > 100):
        print('pbl_thetav_infl: z must by in km')
        raise SystemExit()
    
    
    # 1st derivative of virtual potential temperature
    dthv_dz = ( thetav[1:] - thetav[0:-1] ) / ( z[1:] - z[0:-1] )
    # Midpoints where derivatives are
    zmid    = ( z[1:] + z[0:-1] ) / 2
    
    # 2nd derivative of virtual potential temperature
    d2thv_dz2 = ( dthv_dz[1:] - dthv_dz[0:-1] ) / ( zmid[1:] - zmid[0:-1] )
    # Pad with zeros
    d2thv_dz2 = np.concatenate( ([0], d2thv_dz2, [0]) )
    
    ############################
    ## Original method suggested by Allison Wing; sometimes picked out small 
    ## thetav variations within the mixed layer 
    ## This method is unable to identify mixing heights in the first model layer 
    ## because the second derivative is defined only in layer 2 and above.
    ## It also frequently identifies small thetav fluctuations within the boundary layer
    ## as the mixing height when they are not.pwa
    # # Find the level L with the maximum 2nd derivative of thetav, the inflection point
    # # This is the lowest level with a *local* maximum
    # L = 0
    # d2max = -999
    # while (d2thv_dz2[L+1] > d2max ):
    #    L=L+1
    #    d2max = d2thv_dz2[L]
    ############################

    ############################
    # Note that this new method sometimes misses modest low-level gradients
    # in favor of stronger high-altitude gradients. This results in high bias.
    #
    # Find the maximum 2nd derivative (inflection point) in the lowest 4 km
    idx = np.where(z-z[0] < 4 )[0]
    L = np.argmax( d2thv_dz2[idx] )
    
    # PBL height is location of max
    zpbl1 = z[L]
    
    # Use a 2nd order polynomial fit to smooth the profile and 
    # refine the inflection point location
    pf = np.polyfit( z[L-1:L+2], d2thv_dz2[L-1:L+2], 2 )

    # Find max from polynomial
    zpbl = -pf[1] / (2*pf[0])
    
    # Add a fraction to L indicating where in the interval the pbl height is
    if (zpbl >= zpbl1):
        L = L + (zpbl-zpbl1) / ( z[L+1] - z[L] )
    else:
        L = L + (zpbl-zpbl1) / ( z[L]   - z[L-1])
       
    return zpbl, L


def pbl_bulk_richardson( z, thetav, ws, crit=0.25):
    
    # Locate the mixing layer depth, 
    # using the Bulk Richardson number, Ri.
    # Ri represents the ratio of buoyant dissipation to shear production of 
    # turbulent kinetic energy. 
    # Negative Ri values imply unstable convective conditions (buoyant production of TKE).
    # Positive Ri values imply stable conditions (buoyant consumption of TKE; ie dtheta/dz >0)
    # Zero Ri implies zero buoyancy or neutral stability.
    # When Ri exceeds a critical value of about Ri_c = 0.25, the shear
    # production of TKE becomes small and the flow becomes laminar. 
    # 
    # This definition of the mixed layer is most appropriate where wind stress
    # is the predominant source of mixing; thus it is best for 
    # stable or neutral, not unstable, mixed layers.
    # Based on definition from Seibert et al., 2000 (Atmos Environ)
    # See also Ri definition in Garratt (1992, The Atmospheric Boundary Layer)
    #
    # INPUTS:
    #   z       : altitude in km
    #   thetav  : virtual potential temperature, K or C
    #   ws      : wind speed, m/s
    #
    # OUTPUT:
    #   zpbl    : altitude of the mixed layer top, km
    #   L       : index location of zpbl as integer + fraction
    #    
    #

    if (max(z) > 100):
        print('pbl_bulk_richardson: z must by in km')
        raise SystemExit()
    

    g0 = 9.81
    
    # Convert km -> m
    zm = z * 1e3
    
    # Bulk Richardson number
    Ri = g0 * zm * ( thetav - thetav[0]) / thetav[0] / ws**2
    
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
    
    
file = '/Users/cdholmes/MERRA2.20150101.I3.4x5.nc4'
f2   = '/Users/cdholmes/MERRA2.20150101.A3dyn.4x5.nc4'

# Surface pressure, 3D temperature, 3D specific humidity
PS2D = nct.get_nc_var( file, 'PS' ) # Pa
T3D  = nct.get_nc_var( file, 'T'  ) # K
Q3D  = nct.get_nc_var( file, 'QV' ) # kg/kg
lat  = nct.get_nc_var( file, 'lat')
lon  = nct.get_nc_var( file, 'lon')
WS3D = np.sqrt( nct.get_nc_var( f2,   'U'  )**2 + # m/s
                nct.get_nc_var( f2,   'V'  )**2 ) # m/s

# Coordinates of some location
i=15
j=43#; j=6
t=4


# Surface pressure at this location, hPa
PS = PS2D[t,j,i] / 100
# Temperature profile at this location
T  = T3D[t,:,j,i]
# Specific humidity profile at this location
Q = Q3D[t,:,j,i]
# Wind speed, note that winds are probably defined on grid edges while 
# thermodynamic properties are defined at centers
WS = WS3D[t,:,j,i]

# Number of layers in model
nlev = T.size

# Pressure at level centers, hPa
P = gc.get_lev_p( Psurf=PS, nlev=nlev, edge=False )

# Altitude at level centers, km
z = gc.get_lev_z(TK=T, Psurf=PS, Zsurf=0, Q=Q, edge=False) / 1e3

# Calculate the potential temperature, K
theta = T * (1000/P)**0.286

# Virtual potential temperature, neglecting liquid water, K
thetav = theta * (1 + 0.61 * Q )

zpbl, lpbl = pbl_thetav_infl( z, thetav )
lpbl = int(round(lpbl))

zpblR, lpblR = pbl_bulk_richardson( z, thetav, WS )
lpblR = int(round(lpblR))

plt.clf()
plt.plot(thetav-273,z, label='theta_v')
# Mixed layer depth from two definitions
plt.plot(thetav[lpbl]-273+[-5,5],np.array([zpbl,zpbl]),label='ML theta_v inflection')
plt.plot(thetav[lpblR]-273+[-5,5],np.array([zpblR,zpblR]),label='ML Richardson')
#plt.plot(Q,z, label='Humidity')
plt.xlabel('Temperature, K')
plt.ylabel('Altitude, km')
plt.title('lat={}, lon={}'.format(lat[j],lon[i]))
plt.ylim([0,5])
plt.xlim(thetav[0]-273+[-10,20])
plt.legend(fontsize=7)
plt.tight_layout()
