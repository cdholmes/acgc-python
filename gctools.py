#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:14:12 2017

@author: cdholmes
"""

import numpy as np
from numba import jit
from scipy.interpolate import interp1d 

@jit
def mapweight1d( edge1, edge2, edgelump=True ):
    '''Calculate mapping weights from source grid with edge1 to destination grid with edge2
    Returns weights, which can then be applied 
    new = np.dot(W,old)
    '''

    reverse = False

    # Check if the input pedge1 are inceasing
    if np.all(np.diff(edge1) > 0):
        # All are increasing, do nothing
        pass
    elif np.all(np.diff(edge1) < 0):
        raise ValueError( 'decreasing edge1 values not implemented')
        # All are decreasing, reverse
        edge1 = np.flip( edge1 )
    else:
        raise ValueError( 'Input edge1 must be sorted ')

    # Check if the output edges are increasing 
    if np.all(np.diff(edge2) > 0):
        # All are increasing, do nothing
        pass
    elif np.all(np.diff(edge2) < 0):
        raise ValueError( 'decreasing edge2 values not implemented')
        # All are decreasing, reverse
        edge2 = np.flip( edge2 )
        reverse = True
    else:
        raise ValueError( 'Input edge2 must be sorted ')

    # Interpolate pedge2 into pedge1
    idx21 = np.interp( edge2, edge1, np.arange(len(edge1)) )

    # If the pedge1 max or min are outside pedge2, then make sure the 
    # edges go into the end levels
    if edgelump:
        idx21[0]  = 0
        idx21[-1] = len(edge1)-1  

    # Empty array for result
    weight = np.zeros( (len(edge2)-1, len(edge1)-1) )
    
    for iNew in range(len(edge2)-1):
        for iOld in range( np.floor(idx21[iNew]).astype(int), np.ceil(idx21[iNew+1]).astype(int) ):

            # Concise version
            weight[iNew,iOld] = ( np.minimum(idx21[iNew+1],iOld+1) - np.maximum(idx21[iNew],iOld) )
   
    return weight

def regrid2d( array1, xe1, ye1, xe2, ye2 ):
    ''' Regrid regular x-y or lat-lon array

    inputs:
    array1 : 2D array input to regrid
    xe1    : x dimension edges of input
    ye1    : y dimension edges of input
    xe2    : x dimension edges for output
    ye2    : y dimension edges for output
    '''

    # Sizes
    insize = np.shape(array1)
    nxe1 = np.size(xe1)
    nye1 = np.size(ye1)
    nxe2 = np.size(xe2)
    nye2 = np.size(ye2)

    # Ensure arrays conform
    if (len(insize) != 2 ):
        raise TypeError( 'array1 must be 2D')
    if (insize[1] != nxe1-1):
        raise ValueError( 'xe1 must match size of array1' )
    if (insize[0] != nye1-1):
        raise ValueError( 'ye1 must match size of array1' )

    # Regridding weight matrices
    weightY = mapweight1d(ye1, ye2)
    weightX = mapweight1d(xe1, xe2)

    # Size of temporary and output arrays
    tmpsize = [nye2-1,nxe1-1]
    outsize = [nye2-1,nxe2-1]

    # Temporary array
    a1b = np.empty( tmpsize )
    a1b[:] = np.nan

    # Output array
    outarray = np.empty(outsize)
    outarray[:] = np.nan

    # Regrid latitude
    for i in range(np.size(xe1)-1):
        a1b[:,i] = np.dot( weightY, array1[:,i].flatten() )

    # Regrid longitude
    for i in range(outsize[0]):
        outarray[i,:] = np.dot( weightX, a1b[i,:].flatten() )

    return outarray

@jit
def regrid_plevels( array1, pedge1, pedge2, intensive=False, edgelump=True ):
    ''' Mass-conserving regridding of data on pressure levels
    Data from the input array will be remapped to the output pressure levels while conserving total mass.

    Inputs:
    array1: Array of data that will be remapped. Quantities should be extensive (e.g. mass, not mixing ratio) 
        unless the intensive keyword is used.
    pedge1: Pressure edges of the input array1. The length of pedge1 should be one greater than array1. 
        The element array1[i] is bounded by edges pedge1[i], pedge1[i+1]
    pedge2: Pressure edges of the desired output array.
    intensive: (boolean) If true, then array1 will be treated as an intensive quantity (e.g mole fraction, 
        mixing ratio) during regridding. Each element array1[i] will be multiplied by (pedge1[i]-pedge[i+1]), 
        which is proportional to airmass if pedge1 is pressure, before regridding. The output array2, will 
        similarly be dividied by (pedge2[i]-pedge[i+1]) so that the output is also intensive. 
        By default, intensive=False is used and the input array is assumed to be extensive.
    edgelump: (boolean) If the max or min of pedge1 extend beyond the max or min of pedge2, the mass will 
        be placed in in the first or last grid level of the output array2, ensuring that the mass of array2 
        is the same as array1. 
        By default, edgelump=True. If edgelump=False, then the mass of array2 can be less than array1.

    Outputs:
    array2: Array of remapped data. The length of array2 will be one less than pedge2.
    '''

    if (len(pedge1) != len(array1)+1):
        raise ValueError("pedge1 must have size len(array1)+1")

    reverse = False

    # Check if the input pedge1 are inceasing
    if np.all(np.diff(pedge1) > 0):
        # All are increasing, do nothing
        pass
    elif np.all(np.diff(pedge1) < 0):
        # All are decreasing, reverse
        array1 = np.flip( array1 )
        pedge1 = np.flip( pedge1 )
    else:
        raise ValueError( 'Input pedge1 must be sorted ')

    # Check if the output edges are increasing 
    if np.all(np.diff(pedge2) > 0):
        # All are increasing, do nothing
        pass
    elif np.all(np.diff(pedge2) < 0):
        # All are decreasing, reverse
        pedge2 = np.flip( pedge2 )
        reverse = True
    else:
        raise ValueError( 'Input pedge2 must be sorted ')

    # If input is an intensive quantity, then multiply by the pressure grid spacing
    if intensive:
        array1 = array1 * ( pedge1[1:] - pedge1[:-1] )

    # Interpolate pedge2 into pedge1
    idx21 = np.interp( pedge2, pedge1, np.arange(len(pedge1)) )

    # If the pedge1 max or min are outside pedge2, then make sure the 
    # edges go into the end levels
    if edgelump:
        idx21[0]  = 0
        idx21[-1] = len(pedge1)-1  

    # Empty array for result
    array2 = np.zeros( len(pedge2)-1 )

    for i in range(len(array2)):
        for j in range( np.floor(idx21[i]).astype(int), np.ceil(idx21[i+1]).astype(int) ):

            # Concise version
            array2[i] += array1[j] * ( np.minimum(idx21[i+1],j+1) - np.maximum(idx21[i],j) )

            # Lengthy version
            #if ( j >= idx21[i] and j+1 <= idx21[i+1] ):
            #    array2[i] += array1[j]
            #    #print( 'input level {:d} inside output level {:d}'.format(j,i))
            #elif ( j < idx21[i] and j+1 > idx21[i+1] ):
            #    array2[i] += array1[j] * (idx21[i+1] - idx21[i])
            #    #print( 'output level inside input level')
            #elif ( j < idx21[i] ):
            #    array2[i] +=  array1[j] * (1 - np.mod(idx21[i],  1)) 
            #    #print( 'input level {:d} at bottom of output level {:d}'.format(j,i))
            #elif ( j+1 > idx21[i+1] ):
            #    array2[i] += array1[j] * np.mod( idx21[i+1], 1)
            #    #print( 'input level {:d} at top of output level {:d}'.format(j,i))
            #else:
            #    raise NotImplemented

    # Convert back to an intensive quantity, if necessary
    if intensive:
        array2 = array2 / ( pedge2[1:] - pedge2[:-1] )

    # Reverse the output array, if necessary
    if reverse:
        array2 = np.flip( array2 )

    return array2

def set_center_edge( center, edge, name ):

    # Resolve any conflict between center and edge requests
    # Values must be None or bool

    if ((center is None) and (edge is None) ):

        # If both center and edge are None, default to center
        center = True
        edge   = False

    elif (center is None):
        
        # Use edge, if that value is boolean
        if (type(edge) is bool):
            center = not edge
        else:
            raise TypeError( name + ': edge must be boolean or None')

    elif (edge is None):
        
        # Use center, if that value is boolean
        if (type(center) is bool):
            edge = not center
        else:
            raise TypeError( name + ': center must be boolean or None')

    elif((type(center) is bool) and (type(edge) is bool)):

        # If both are True or both are False, then raise error
        if ( ((center==True)  and (edge==True)) or 
             ((center==False) and (edge==False)) ):
            # If both center and edge are specified, then raise error
            raise ValueError( name + ': Choose either edge=True or center=True' )

        # At this point, either center=True or edge=True, but not both 
    
    else:

        # Values aren't boolean
        raise TypeError( name + ': center and edge must be boolean or None')

    return center, edge

def lon2i(lon,res=4):
    '''Find I grid index for longitude
    '''
    # Put longitude in expected GC the range [-180,180]
    alon = np.mod( np.asarray(lon) + 180, 360 ) - 180
    longrid = get_lon(res=res,center=True)
    nlon = len(longrid)
    f = interp1d(np.append(longrid,longrid[0]+360),
                np.append(np.arange(nlon),0),
                kind='nearest',
                fill_value='extrapolate')
    return f(alon).astype(np.int)

def lat2j(lat,res=4):
    '''Find J grid index for latitude
    '''
    alat = np.asarray(lat)
    latgrid = get_lat(res=res,center=True)
    nlat = len(latgrid)
    if (np.any(alat<-90) or np.any(alat>90)):
        raise ValueError('lat must be in the range [-90,90]')    
    f = interp1d(latgrid,
                np.arange(nlat),
                kind='nearest',
                fill_value='extrapolate')
    return f(alat).astype(np.int)

def ll2ij(lon,lat,res=4):
    '''Find (I,J) grid index for longitude, latitude
    '''
    alon = np.asarray(lon)
    alat = np.asarray(lat)
    nlon = alon.size
    nlat = alat.size
    if (nlon==1 and nlat>1):
        alon = np.ones_like(alat) * alon
    elif (nlat==1 and nlon>1):
        alat = np.ones_like(alon) * alat
    elif (nlon != nlat):
        raise ValueError('lon and lat must have the same length or be conformable')
    return lon2i(alon,res=res), lat2j(alat,res=res)

def get_lon(res=4, center=None, edge=None):
    '''Return the longitude of grid centers or edges. 
    GMAO grids assumed
    res = 0.5 for 0.5x0.625
    res = 2 for 2x2.5
    res = 4 for 4x5
    '''

    # Resolve any conflict between center and edge
    center, edge = set_center_edge( center, edge, 'get_lon' )

    # Longitude edges
    if (res==2):
        dx=2.5
    elif (res==4):
        dx=5
    elif(res==0.5):
        dx=0.625
    else:
        raise NotImplementedError( 'Resolution '+str(res)+' is not defined' )

    # Longitude edges
    lonedge = np.arange(-180 - dx/2, 180, dx)

    # Get grid centers, if needed
    if (center):
        lon = ( lonedge[0:-1] + lonedge[1:] ) / 2
    else:
        lon = lonedge
    
    return lon
    
def get_lat(res=4, center=None, edge=None):
    # Return the latitude of grid centers or edges. 
    # GMAO grids assumed
    # res = 2 for 2x2.5
    # res = 4 for 4x5
    
    # Resolve any conflict between center and edge
    center, edge = set_center_edge( center, edge, 'get_lat' )

    # Latitude edges
    if (res==2):
        dy=2
    elif (res==4):
        dy=4
    elif (res==0.5):
        dy=0.5
    else:
        raise NotImplementedError( 'Resolution '+str(res)+' is not defined' )

    # Latitude edges
    latedge = np.hstack([-90, np.arange(-90+dy/2,90,dy), 90] )

    # Get grid centers, if needed
    if (center):
        lat = ( latedge[0:-1] + latedge[1:] ) / 2
    else:
        lat = latedge
    
    return lat

def get_lev_p(Psurf=1013.25, nlev=47, edge=False ):
    
    # Get A,B values for vertical coordinate
    A, B = get_hybrid_ab(nlev=nlev, edge=edge )
    
    # Pressure, hPa
    P = A + B * Psurf
    
    return P

@jit
def get_lev_z(TK=[273], Psurf=1013.25, Pedge=[0], Q=0, Zsurf=0, edge=False ):
    
    # Calculate the heights of level edges or centers,
    # Given mean Temperature and Specific humidity (Tmean, Qmean)
    # of the layer between those pressures
    # Assume hydrostatic balance using hypsometric equation
    
    # Results will be geopotential height, m

    funname='gctools: get_lev_z: '

    # Acceleration due to gravity, m/s2    
    g0 = 9.81
    
    if (len(TK)<=1):
        print(funname,'TK must be an array')
        return -1
    
    # Number of levels in the temperature profile
    nlev = TK.size
    
    # Calculate the pressure edges if they aren't provided in input
    if (len(Pedge)!=nlev+1):
        # Get pressure at level edges, hPa
        Pedge = get_lev_p(Psurf=Psurf, nlev=nlev, edge=True )
    
    # Initialize array
    Zedge = np.zeros_like(Pedge)
    Zedge[0]=Zsurf
    
    # Virtual temperature, K
    Tv = TK * ( 1 + 0.61 * Q )
        
    # Calculate altitudes with hypsometric equation, m
    for L in range(1, Pedge.size):
        Zedge[L] = Zedge[L-1] + 287 * Tv[L-1] / g0 * np.log( Pedge[L-1] / Pedge[L] )
    
    # Choose grid edge or center altitudes, m
    if (edge==True):
        Z = Zedge
    else:
        Z = (Zedge[0:-1] + Zedge[1:]) / 2
        
    return Z
        
def get_hybrid_ab(nlev=47, edge=False, center=False):
    
    # A and B values for the GEOS-5/MERRA/MERRA2 
    # hybrid sigma-pressure vertical coordinate
    # The pressure at level edge L is P(L) = A(L) + B(L) * PS,
    # where PS is surface pressure
    
    # Resolve any conflict between center and edge requests
    if ( (center==False) and (edge==False) ):
        # If neither center nor edge is specified, default to center
        center=True
    elif ( (center==True) and (edge==True)):
        # If both center and edge are specified, then raise error
        print('get_hybrid_ab: Choose either edge=True or center=True')
        raise SystemExit()
    else:
        # At this point, either center=True or edge=True, but not both
        center = not edge
        
    if (nlev == 47):

         # A [hPa] for 47 levels (48 edges)
         A = np.array([ 0.000000E+00, 4.804826E-02, 6.593752E+00, 1.313480E+01, 
               1.961311E+01, 2.609201E+01, 3.257081E+01, 3.898201E+01, 
               4.533901E+01, 5.169611E+01, 5.805321E+01, 6.436264E+01, 
               7.062198E+01, 7.883422E+01, 8.909992E+01, 9.936521E+01, 
               1.091817E+02, 1.189586E+02, 1.286959E+02, 1.429100E+02, 
               1.562600E+02, 1.696090E+02, 1.816190E+02, 1.930970E+02, 
               2.032590E+02, 2.121500E+02, 2.187760E+02, 2.238980E+02, 
               2.243630E+02, 2.168650E+02, 2.011920E+02, 1.769300E+02, 
               1.503930E+02, 1.278370E+02, 1.086630E+02, 9.236572E+01, 
               7.851231E+01, 5.638791E+01, 4.017541E+01, 2.836781E+01, 
               1.979160E+01, 9.292942E+00, 4.076571E+00, 1.650790E+00, 
               6.167791E-01, 2.113490E-01, 6.600001E-02, 1.000000E-02 ] )

         # Bp [unitless] for 47 levels (48 edges)
         B = np.array([ 1.000000E+00, 9.849520E-01, 9.634060E-01, 9.418650E-01, 
               9.203870E-01, 8.989080E-01, 8.774290E-01, 8.560180E-01, 
               8.346609E-01, 8.133039E-01, 7.919469E-01, 7.706375E-01, 
               7.493782E-01, 7.211660E-01, 6.858999E-01, 6.506349E-01, 
               6.158184E-01, 5.810415E-01, 5.463042E-01, 4.945902E-01, 
               4.437402E-01, 3.928911E-01, 3.433811E-01, 2.944031E-01, 
               2.467411E-01, 2.003501E-01, 1.562241E-01, 1.136021E-01, 
               6.372006E-02, 2.801004E-02, 6.960025E-03, 8.175413E-09, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00 ] )
        
    elif (nlev == 72):

         # A [hPa] for 72 levels (73 edges)
         A = np.array([ 0.000000E+00, 4.804826E-02, 6.593752E+00, 1.313480E+01, 
               1.961311E+01, 2.609201E+01, 3.257081E+01, 3.898201E+01, 
               4.533901E+01, 5.169611E+01, 5.805321E+01, 6.436264E+01, 
               7.062198E+01, 7.883422E+01, 8.909992E+01, 9.936521E+01, 
               1.091817E+02, 1.189586E+02, 1.286959E+02, 1.429100E+02, 
               1.562600E+02, 1.696090E+02, 1.816190E+02, 1.930970E+02, 
               2.032590E+02, 2.121500E+02, 2.187760E+02, 2.238980E+02, 
               2.243630E+02, 2.168650E+02, 2.011920E+02, 1.769300E+02, 
               1.503930E+02, 1.278370E+02, 1.086630E+02, 9.236572E+01, 
               7.851231E+01, 6.660341E+01, 5.638791E+01, 4.764391E+01, 
               4.017541E+01, 3.381001E+01, 2.836781E+01, 2.373041E+01, 
               1.979160E+01, 1.645710E+01, 1.364340E+01, 1.127690E+01, 
               9.292942E+00, 7.619842E+00, 6.216801E+00, 5.046801E+00, 
               4.076571E+00, 3.276431E+00, 2.620211E+00, 2.084970E+00, 
               1.650790E+00, 1.300510E+00, 1.019440E+00, 7.951341E-01, 
               6.167791E-01, 4.758061E-01, 3.650411E-01, 2.785261E-01, 
               2.113490E-01, 1.594950E-01, 1.197030E-01, 8.934502E-02, 
               6.600001E-02, 4.758501E-02, 3.270000E-02, 2.000000E-02, 
               1.000000E-02 ] )

         # B [unitless] for 72 levels (73 edges)
         B = np.array([ 1.000000E+00, 9.849520E-01, 9.634060E-01, 9.418650E-01, 
               9.203870E-01, 8.989080E-01, 8.774290E-01, 8.560180E-01, 
               8.346609E-01, 8.133039E-01, 7.919469E-01, 7.706375E-01, 
               7.493782E-01, 7.211660E-01, 6.858999E-01, 6.506349E-01, 
               6.158184E-01, 5.810415E-01, 5.463042E-01, 4.945902E-01, 
               4.437402E-01, 3.928911E-01, 3.433811E-01, 2.944031E-01, 
               2.467411E-01, 2.003501E-01, 1.562241E-01, 1.136021E-01, 
               6.372006E-02, 2.801004E-02, 6.960025E-03, 8.175413E-09, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
               0.000000E+00 ])
        
    else:
        
        print('get_hybrid_ab: A, B not defined for nlev=',nlev)
        A = -1
        B = -1
        raise SystemExit()
       
    # Calculate values at level center    
    if (center == True):

        A = (A[0:-1] + A[1:]) / 2
        B = (B[0:-1] + B[1:]) / 2
        
    
    return A, B
    
    
    