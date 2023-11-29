#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import datetime as dt
import numpy as np
import netCDF4 as ncdf
import pandas as pd
from scipy.interpolate import interp1d
from numba import jit



#========================================
# Read MERRA2 coordinates

# MERRA2 directory
m2_dir = '/data/MetData/GEOS_2x2.5/MERRA2/'

# Find first MERRA2 file since coordinates are the same in all
fa3 = glob.glob(m2_dir+'*/*/*.nc4', recursive=True )[0]

# Open file for reading
ncfile = ncdf.Dataset(fa3,'r')

# Get the latitude, longitude
mlon = ncfile.variables['lon'][:]
mlat = ncfile.variables['lat'][:]

# Functions that give the index location of an lon or lat
# e.g. latidx(-90) returns 0
lonidx = interp1d(np.append(mlon,mlon[0]+360), np.arange(mlon.size+1))
latidx = interp1d(mlat, range(mlat.size))

# Close the file
ncfile.close()

@jit
def ncinterp2dt( ncfile, varname, lat, lon, time ):
    # Define a function that interpolates varname from file ncfile to the specified lat, lon, time
    # Use bilinear interpolation for spatial interpolation
    # Use linear interpolation for time

    # Convert pandas Timestamp -> datetime
    if ( type(time) == pd.Timestamp ):
        time = time.to_pydatetime()

    # locate indices of points surrounding the Buoy
    # lonidx and latidx must be defined before this function is defined
    ifrac = lonidx(lon)
    jfrac = latidx(lat)

#    try:
#            ifrac = lonidx(lon)
#    except ValueError:
        # value between last element and first element of mlon
 #       ifrac = (-1) + ( ( lon - mlon[-1] ) / ( mlon[0] + 360 - mlon[-1] ) )
        #print(ifrac)

    # index to lower left of point
    i0 = np.int(np.floor(ifrac))
    j0 = np.int(np.floor(jfrac))

    # index to upper right of point
    i1 = i0 + 1
    j1 = j0 + 1
    if (i1 >= mlon.size): # loop around dateline
        i1 = 0

    # linear weight for lower left point
    i0wt = 1 - (ifrac - i0)
    j0wt = 1 - (jfrac - j0)

    # Time variable from file
    nctime = ncdf.MFTime(ncfile.variables['time'])

    # locate index of data before and after observation time
    t0 = ncdf.date2index( time, nctime, select='before' )
    t1 = ncdf.date2index( time, nctime, select='after'  )

    if (t0 == t1):
        # Equal weights for exact time match
        t0wt = 0.5
    else:
       # Convert file times to datetime
       dt0= ncdf.num2date( nctime[t0], nctime.units )
       dt1= ncdf.num2date( nctime[t1], nctime.units )

       # linear weight for first time
       t0wt = 1 - (time-dt0) / (dt1-dt0)

    # Number of dimenions in variable (space and time)
    numDim = len(ncfile.variables[varname].shape)

    if (numDim == 3):

        # surface variable at point
        val =(ncfile.variables[varname][t0,j0,i0] * i0wt     * j0wt     * t0wt     +
              ncfile.variables[varname][t0,j0,i1] * (1-i0wt) * j0wt     * t0wt     +
              ncfile.variables[varname][t0,j1,i0] * i0wt     * (1-j0wt) * t0wt     +
              ncfile.variables[varname][t0,j1,i1] * (1-i0wt) * (1-j0wt) * t0wt     +
              ncfile.variables[varname][t1,j0,i0] * i0wt     * j0wt     * (1-t0wt) +
              ncfile.variables[varname][t1,j0,i1] * (1-i0wt) * j0wt     * (1-t0wt) +
              ncfile.variables[varname][t1,j1,i0] * i0wt     * (1-j0wt) * (1-t0wt) +
              ncfile.variables[varname][t1,j1,i1] * (1-i0wt) * (1-j0wt) * (1-t0wt) )

    elif (numDim == 4):

        # vertical profile
        val =(ncfile.variables[varname][t0,:,j0,i0] * i0wt     * j0wt     * t0wt     +
              ncfile.variables[varname][t0,:,j0,i1] * (1-i0wt) * j0wt     * t0wt     +
              ncfile.variables[varname][t0,:,j1,i0] * i0wt     * (1-j0wt) * t0wt     +
              ncfile.variables[varname][t0,:,j1,i1] * (1-i0wt) * (1-j0wt) * t0wt     +
              ncfile.variables[varname][t1,:,j0,i0] * i0wt     * j0wt     * (1-t0wt) +
              ncfile.variables[varname][t1,:,j0,i1] * (1-i0wt) * j0wt     * (1-t0wt) +
              ncfile.variables[varname][t1,:,j1,i0] * i0wt     * (1-j0wt) * (1-t0wt) +
              ncfile.variables[varname][t1,:,j1,i1] * (1-i0wt) * (1-j0wt) * (1-t0wt) )
    else:

        print('***Unexpected number of dimensions in variable '+varname)
        return -1

    return val

