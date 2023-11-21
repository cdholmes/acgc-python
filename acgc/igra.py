#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:12:49 2018

Functions for reading and using IGRA v2 radiosonde data

@author: cdholmes
"""

import numpy as np
import pandas as pd
import datetime as dt

def read_igra_country(file):

    # Read file of country codes
    country = pd.read_fwf( file, header=None, 
            widths=[2,41],
            names=['countryID','countryName'] )

    return country

def read_igra_stations(file):

    # Read file of IGRA station coordinates and dates of observation
    stations = pd.read_fwf( file,
                            header=None,
                            na_values={'lat':-98.8888,'lon':-998.8888,'elev':[-999.9,-998.8]},
                            widths=[11,9,10,7,3,31,5,5,7], 
                            names=['siteID','lat','lon','elev',
                                   'state','name','firstyear','lastyear','nobs'] )

    # Country code is the first two characters of the siteID
    stations['country'] = stations.siteID.str[0:2]

    return stations

def read_igra_file( file, derived=False, readprofiles=True ):
    # Read sounding file
    # If derived=True, then read the IGRA-derived files
    # If derived=False, then read the IGRA sounding

                #    Derivd data contains the following variables
                #    pw         = precipitable water, mm
                #    invpress   = inversion pressure, hPa
                #    invhgt     = inversion height, m AGL
                #    invtempdif = temperature difference from surface to inversion, K
                #    mixpress   = pressure at top of mixed layer (parcel method), hPa
                #    mixhgt     = height of mixed layer, m AGL
                #    frzpress   = pressure at freezing level, hPa
                #    frzhgt     = height of mixing level, m AGL
                #    lclpress   = pressure at the LCL, hPa
                #    lclhgt     = height of the LCL, m AGL
                #    lfcpress   = pressure of the LFC, hPa
                #    lfchgt     = height of the LFC, m AGL
                #    lnbpress   = pressure of the LNB, hPa
                #    lnbhgt     = height of LNB, m AGL
                #    LI         = Lifted index, C
                #    SI         = Showalter index, C
                #    KI         = K index, C
                #    TTI        = Total totals index, C
                #    CAPE       = CAPE, J/kg
                #    CIN        = Convective inhibition, J/kg

    # Try to read the data from pickle; Read it from ascii file if pickle doesn't exist
    try:
        data = pd.read_pickle(file+'.pkl')
    except FileNotFoundError:
    
        # line number counter
        lnum = 0
        
        # profile number
        pnum = 0
        
        # Define empty data frame
        data = pd.DataFrame(columns=('siteID','time','syntime','profile'))
        
        first = True

        basewidth = [12,5,3,3,3,5,5]
        basenames = ['siteID','year','month','day','hour','reltime','numlev']

        # Open file for line-by-line reading
        with open( file, 'r' ) as f:
            for line in f:
        
                # Increment line counter
                lnum += 1
                                
                # Raise error if line doesn't begin with "#"
                if (line[0] != '#' ):
                    print('Unexpected IGRA file format. Header lines should begin with "#"')
                    print('line ',lnum,' in file ',file)
                    print(line)
                    raise SystemExit
                
                # Fields that are the same for sounding and derived files
                siteID     =      line[1:12]
                year       = int( line[13:17] )
                month      = int( line[18:20] )
                day        = int( line[21:23] )
                hour       = int( line[24:26] )
                reltime    = int( line[27:31] )
                numlev     = int( line[31:36] )

                # Extract hour and minute from release time
                relhour    = int( reltime / 100 )
                relmin     = np.mod( reltime, 100 )

                # Use the nominal time when release time is missing
                if (relhour==99):
                    relhour = hour
                if (relmin==99):
                    relmin = 0

                # Actual launch time
                time = dt.datetime( year, month, day, relhour, relmin )

                # Synoptic time (Typically 0Z or 12Z)
                syntime = dt.datetime( year, month, day, hour )

                # Read variables that differ between derived and standard files
                if (derived):

                    # Header items for derived files
                    pw         = float( line[37:43] ) / 100
                    invpress   = float( line[43:49] ) / 100
                    invhgt     = float( line[49:55] )       
                    invtempdif = float( line[55:61] ) / 10
                    mixpress   = float( line[61:67] ) / 100
                    mixhgt     = float( line[67:73] )
                    frzpress   = float( line[73:79] ) / 100
                    frzhgt     = float( line[79:85] )
                    lclpress   = float( line[85:91] ) / 100
                    lclhgt     = float( line[91:97] )
                    lfcpress   = float( line[97:103] ) / 100
                    lfchgt     = float( line[103:109] )
                    lnbpress   = float( line[109:115] ) / 100
                    lnbhgt     = float( line[115:121] )
                    LI         = float( line[121:127] )
                    SI         = float( line[127:133] )
                    KI         = float( line[133:139] )
                    TTI        = float( line[139:145] )
                    CAPE       = float( line[145:151] )
                    CIN        = float( line[151:157] )

                    # Profile metadata 
                    info = { 'siteID':    siteID,
                             'time':      time,
                             'syntime':   syntime,
                             'numlev':    numlev,
                             'pw':        pw,
                             'pInversion':invpress,
                             'zInversion':invhgt,
                             'dTinversion':invtempdif,
                             'pMix':      mixpress,
                             'zMix':      mixhgt,
                             'pFreeze':   frzpress,
                             'zFreeze':   frzhgt,
                             'pLCL':      lclpress,
                             'zLCL':      lclhgt,
                             'pLFC':      lfcpress,
                             'zLFC':      lfchgt,
                             'pLNB':      lnbpress,
                             'zLNB':      lnbhgt,
                             'LI':      LI,
                             'SI':      SI,
                             'KI':      KI,
                             'TTI':     TTI,
                             'CAPE':    CAPE,
                             'CIN':     CIN }

                else:
                    
                    p_src  =        line[37:45]
                    np_src =        line[46:54]
                    lat    = float( line[55:62] ) / 1e4
                    lon    = float( line[63:71] ) / 1e4

                    # Profile metadata 
                    info = { 'siteID':  siteID,
                             'time':    time,
                             'syntime': syntime,
                             'numlev':  numlev }

                # Replace missing data
                for key in info.keys():
                    if (info[key] in [-99999, -9999.9, -999.99]):
                        info[key] = np.nan
                
                # Print some info every 100 entries
                if (np.mod( pnum, 100 )==0):
                    print('{:4d}-{:02d}-{:02d} {:02d}:{:02d}'.format(
                        year, month, day, relhour, relmin ))

                # Read the vertical profile
                if (readprofiles and derived):
                    profile = pd.read_fwf( f, nrows=numlev,
                                           header=None,
                                           na_values=[-9999],
                                           widths=[7]+[8]*18,
                                           names=['p','zrep','z','T','Tgrad',
                                                  'Tpot','Tpotgrad','Tvirt','Tvirtpot',
                                                  'e','es',
                                                  'RHrep','RH','RHgrad',
                                                  'u','ugrad','v','vgrad','N'] )
                
                    # Convert Pa -> hPa
                    profile['p'] /= 100

                    # Convert K*10 -> K
                    profile[['T','Tgrad','Tpot','Tpotgrad','Tvirt','Tvirtpot']] /= 10 

                    # Convert vapor pressure, hPa*1000 -> hPa
                    profile[['e','es']] /= 1000

                    # Convert %*10 -> %
                    profile[['RH','RHrep']] /= 10

                    # Convert m/s*10 -> m/s
                    profile[['u','ugrad','v','vgrad']] /= 10

                    # Add profile to data
                    info.update({'profile': profile})
                    
                elif (readprofiles):

                    # Read the sounding
                    # Units: p, Pa; z, m; T, C*10; RH, %*10; dpdp, C*10 (dewpoint depression);
                    # wdir, degree; wspd, m/s*10
                    profile = pd.read_fwf(f, nrows=numlev,
                                          header=None,
                                          na_values=[-8888,-9999], 
                                          widths=[1,1,6,7,1,5,1,5,1,5,6,6,6],
                                          names=['levtype1','levtype2','etime',
                                                 'p','pflag','z','zflag','T','Tflag',
                                                 'RH','dpdp','wdir','wspd'] )
               
                    # Keep level types 1* (standard pressure), 2* (other pressure level)
                    # Drop level type 3* (non-pressure levels)
                    profile = profile[ profile.levtype1 != 3 ]
                
                    # Convert Pa -> hPa
                    profile['p'] /= 100
                
                    # Convert C*10 -> C
                    profile['T']     = profile['T'] / 10 
                    profile['dpdp'] /= 10
                
                    # Convert RH
                    profile['RH'] /= 10
                
                    # Convert m/s*10 -> m/s
                    profile['wspd'] /= 10
                
                    # Dewpoint, K
                    profile['Td'] = profile['T'] - profile['dpdp']

                    # Add profile to data
                    info.update({'profile': profile})
                    
                else:
                    # Don't read the profile
                    # Skip the lines containing the profile
                    for i in range(numlev):
                        next(f)
                    
                # Increment line counter
                lnum += numlev

                # Increment profile number
                pnum += 1
        
                # Create an empty dataframe on first pass
                if first:
                    data = pd.DataFrame(columns=info.keys())
                    first= False

                # Add this datapoint; Use nominal time for the index
                data.loc[syntime] = info
                    
        # Save data as pickle file
        data.to_pickle(file+'.pkl')

    return data        

def demo():

    # Read some data and plot it
    
    import matplotlib.pyplot as plt
    import datetime as dt

    # Read the Barrow data
    data = read_igra_file( 'data/Barrow_2000.txt' )

    # Find the souding closest to 2000-06-01 12:00 UTC
    idx = np.argmin( np.abs( data.index - dt.datetime(2000,6,1,0) ) )

    profile = data.iloc[idx].profile

    plt.clf()
    plt.plot( profile['T'], profile['z'], label='T' )
    plt.plot( profile['Td'], profile['z'], label='Td' )
    plt.title( data.index[idx] )
    plt.xlabel( 'Temperature, K' )
    plt.ylabel( 'Altitude, m' )
    plt.ylim((0,4000))
    plt.legend()
