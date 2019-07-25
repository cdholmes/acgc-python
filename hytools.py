#!/usr/bin/env python3

# Package of functions to read HYSPLIT output 

import numpy  as np
import pandas as pd
import os

metroot = '/data/MetData/ARL/'

# Read trajectory file
def read_tdump(file):

    # Open the file
    with open(file,"r") as fid:

        # First line gives number of met fields
        nmet = int(fid.readline().split()[0])
        
        # Skip the met file lines
        for i in range(nmet):
            next(fid)

        # Number of trajectories
        ntraj = int(fid.readline().split()[0])

        # Skip the next lines
        for i in range(ntraj):
            next(fid)

        # Read the variables that are included
        line = fid.readline()
        nvar = int(line.split()[0])
        vnames = line.split()[1:]
        
        # Read the data
        df = pd.read_csv( fid, delim_whitespace=True, header=None, index_col=False,
                names=['tnum','metnum','year','month','day','hour','minute','fcasthr','thour','lat','lon','alt']+vnames )
        
        # Convert year to 4-digits
        df.loc[:,'year'] += 2000

        # Convert to time
        df['time'] = pd.to_datetime( df[['year','month','day','hour','minute']] )
        
        #print(df.iloc[-1,:])
        return df

def get_gdas1_filename( time ):

    # Directory for GDAS 1 degree
    dirname = metroot+'gdas1/'

    # Filename template
    filetmp = 'gdas1.{mon:s}{yy:s}.w{week:d}'

    # week number in the month
    wnum = ((time.day-1) // 7) + 1

    # GDAS 1 degree file
    filename = filetmp.format( mon=time.strftime("%b").lower(), yy=time.strftime("%y"), week=wnum )

    return dirname, filename

def get_gdas0p5_filename( time ):

    # Directory for GDAS 0.5 degree
    dirname  = metroot+'gdas0p5/'

    # Filename template
    filetmp = '{date:s}_gdas0p5'

    # GDAS 0.5 degree file
    filename = filetmp.format( date=time.strftime("%Y%m%d") )

    return dirname, filename

def get_met_filename( date, metversion='gdas0p5' ):

    # If metversion is a single string, then get value from appropriate function
    if isinstance( metversion, str ):

        if (metversion == 'gdas1'):
            dirname, filename = get_gdas1_filename( date )
        elif (metversion == 'gdas0p5'):
            dirname, filename = get_gdas0p5_filename( date )
        else:
            raise NotImplementedError( "get_met_filename: {metversion:s} unrecognized".format(metversion) )

    elif isinstance( metversion, list ):

        dirname  = None
        filename = None

        # Loop over all the metversion and use the first one with a file present
        for met in metversion:

            # Find filename for this met version
            d, f = get_met_filename( date, met )

            # If the file exists, use this and exit; otherwise keep looking
            if ( os.path.isfile( d+f ) ):
                dirname  = d
                filename = f
                break

        if (filename is None):
            raise FileNotFoundError( "get_met_filename: no files found for "+','.join(metversion) )

    else:
        raise TypeError( "get_met_filename: metversion must be a string or list" )

    return dirname, filename

def write_control( time, lat, lon, alt, trajhours, fname='CONTROL.000', metversion=['gdas0p5','gdas1'] ):

    # Write a control file for trajectory starting at designated time and coordinates

    # Number of days of met data
    ndays = int( np.ceil( np.abs( trajhours ) / 24 ) + 1 )

    if ( trajhours < 0 ):
        d0 = -ndays+1
        d1 = 1
    else:
        d0 = 0
        d1 = ndays

    # List of met directories and met files that will be used
    metdirs  = []
    metfiles = []
    for d in range(d0,d1):
        # date of met data
        metdate = time.date() + pd.Timedelta( d, "D" )

        dirname, filename = get_met_filename( metdate, metversion )

        # Add the file, if it isn't already in the list
        if ( filename not in metfiles ):
            metdirs.append(  dirname  )
            metfiles.append( filename )

    # Number of met file
    nmet = len( metfiles )

    # Runs will fail if the initial time is not bracketed met data.
    # When met resolution changes on the first day, this condition may not be met.
    # Starting the midnight trajectories at one 00:01 avoids this problem.
    if (time.hour==0 and time.minute == 0):
        time = time + pd.Timedelta( 1, "m" )

    # Start date-time, formatted as YY MM DD HH {mm}
    startdate = time.strftime( "%y %m %d %H %M" )

    f = open( fname, 'w' )

    f.write( startdate+'\n' )
    f.write( "1\n" )
    f.write( '{:<10.4f} {:<10.4f} {:<10.4f}\n'.format( lat, lon, alt ) )
    f.write( '{:d}\n'.format( trajhours ) )
    f.write( "0\n" )
    f.write( "10000.0\n" )
    f.write( '{:d}\n'.format( nmet ) )
    for i in range( nmet ):
        f.write( metdirs[i]+'\n' )
        f.write( metfiles[i]+'\n' )
    f.write( "./\n" )
    f.write( "tdump\n" )

    f.close()

