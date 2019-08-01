#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 
Read ICARTT (ffi1001) format files

Created on Mon Aug 13 16:59:25 2018
@author: C.D. Holmes
'''

import numpy  as np 
import pandas as pd
import os

def read_icartt( files, usePickle=False ):

    # Files input must be string or list of strings
    if isinstance( files, str ):
        # Convert to list
        files = [files]
    elif isinstance( files, list ):
        # Do nothing
        pass
    else:
        raise TypeError( 'read_icartt: files must be a filename or list of filenames' )

    obsall = []
    for n,file in enumerate(files):
    
        # Read from Pickle file, if it exists and is requested
        pklfile = file+'.pkl'
        if (usePickle and os.path.isfile(pklfile)):
            obs = pd.read_pickle(pklfile)
        
        else:

            # Ensure file exists
            if (not os.path.isfile( file )):
                raise FileNotFoundError( file+" doesn't exist" )

            # Read the ICARTT file
            with open( file, 'r' ) as f:
            
                # Read the number of header lines
                nheader, fmt = f.readline().split(',')
                nheader = int(nheader)

                # Ensure this is a 
                if (int(fmt) != 1001 ):
                    raise Exception( 'read_icartt: '+file+' is not an ICARTT (ffi1001) file' )
                    
                # Skip 5 lines
                for junk in range(5):
                    next(f)
                
                # Read date
                year, month, day = map( int, f.readline().split(',')[0:3] )
            
                # Skip line
                next(f)

                # Read name of the time variable
                tname, tunit = [s.strip() for s in f.readline().split(',')[0:2]]

                # Raise exception if the time unit is not seconds; may need to be handled differently below
                if (not tunit in ['s','seconds','seconds (from midnight UTC)']):
                    print(tunit)
                    raise Exception( 'read_icartt: time expected in seconds (s); unit in file: ',tunit )

            # Read data
            obs = pd.read_csv(file,skiprows=nheader-1,na_values=' -99999')
            
            # Catch missing data that are not reported as integers
            #obs[obs==-99999] = np.nan
            
            # Strip whitespace from column names
            obs.columns = obs.columns.str.strip()
            
            # Rename selected variables
            obs.rename(columns={tname:'time'}, inplace=True)
        
            # Convert time to a datetime
            #obs.time = dt.datetime(year,month,day) + pd.TimedeltaIndex(obs.time,'s')
            obs.time = pd.DatetimeIndex( pd.Timestamp(year=year,month=month,day=day) + pd.TimedeltaIndex(obs.time,tunit) )
        
            # Add flight number
            obs['file'] = n+1

            # Save pickle
            if (usePickle):         
                obs.to_pickle(pklfile)

        # Add to list 
        obsall.append(obs)
    
    # Concatenate all files into one dataframe
    obs = pd.concat( obsall )
    
    return obs
