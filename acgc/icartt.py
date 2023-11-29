#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Read and write ICARTT (ffi1001) format files

Created on Mon Aug 13 16:59:25 2018
@author: C.D. Holmes
'''

import os
import pandas as pd


def read_icartt( files, usePickle=False, timeIndex=False ):
    '''Read ICARTT file or files into a pandas DataFrame
    
    Parameters
    ----------
    files : list or str
        path to ICARTT file or files that will be read. 
        Data within these files will be concatenated, so files should all contain the same variables
    usePickle : bool (default=False)
        if usePickle=True, the data will be written to a pkl file with ".pkl" appended to path
        On subsequent read_icartt calls, data will be read from the .pkl file, if it exists
    timeIndex : bool (default=False)
        sets DataFrame index to the time variable from the ICARTT file, rather than a row counter
        
    Returns
    -------
    obs : pandas.DataFrame
        ICARTT file contents. In addition to column names for the ICARTT variables, 
        the DataFrame columns also include 'time' in pandas.DatetimeIndex format and 
        'file' that is the ordinal number of the input file that each row was read from
    '''

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
            if not os.path.isfile( file ):
                raise FileNotFoundError( file+" doesn't exist" )

            # Read the ICARTT file
            with open( file, 'r', encoding='ascii' ) as f:

                # Read the number of header lines
                nheader, fmt = f.readline().split(',')
                nheader = int(nheader)

                # Ensure this is a 
                if int(fmt) != 1001:
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

                # Raise exception if the time unit is not seconds;
                # may need to be handled differently below
                if (tunit in ['s','seconds','seconds (from midnight UTC)','seconds_past_midnight']):
                    # Use unit expected by pandas
                    tunit = 's'
                else:
                    print(tunit)
                    raise Exception( 'read_icartt: time expected in seconds (s); '+
                                     'unit in file: ',tunit )

                # Number of dependent variables
                nvar = int( f.readline() )

                # Scale factor for dependent variables
                scale = [ float(s) for s in f.readline().split(',') ]

                # Missing value flag for dependent variables
                naflag = [ s.strip() for s in f.readline().split(',') ]

                # Dependent variable names from the next nvar lines
                varnames = [ f.readline().split(',')[0] for v in range(nvar) ]

                # Missing flags for each variable as a dict
                nadict = { varnames[i]: naflag[i] for i in range(nvar) }

            # Read data
            obs = pd.read_csv(file, skiprows=nheader-1,
                              na_values=nadict, skipinitialspace=True)

            # Catch missing data that are not reported as integers
            #obs[obs==-99999] = np.nan

            # Strip whitespace from column names
            #obs.columns = obs.columns.str.strip()

            # Apply scale factors
            for i, s in enumerate(scale):
                if s != 1:
                    obs[varnames[i]] *= s

            # Add a time variable in datetime format
            obs['time'] = pd.DatetimeIndex( pd.Timestamp(year=year,month=month,day=day) +
                                         pd.TimedeltaIndex(obs[tname],tunit) )

            # Add flight number
            obs['file'] = n+1

            # Use time variable for index
            if timeIndex:
                obs.index = obs.time

            # Save pickle
            if usePickle:
                obs.to_pickle(pklfile)

        # Add to list
        obsall.append(obs)

    # Concatenate all files into one dataframe
    obs = pd.concat( obsall, sort=True )

    return obs

def write_icartt(filename, df, **kwargs):
    '''Write an ICARTT ffi1001 file

    Arguments
    ---------
    filename : str
        File to be created
    df : Pandas dataframe
        df should contain data and metadata
        df attributes must include all of the "required" and "normal_comments" names.
        The INDEPENDENT_VARIABLE_DEFINITION and DEPENDENT_VARIABLE_DEFINITION should be dicts containing
        {'VariableName':'units, standard name, description'}
        Only variables listed in INDEPENDENT_VARIABLE_DEFINITION and DEPENDENT_VARIABLE_DEFINITION
        will be written to the output file.
    **kwargs
        passed to pandas.to_csv
    '''

    normal_comments = ['PI_CONTACT_INFO',
                'PLATFORM',
                'LOCATION',
                'ASSOCIATED_DATA',
                'INSTRUMENT_INFO',
                'DATA_INFO',
                'UNCERTAINTY',
                'ULOD_FLAG',
                'ULOD_VALUE',
                'LLOD_FLAG',
                'LLOD_VALUE',
                'DM_CONTACT_INFO',
                'PROJECT_INFO',
                'STIPULATIONS_ON_USE',
                'OTHER_COMMENTS',
                'REVISION',
                'REVISION_COMMENTS']

    required = ['PI_NAME',
                'ORGANIZATION_NAME',
                'SOURCE_DESCRIPTION',
                'MISSION_NAME',
                'VOLUME_INFO',
                'DATE_LINE',
                'TIME_INTERVAL',
                'INDEPENDENT_VARIABLE_DEFINITION',
                'NUMBER_DEPENDENT_VARIABLES',
                'DEPENDENT_SCALE_LINE',
                'DEPENDENT_MISSING_FLAGS',
                'DEPENDENT_VARIABLE_DEFINITION',
                'SPECIAL_COMMENTS',
                'NORMAL_COMMENTS']

    # Variables that will be written to file
    ictvars = list(df.INDEPENDENT_VARIABLE_DEFINITION.keys()) + \
              list(df.DEPENDENT_VARIABLE_DEFINITION.keys())

    # Form the header
    header = []

    for k in required:
        # Special handling for some 
        if k=='VOLUME_INFO':
            header.append( '1, 1')
        elif k=='DATE_LINE':
            header.append( df.time.iloc[0].strftime('%Y, %m, %d, ') + 
                           pd.Timestamp.today().strftime('%Y, %m, %d'))
        elif k=='TIME_INTERVAL':
            header.append( '1'  ) # ***UPDATE LATER: TIME INTERVAL IN SECONDS
        elif k=='INDEPENDENT_VARIABLE_DEFINITION':
            keydict = getattr( df, k )
            header.append( list(keydict.keys())[0] + ', ' + list(keydict.values())[0] )            
        elif k=='DEPENDENT_VARIABLE_DEFINITION':
            keydict = getattr( df, k )
            nvars = len(keydict)
            header.append( str(nvars) )
            header.append( ','.join(['1']*nvars) ) # Scale line
            header.append( ','.join(['-9999']*nvars) ) # Missing flags
            for kn in keydict.keys():
                header.append( kn + ', ' + keydict[kn])
        elif k in ['TIME_INTERVAL',
                'NUMBER_DEPENDENT_VARIABLES', 
                'DEPENDENT_SCALE_LINE',
                'DEPENDENT_MISSING_FLAGS']:
            #*** Handled elsewhere
            pass
        elif k=='SPECIAL_COMMENTS':
            v = getattr( df, k )
            if v:
                # Expect a string or array of several lines
                header.append( str(len(list(v))) )
                header.extend( list(v) )
            else:
                header.append( '0' )
        elif k=='NORMAL_COMMENTS':
            # Form the comment block
            nc= []
            for kn in normal_comments:
                v = getattr( df, kn )
                if kn=='REVISION_COMMENTS':
                    # Expect a string or array of several lines
                    nc.extend( list(v) )
                else:    
                    nc.append( '{:s}: {:s}'.format(kn,v) )
            nc.append( ', '.join(ictvars)) # Also add list of variable names

            # Add normal comments to the header
            header.append( str(len(nc)) )
            header.extend( nc )
        else:
            v = getattr( df, k )
            header.append( v )

    # Write the file
    with open(filename,'w',encoding='ascii') as f:
        f.write('{:d}, 1001\n'.format(len(header)+1))  # +1 accounts for this line
        for line in header:
            f.write(line+'\n')
        df[ictvars].to_csv( f,
                            index=False,
                            header=False,
                            na_rep='-9999',
                            **kwargs )
