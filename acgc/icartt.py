#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Read and write ICARTT (ffi1001) format files
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
    usePickle : bool, default=False
        if usePickle=True, the data will be written to a pkl file with ".pkl" appended to path
        On subsequent read_icartt calls, data will be read from the .pkl file, if it exists
    timeIndex : bool, default=False
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

def _get(obj,name,default_value=None):
    '''Get value from either attribute or key `name` 
    
    Parameters
    ----------
    obj : dict or object
    name : str
        name of an attribute or key
    default_value :
        return value if the attribute or key do not exist
    
    Returns
    -------
    value :
        value of attribute or dict key
    '''
    try:
        # Try access as attribute
        value = getattr(obj,name)
    except AttributeError:
        try:
            # Try access as dict key
            value = obj[name]
        except KeyError:
            # Not found so return default value
            value = default_value
    return value

def write_icartt(filename, df, metadata, **kwargs):
    '''Write an ICARTT ffi1001 file

    The contents of a pandas DataFrame (``df``) are written to a text file in ICARTT format
    using `metadata` to specify which variables are written and provide ICARTT file header.

    ICARTT file format specification document:
    https://www.earthdata.nasa.gov/esdis/esco/standards-and-practices/icartt-file-format

    Parameters
    ----------
    filename : str
        File to be created
    df : pandas.DataFrame
        Data values that will be written
    metadata : dict or obj
        See notes below for the attributes or keys that `metadata` must contain
    **kwargs
        passed to pandas.to_csv

    Notes
    -----
    `metadata` can be a dict or any object, so long as it contains the following attributes or keys:
    - independent_variable_definition (dict)
        should have only one key
    - dependent_variable_definition (dict)
        Controls which variables from `df` are written to file
    - measurement_start_date (pandas.Timestamp or datetime.datetime)
        date UTC when measurement collection began
    - pi_name (str)
    - pi_contact_info (str)
    - organization_name (str)
    - dm_contact_info (str)
    - mission_name (str)
    - project_info (str)
    - special_comments (list of str)
    - platform (str)
    - location (str)
    - associated_data (str)
    - intrument_info (str)
    - data_info (str)
    - uncertainty (str)
    - ulod_flag (str)
        commonly '-7777'
    - ulod_value (str)
    - llod_flag (str)
        commonly '-8888'
    - llod_value (str)
    - stipulations_on_use (str)
    - other_comments (str)
    - revision (str)
    - revision_comments (list of str)

    The `independent_variable_defintion` and `dependent_variable_definition` are dicts
    with entries of the form `{'VariableName':'units, standard name, [optional long name]'}`
    The keys must correspond to columns of `df`.
    `independent_variable_definition` should have only one key while 
    `dependent_variable_definition` can have many. For example,
    ``metadata.INDEPENDENT_VARIABLE_DEFINITION = 
            {'Time_Start':'seconds, time at start of measurement, seconds since midnight UTC'}``
    See Examples below.

        
    Examples
    --------
    ```
    import pandas as pd
    from acgc import icartt

    df = pd.DataFrame( [[1,0,30],
                        [2,10,29],
                        [3,20,27],
                        [4,30,25]],
                        columns=['Time_Start','Alt','Temp'])

    metadata = dict(
        INDEPENDENT_VARIABLE_DEFINITION = 
            {'Time_Start':'seconds, time, measurement time in seconds after takeoff'},
        DEPENDENT_VARIABLE_DEFINITION = 
            {'Alt':'m, altitude, altitude above ground level',
             'Temp':'C, temperature, air temperature in Celsius'},
        PI_NAME = 'Jane Doe',
        ORGANIZATION_NAME = 'NASA',
        SOURCE_DESCRIPTION = 'Invented Instrument',
        MISSION_NAME = 'FIREX-AQ',
        SPECIAL_COMMENTS = ['Special comments are optional and can be omitted.',
                        'If used, they should be a list of one or more strings'],
        PI_CONTACT_INFO = 'jdoe@email.com or postal address',
        PLATFORM = 'NASA DC-8',
        LOCATION = 'Boise, ID, USA',
        ASSOCIATED_DATA = 'N/A',
        INSTRUMENT_INFO = 'N/A',
        DATA_INFO = 'N/A',
        UNCERTAINTY = r'10% uncertainty in all values',
        ULOD_FLAG = '-7777',
        ULOD_VALUE = 'N/A',
        LLOD_FLAG = '-8888',
        LLOD_VALUE = 'N/A',
        DM_CONTACT_INFO = 'Alice, data manager, alice@email.com',
        STIPULATIONS_ON_USE = 'FIREX-AQ Data Use Policy',
        PROJECT_INFO = 'FIREX-AQ 2019, https://project.com',
        OTHER_COMMENTS = 'One line of comments',
        REVISION = 'R1',
        REVISION_COMMENTS = ['R0: Initial data',
                            'R1: One string per revision'],
        measurement_start_date = pd.Timestamp('2020-01-30 10:20')
        )
    
    icartt.write_icartt( 'test.ict', df, metadata )
    ```
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

    # Variables that will be written to file
    ictvars = list(_get(metadata,'INDEPENDENT_VARIABLE_DEFINITION')) + \
              list(_get(metadata,'DEPENDENT_VARIABLE_DEFINITION'))

    # Variables that are not in the dataframe
    missingvars = set(ictvars) - set(df.columns)

    # Raise an error if there are missing variables
    if len(missingvars)>0:
        raise KeyError('Some output variables are not in the DataFrame: '+str(missingvars))

    # Coerce to Timestamp
    measurement_start_date = pd.Timestamp( _get(metadata, 'measurement_start_date') )

    # Form the header
    header = []

    for k in ['PI_NAME',
              'ORGANIZATION_NAME',
              'SOURCE_DESCRIPTION',
              'MISSION_NAME',]:
        v = _get( metadata, k )
        header.append( v )

    # File volume
    header.append( '1, 1')

    # Date line
    header.append( measurement_start_date.strftime('%Y, %m, %d, ') +
                    pd.Timestamp.today().strftime('%Y, %m, %d'))

    # Time interval
    # Time spacing between records, set of unique values
    independent_variable_name = list(_get(metadata,'INDEPENDENT_VARIABLE_DEFINITION'))[0]
    dt = set( df[independent_variable_name].diff(1).dropna() )
    if len(dt)==1:
        # Constant time interval, use value
        interval = list(dt)[0]
    else:
        # Time interval is not constant so code as 0
        interval = 0
    header.append( str(interval) )

    # Independent variable 
    keydict = _get( metadata, 'INDEPENDENT_VARIABLE_DEFINITION' )
    header.append( list(keydict)[0] + ', ' + list(keydict.values())[0] )

    # Dependent variables
    keydict = _get( metadata, 'DEPENDENT_VARIABLE_DEFINITION' )
    nvars = len(keydict)
    header.append( str(nvars) )                 # Number of dependent variables
    header.append( ','.join(['1']*nvars) )      # Scale factors for dependent variables
    header.append( ','.join(['-9999']*nvars) )  # Missing data flags for dependent vars.
    for kn in keydict.keys():
        header.append( kn + ', ' + keydict[kn]) # Dependent variable definitions

    # Special comments
    v = _get( metadata, 'SPECIAL_COMMENTS' )
    if v:
        # Expect a string or array of several lines
        header.append( str(len(list(v))) )
        header.extend( list(v) )
    else:
        header.append( '0' )

    # Normal Comments
    nc= []
    for kn in normal_comments:
        v = _get( metadata, kn )
        if kn=='REVISION_COMMENTS':
            # Expect a string or array of several lines
            nc.extend( list(v) )
        else:    
            nc.append( '{:s}: {:s}'.format(kn,v) )
    # Variable short names
    nc.append( ', '.join(ictvars))
    # Add normal comments to the header
    header.append( str(len(nc)) )
    header.extend( nc )

    # Write the file
    with open(filename,'w',encoding='ascii') as f:
        f.write(f'{len(header)+1:d}, 1001\n')  # +1 accounts for this line
        for line in header:
            f.write(line+'\n')
        df[ictvars].to_csv( f,
                            index=False,
                            header=False,
                            na_rep='-9999',
                            **kwargs )
