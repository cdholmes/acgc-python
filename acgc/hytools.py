#!/usr/bin/env python3
'''Package of functions to read HYSPLIT output 

Assumed directory structure for meteorological data. 
All directories are archived meteorology except for "forecast/" directory.
metroot/
  forecast/ - All forecasts organized by initialization date
    YYYYMMDD/ 
  gdas1/
  gdas0p5/
  gfs0p25/
  hrrr/
  nam12/ - what ARL calls nams
  nam3/ - pieced forecast of NAM CONUS nest (3 km)
'''

import os
import glob
import datetime as dt
import numpy  as np
import pandas as pd
from . import nctools as nct

# Location for ARL met data
METROOT = '/data/MetData/ARL/'

def tdump2nc( inFile, outFile, clobber=False, globalAtt=None, 
             altIsMSL=False, dropOneTime=False, pack=False ):
    '''Convert a HYSPLIT tdump file to netCDF
    Works with single point or ensemble trajectories

    Parameters
    ----------
    inFile : str
        name/path of HYSPLIT tdump file
    outFile : str
        name/path of netCDF file that will be created
    clobber : bool (default=False)
        determines whether outFile will be overwrite any previous file
    globalAtt : dict (default=None)
        If present, dict keys will be added to outFile as global attributes
    altIsMSL : bool (default=False)
        Determines whether altitude in HYSPLIT tdump file is treated as altitude above sea level
        (altIsMSL=True) or altitude above ground (altIsMSL=False). In either case, the netCDF
        file will contain both altitude variables.
    dropOneTime : bool (default=False)
        Kludge to address back trajectories that start 1 minute after the hour,
        due to CONTROL files created with write_control(... exacttime=False )
        set True only for trajectories using this setup.
    pack : bool (default=False)
        NOT IMPLEMENTED
        determines whether variables in the netCDF file should be compressed with *lossy*
        integer packing. 
    '''
    # Return if the file already exists and not set to clobber
    if os.path.exists(outFile) and clobber==False:
        return

    # Trajectory points
    traj = read_tdump( inFile )

    # Trajectory numbers; convert to int32
    tnums = traj.tnum.unique().astype('int32')

    # Number of trajectories (usually 1 or 27)
    ntraj = len( tnums )

    # Trajectory start time
    starttime = traj.time[0] 

    # Time along trajectory, hours since trajectory start
    ttime  = traj.thour.unique().astype('f4')

    # Number of times along trajectory
    nttime = len( ttime )

    # Empty arrays
    lat    = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    lon    = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    alt    = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    altTerr= np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    p      = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    T      = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    Q      = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    U      = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    V      = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    precip = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    zmix   = np.zeros( (ntraj, nttime), np.float32 ) * np.nan
    inBL   = np.zeros( (ntraj, nttime), np.int8 )    * -9

    # Check if optional variables are present
    doP        = ('PRESSURE' in traj.columns)
    doTerr     = ('TERR_MSL' in traj.columns)
    doBL       = ('MIXDEPTH' in traj.columns)
    doT        = ('AIR_TEMP' in traj.columns)
    doQ        = ('SPCHUMID' in traj.columns)
    doU        = ('UWIND'    in traj.columns)
    doV        = ('VWIND'    in traj.columns)
    doPrecip   = ('RAINFALL' in traj.columns)

    for t in tnums:

        # Find entries for this trajectory
        idx = traj.tnum==t

        # Number of times in this trajectory
        nt = np.sum(idx)

        if dropOneTime:
            # Drop the second time element (at minute 0) to retain one point per hour
            # Find entries and Drop the second element
            tmpidx = np.where(traj.tnum==t)[0]
            idx = [tmpidx[0]]
            idx.extend(tmpidx[2:])

        # Save the coordinates
        lat[t-1,:nt] = traj.lat[idx]
        lon[t-1,:nt] = traj.lon[idx]
        alt[t-1,:nt] = traj.alt[idx]

        # Add optional variables
        if doP:
            p[t-1,:nt] = traj.PRESSURE[idx]
        if doT:
            T[t-1,:nt]      = traj.AIR_TEMP[idx]
        if doQ:
            Q[t-1,:nt]      = traj.SPCHUMID[idx]
        if doU:
            U[t-1,:nt]      = traj.UWIND[idx]
        if doV:
            V[t-1,:nt]      = traj.VWIND[idx]
        if doPrecip:
            precip[t-1,:nt] = traj.RAINFALL[idx]
        if doTerr:
            altTerr[t-1,:nt]= traj.TERR_MSL[idx]
        if doBL:
            inBL[t-1,:nt]   = traj.alt[idx] < traj.MIXDEPTH[idx]
            zmix[t-1,:nt]   = traj.MIXDEPTH[idx]

    if altIsMSL:
        altName=     'altMSL'
        altLongName= 'altitude above mean sea level'
        if doTerr:
            alt2Name=     'altAGL'
            alt2LongName= 'altitude above ground level'
            alt2=         alt-altTerr
    else:
        altName=     'altAGL'
        altLongName= 'altitude above ground level'
        if doTerr:
            alt2Name=     'altMSL'
            alt2LongName= 'altitude above mean sea level'
            alt2=         alt+altTerr

    # Put output variables into a list
    variables = [
        {'name':'lat',
            'long_name':'latitude of trajectory',
            'units':'degrees_north',
            'value':np.expand_dims(lat,axis=0),
            'fill_value':np.float32(np.nan)},
        {'name':'lon',
           'long_name':'longitude of trajectory',
           'units':'degrees_east',
           'value':np.expand_dims(lon, axis=0),
           'fill_value':np.float32(np.nan)},
        {'name':altName,
           'long_name':altLongName,
           'units':'m',
           'value':np.expand_dims(alt, axis=0),
           'fill_value':np.float32(np.nan)} ]

    # Add optional variables to output list
    if doTerr:
        variables.append(
           {'name':'altTerr',
           'long_name':'altitude of terrain',
           'units':'m',
           'value':np.expand_dims(altTerr,axis=0),
           'fill_value':np.float32(np.nan)} )
        variables.append(
           {'name':alt2Name,
           'long_name':alt2LongName,
           'units':'m',
           'value':np.expand_dims(alt2,axis=0),
           'fill_value':np.float32(np.nan)} )
    if doP:
        variables.append(
            {'name':'p',
           'long_name':'pressure',
           'units':'hPa',
           'value':np.expand_dims(p,axis=0),
           'fill_value':np.float32(np.nan)} )
    if doT:
        variables.append(
            {'name':'T',
           'long_name':'temperature',
           'units':'K',
           'value':np.expand_dims(T,axis=0),
           'fill_value':np.float32(np.nan)} )
    if doQ:
        variables.append(
            {'name':'q',
           'long_name':'specific humidity',
           'units':'g/kg',
           'value':np.expand_dims(Q,axis=0),
           'fill_value':np.float32(np.nan)} )
    if doU:
        variables.append(
            {'name':'U',
           'long_name':'eastward wind speed',
           'units':'m/s',
           'value':np.expand_dims(U,axis=0),
           'fill_value':np.float32(np.nan)} )
    if doV:
        variables.append(
            {'name':'V',
           'long_name':'northward wind speed',
           'units':'m/s',
           'value':np.expand_dims(V,axis=0),
           'fill_value':np.float32(np.nan)} )
    if doPrecip:
        variables.append(
            {'name':'precipitation',
           'long_name':'precipitation',
           'units':'mm/hr',
           'value':np.expand_dims(precip,axis=0),
           'fill_value':np.float32(np.nan)} )
    if doBL:
        variables.append(
            {'name':'inBL',
           'long_name':'trajectory in boundary layer flag',
           'units':'unitless',
           'value':np.expand_dims(inBL,axis=0),
           'fill_value':-9} )
        variables.append(
            {'name':'mixdepth',
           'long_name':'boundary layer mixing depth',
           'units':'m',
           'value':np.expand_dims(zmix,axis=0),
           'fill_value':np.float32(np.nan)} )

    # Add dimension information to all variables
    for v in range(len(variables)):
        variables[v]['dim_names'] = ['time','trajnum','trajtime']

    # Construct global attributes
    # Start with default and add any provided by user input
    gAtt = {'Content': 'HYSPLIT trajectory'}
    if isinstance(globalAtt, dict):
        gAtt.update(globalAtt)

    # Create the output file
    nct.write_geo_nc( outFile, variables,
        xDim={'name':'trajnum',
            'long_name':'trajectory number',
            'units':'unitless',
            'value':tnums},
        yDim={'name':'trajtime',
            'long_name':'time since trajectory start',
            'units':'hours',
            'value':ttime},
        tDim={'name':'time',
            'long_name':'time of trajectory start',
            'units':'hours since 2000-01-01 00:00:00',
            'calendar':'standard',
            'value':np.array([starttime]),
            'unlimited':True},
        globalAtt=gAtt,
        nc4=True, classic=True, clobber=clobber )

def read_tdump(file):
    '''Read trajectory file output from HYSPLIT
    
    Parameters
    ----------
    file : str
        name of trajectory file to read
        
    Returns
    -------
    Pandas dataframe containing data from file. Columns:
        time : datetime object
        year, month, day, hour, minute : floats, same as time
        lat, lon, alt : trajectory location
        thour : hours since trajectory initialization, negative for back trajectories
        tnum : trajectory number tnum=1 for single trajectory, tnum=1-27 for trajectory ensemble
        metnum : index number of met file used at this point in trajectory, 
            see tdump file for corresponding file paths
        fcasthr : hours since the meteorological dataset was initialized
    '''

    # Open the file
    with open(file,"r",encoding="ascii") as fid:

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
        df = pd.read_csv( fid, delim_whitespace=True,
                          header=None, index_col=False, na_values=['NaN','********'],
                names=['tnum','metnum',
                       'year','month','day','hour','minute','fcasthr',
                       'thour','lat','lon','alt']+vnames )

        # Convert 2-digit year to 4-digits
        df.loc[:,'year'] += 2000

        # Convert to time
        df['time'] = pd.to_datetime( df[['year','month','day','hour','minute']] )

        #print(df.iloc[-1,:])
        return df

def get_gdas1_filename( time ):
    '''Directory and File names for GDAS 1 degree meteorology, for given date'''

    # Directory for GDAS 1 degree
    dirname = METROOT+'gdas1/'

    # Filename template
    filetmp = 'gdas1.{mon:s}{yy:%y}.w{week:d}'

    # week number in the month
    wnum = ((time.day-1) // 7) + 1

    # GDAS 1 degree file
    filename = dirname + filetmp.format( mon=time.strftime("%b").lower(), yy=time, week=wnum )

    return filename

def get_hrrr_filename( time ):
    '''Directory and File names for HRRR meteorology, for given date'''

    # Directory
    dirname = METROOT+'hrrr/'

    # There are four files per day containing these hours
    hourstrings = ['00-05','06-11','12-17','18-23']

    filenames = [ '{:s}/{:%Y%m%d}_{:s}_hrrr'.format( dirname, time, hstr )
                  for hstr in hourstrings ]
    dirnames = [ dirname for i in range(4) ]

    # Return
    return filenames

def get_met_filename( metmodel, time ):
    '''Directory and file names for given meteorology and date
    
    Parameters
    ----------
    metmodel : str
        met model wanted: gdas1, gdas0p5, gfs0p25, nam12, nam3, hrrr
    time : pandas.Timestamp, datetime.date, datetime.datetime
        day of desired meteorology
    
    Returns
    -------
    filename : str or list
        name of met file or files containing met data for input date
    '''

    # Ensure that time is a datetime object
    if not isinstance( time, dt.date) :
        raise TypeError( "get_met_filename: time must be a datetime object" )

    # Directory and filename template
    doFormat=True
    if metmodel == 'gdas1':
        # Special treatment
        filename = get_gdas1_filename( time )
    elif metmodel == 'gdas0p5':
        dirname  = METROOT+'gdas0p5/'
        filename = dirname+'{date:%Y%m%d}_gdas0p5'
    elif metmodel == 'gfs0p25':
        dirname  = METROOT+'gfs0p25/'
        filename = dirname+'{date:%Y%m%d}_gfs0p25'
    elif metmodel == 'nam3':
        dirname  = METROOT+'nam3/'
        filename = dirname+'{date:%Y%m%d}_hysplit.namsa.CONUS'
    elif metmodel == 'nam12':
        dirname  = METROOT+'nam12/'
        filename = dirname+'{date:%Y%m%d}_hysplit.t00z.namsa'
    elif metmodel == 'hrrr':
        # Special treatment
        filename = get_hrrr_filename( time )
        doFormat=False
    else:
        raise NotImplementedError(
           f"get_met_filename: {metmodel} unrecognized" )

    # Build the filename
    if doFormat:
        filename = filename.format( date=time )

    return filename

def get_archive_filelist( metmodels, time, useAll=True ):
    '''Get a list of met directories and files
    When there are two met version provided, the first result will be used
    
        
    Parameters
    ----------
    metmodels : list or str
        met models wanted: gdas1, gdas0p5, gfs0p25, nam12, nam3, hrrr
        commonly provide a list with both regional and global models e.g. ['hrrr','gfs0p25']
    time : pandas.Timestamp, datetime.date, datetime.datetime
        day of desired meteorology
    useAll : bool (default=True)
        if True, then return files for all metmodels found
        if False, then return files only for the first metmodel found
    
    Returns
    -------
    filename : list
        names of files containing met data for input date
        typically 
    '''

    if isinstance( metmodels, str ):

        # If metmodels is a single string, then get value from appropriate function
        filename = get_met_filename( metmodels, time )

        # Convert to list, if isn't already
        if isinstance(filename, str):
            filename = [filename]

    elif isinstance( metmodels, list ):

        # If metmodels is a list, then get files for each met version in list

        filename = []

        # Loop over all the metmodelss
        # Use the first one with a file present
        for met in metmodels:

            # Find filename for this met version
            f = get_met_filename( met, time )

            if useAll:

                # Ensure that directory and file are lists so that we can use extend below
                if isinstance(f, str):
                    f = [f]
                elif isinstance(f, list):
                    pass
                else:
                    raise NotImplementedError(
                        'Variable expected to be string or list but is actually ',type(f) )

                # Append to the list so that all can be used
                filename.extend(f)

            else:
                # Use just the first met file that exists
                filename = f
                break
                # If the file exists, use this and exit;
                # otherwise keep looking
                #if isinstance( f, str ):
                #    if ( os.path.isfile( d+f ) ):
                #        dirname  = d
                #        filename = f
                #        break
                #elif isinstance( f, list ):
                #    if ( os.path.isfile( d[0]+f[0] ) ):
                #        dirname  = d
                #        filename = f
                #        break

        # Raise an error if 
        if filename is []:
            raise FileNotFoundError(
                "get_archive_filename: no files found for " + ','.join(metmodels) )

    else:
        raise TypeError( "get_archive_filelist: metmodels must be a string or list" )

    return filename

def get_hybrid_filelist( metmodels, time ):
    '''Get list of met files combining archive and forecast
    
    Starting 1 day before 'time' and extending 10 days ahead, find all archived analysis
    files, then add forecast files

    Parameters
    ----------
    metmodels : str 
        IGNORED
        Current implementation uses nam3 for past meteorology and namsfCONUS for forecast
    time : datetime.datetime or pandas.Timestamp
        date of interest, should be near present. 
        If farther in past, then use archived met. Distance future won't have any forecst available.

    Returns
    -------
    metfiles : list
        file path names that are found
    '''

    print( "Hybrid Archive/Forecast meteorology using NAM3 CONUS nest" )

    archivemet = 'nam3'
    forecastmet = 'namsfCONUS'

    # List of met directories and met files that will be used
    metfiles = []

    # Loop over 10 days, because that's probably enough for FIREX-AQ
    for d in range(-1,10):

        # date of met data
        metdate = time.date() + pd.Timedelta( d, "D" )

        #dirname, filename = get_archive_filelist( metmodel, metdate )
        filename = get_met_filename( archivemet, metdate )

        # Check if the file exists
        if os.path.isfile( filename ):

            # Add the file, if it isn't already in the list
            if filename not in metfiles:
                metfiles.append( filename )

        else:

            # We need to get forecast meteorology

            # Loop over forecast cycles
            for hr in [0,6,12,18]:

                cy =  dt.datetime.combine( metdate, dt.time( hour=hr ) )

                try:
                    # If this works, then the file exists
                    f = get_forecast_filename( forecastmet, cy, partial=True )
                    # Add the first 6-hr forecast period
                    metfiles.append( f[0] )
                    # If we have a full forecast cycle, save it
                    if len(f)==8:
                        flast = f
                except FileNotFoundError:
                    # We have run out of forecast meteorology,
                    # So add the remainder of the prior forecast cycle and we're done
                    metfiles.extend( flast[1:] )

                    return metfiles

    return metfiles

def get_forecast_template( metmodel ):
    '''Get filename template for forecast meteorology

    Parameters
    ----------
    metmodel : str
        name of forecast model: namsfCONUS, namf, gfs0p25, hrrr

    Returns
    -------
    template : str
        template filename with {:%H} field for initialization time and ?? for forecast hour
    nexpected : int
        number of forecast files expected for each forecast initialization
    '''

    if metmodel == 'namsfCONUS':
        # Filename template
        filetemplate = 'hysplit.t{:%H}z.namsf??.CONUS'
        nexpected = 8
    else:
        filetemplate = 'hysplit.t{:%H}z.'+metmodel
        nexpected = 1

    return filetemplate, nexpected

def get_forecast_filename( metmodel, cycle, partial=False ):
    '''Find files for a particular met model and forecast cycle
    
    Parameters
    ----------
    metmodel : str
        name of forecast model: namsfCONUS, namf, gfs0p25, hrrr
    cycle : datetime.datetime or pandas.Timestamp
        forecast initialization time UTC (date and hour)
    partial : bool (default=True)
        With partial=True, function will raise error if some forecast files are missing
        With partial=False, function will return all forecast files that are found

    Returns
    -------
    filenames : list
        file paths to all files that are found
    '''

    dirname  = METROOT + f'forecast/{cycle:%Y%m%d}/'
    filename, nexpected  = get_forecast_template( metmodel )

    # Filename for this cycle
    filename = filename.format(cycle)

    # Find all the files that match these criteria
    files = glob.glob( dirname + filename )

    # Check if we found the expected number of files
    if (len(files) == nexpected) or (partial and len(files) >=1):

        # When we find then, sort and combine into one list
        filenames = sorted( files )

        # Return
        return filenames

    # Raise an error if no forecasts are found
    raise FileNotFoundError('ARL forecast meteorology found' )

def get_forecast_filename_latest( metmodel ):
    '''Find files for the latest available forecast cycle for the requested met version.
    
    Requires a complete set of forecast files for a cycle

    Parameter
    ---------
    metmodel : str 
        name of forecast model: namsfCONUS, namf, gfs0p25, hrrr
    
    Returns
    -------
    filenames : list
        file paths to all files that are found    
    '''

    # Filename template for forecast files
    filetemplate, nexpected = get_forecast_template( metmodel )

    # Find all of the forecast directories, most recent first
    dirs = [item for item
            in sorted( glob.glob( METROOT+'forecast/????????' ), reverse=True )
            if os.path.isdir(item) ]

    for d in dirs:

        # Loop backwards over the forecast cycles
        for hh in [18,12,6,0]:

            # Check if the forecast files exist
            files = glob.glob(d+'/'+filetemplate.format(dt.time(hh)))
            #'hysplit.t{:02d}z.{:s}'.format(hh,metsearch))

            # Check if we found the expected number of files
            if len(files) == nexpected:

                # When we find then, sort and combine into one list
                filenames = sorted( files )

                # Return
                return filenames

    # Raise an error if no forecasts are found
    raise FileNotFoundError('No ARL forecast meteorology found' )

def get_forecast_filelist( metmodels=None, cycle=None ):
    '''Get list of filenames for forecast cycle
    
    This function calls itself recursively

    Parameters
    ----------
    metmodels : str or list (default=['namsfCONUS','namf'])
        name of forecast model: namsfCONUS, namf, gfs0p25, hrrr
    cycle : datetime.datetime or pandas.Timestamp or None
        forecast initialization time UTC (date and hour) 
        if cycle=None, then will use the latest available forecast cycle       

    Returns
    -------
    filenames : list
        file paths to all files that are found            
    '''

    if metmodels is None:
        metmodels = ['namsfCONUS','namf']

    # If metmodel is a single string, then get value from appropriate function
    if isinstance( metmodels, str ):

        if cycle is None:
            filenames = get_forecast_filename_latest( metmodels )
        else:
            filenames = get_forecast_filename( metmodels, cycle )

    else:

        filenames = []

        # Loop over the list of met types, combine them all
        for met in metmodels:

            # Get directory and file names for one version
            f = get_forecast_filelist( met, cycle )

            # Combine them into one list
            filenames = filenames + f

    return filenames

def find_arl_metfiles( start_time, ndays, back=False, metmodels=None,
                       forecast=False, forecastcycle=None, hybrid=False ):
    '''Find ARL meteorology files for specified dates, models, forecast and archive
    
    Files will be located for start_time and extending ndays forward or backward

    Parameters
    ----------
    start_time : datetime.datetime or pandas.Timestamp
        start date and time for finding meteorology data files
    ndays : int
        number of days of files to retrieve
    back : bool (default=False)
        specifies files should go ndays backward (back=True) from start_time
    metmodels : list or str (default=['gdas0p5','gdas1'])
        meteorological models that will be used, in order of decreasing resolution and priority
    forecast : bool (default=False)
        set forecast=True to use forecast meteorology for trajectory computation
        set forecast=False to use archived (past) meteorology for trajectory computation
        Note: hybrid=True will supercede forecast=True
    forecastcycle : datetime.datetime, pandas.Timestamp, or None (default=None)
        if forecast=True, this sets the forecast initialization cycle that will be used
        set forecastcycle=None to use the latest available cycle for which files are found
        if forecast=False, this parameter has no effect
    hybrid : bool (default=False)
        set hybrid=True for trajectories that use a combination of past archive and 
        forecast meteorlogy. This supercedes forecast=True

    Returns
    -------
    metfiles : list
        path to meteorology files meeting the criteria
    '''

    if metmodels is None:
        metmodels = ['gdas0p5','gdas1']

    # Find the meteorology files
    if hybrid is True:

        # "Hybrid" combines past (archived) and future (forecast) meteorology
        if back:
            raise NotImplementedError( "Combined Analysis/Forecast meteorology "
                                   + "not supported for back trajectories" )
        metfiles = get_hybrid_filelist( metmodels, start_time )

    elif forecast is True:
        # Get the forecast meteorology
        metfiles = get_forecast_filelist( metmodels, forecastcycle )

        # Check if the forecast meteorology covers the entire trajectory duration
    else:
        # Use archived (past) meteorology

        # Relative to trajectory start day,
        # we need meteorology for days d0 through d1
        if back:
            d0 = -ndays+1
            d1 = 1
            # For trajectories that start 23-0Z (nam) or 21-0Z (gfs0p25), 
            # also need the next day to bracket first time step
            if   (('nam3'    in metmodels) and (start_time.hour==23)) \
              or (('gfs0p25' in metmodels) and (start_time.hour>=21)):
                d1 = 2
        else:
            d0 = 0
            d1 = ndays
            # For trajectories that start 00-01Z (nam) or 00-03Z (gfs0p25), 
            # also need the prior day to bracket first time step
            if start_time.hour==0:
                d0 = -1

        # Debug output
        # print('Initial Date Range',d0,d1)
        # datelist = np.unique( ( time +
        #   pd.TimedeltaIndex( np.sign(trajhours) * np.arange(0,np.abs(trajhours)+1),"H" ) ).date )
        # print(datelist)

        # List of met directories and met files that will be used
        metfiles = []
        for d in range(d0,d1):

            # date of met data
            metdate = start_time.date() + pd.Timedelta( d, "D" )

            # Met data for a single day 
            filename = get_archive_filelist( metmodels, metdate )

            # Add the files to the list
            metfiles.extend( filename )
        # Keep only the unique files
        metfiles = np.unique(metfiles)

    if len(metfiles) <= 0:
        raise ValueError( 'Meteorology files not found for ' + str( metmodels ) )

    return metfiles

def write_control( time, lat, lon, alt, trajhours,
                   fname='CONTROL.000', clobber=False,
                   maxheight=15000., outdir='./', tfile='tdump',
                   metfiles=None, exacttime=True, **kwargs ):
    '''Write HYSPLIT control file for trajectory starting at designated time and coordinates
    
    Parameters
    ----------
    time : datetime.datetime or pandas.Timestamp
        trajectory initialization time
    lat, lon : float or list
        trajectory initialization latitude and longitude in degrees
    alt : float or list
        trajectory initialization altitude in meters
        The setup.cfg determines whether this is above ground or above mean sea level
    trajhours : int
        desired trajectory duration in hours. Use negative for back trajectories
    fname : str (default='CONTROL.000')
        path and name for the file that will be written
    clobber : bool (default=False)
        if clobber=True, then fname will be overwritten
    maxheight : float (default=15000)
        terminate trajectories that exceed maxheight altitude in meters
    outdir : str (default='./')
        directory path where HYSPLIT output will be written
    tfile : str (default='tdump')
        name of the trajectory file that HYSPLIT will write
    metfiles : list, str (default=None) 
        paths to ARL meteorology files needed for the trajectory computation
        If metfiles=None, then find_arl_metfiles will be used to locate necessary files
    exacttime : bool (default=True)
        It is not recommended to change this default, but keyword is retained for backward
        compatibility with some scripts. Setting exacttime=False will shift the actual 
        start time of trajectories that begin at 00:00 UTC to 00:01 UTC to avoid reading
        an additional day of meteorological data.
    **kwargs :
        kwargs will be passed to find_arl_metfiles to locate ARL meteorlogy files if metfiles=None
        These keywords should include metmodels and possibly forecast, forecastcycle, or hybrid
        See find_arl_metfiles for definitions of these parameters 
    '''

    if os.path.isfile( fname ) and (clobber is False):
        raise OSError( f'File exists. Set clobber=True to overwrite: {fname:s}' )

    # Ensure that lat, lon, and alt are lists
    try:
        nlat = len( lat )
    except TypeError:
        lat = [ lat ]
        nlat = len( lat )
    try:
        nlon = len( lon )
    except TypeError:
        lon = [ lon ]
        nlon = len( lon )
    try:
        nalt = len( alt )
    except TypeError:
        alt = [ alt ]
        nalt = len( alt )

    # Should add some type checking to ensure that lon, lat, alt
    # have conformable lengths
    if ( (nlat != nlon) or (nlat != nalt) ):
        raise ValueError( "lon, lat, alt must be conformable" )

    # Number of days of met data
    ndays = int( np.ceil( np.abs( trajhours ) / 24 ) + 1 )

    # Get met file names, if not provided
    if (metfiles is None) or (len(metfiles)==0):
        metfiles = find_arl_metfiles( time, ndays, back=trajhours<0, **kwargs )

    # Number of met files
    nmet = len( metfiles )

    # Runs will fail if the initial time is not bracketed by met data.
    # When met resolution changes on the first day, this condition may not be met.
    # Unless exacttime==True, the starting time will be shifted 1 minute to 00:01
    # to avoid reading in an entire extra day of met data.
    if (time.hour==0 and time.minute == 0 and not exacttime):
        time = time + pd.Timedelta( 1, "m" )

    # Start date-time, formatted as YY MM DD HH {mm}
    startdate = time.strftime( "%y %m %d %H %M" )

    # Write the CONTROL file
    with open( fname, 'w', encoding='ascii' ) as f:

        f.write( startdate+'\n' )
        f.write( f'{nlat:d}\n' )
        for i in range(nlat):
            f.write( f'{lat[i]:<10.4f} {lon[i]:<10.4f} {alt[i]:<10.4f}\n' )
        f.write( f'{trajhours:d}\n' )
        f.write( "0\n" )
        f.write( f"{maxheight:<10.1f}\n" )
        f.write( f'{nmet:d}\n' )
        for file in metfiles:
            f.write( os.path.dirname(  file ) + '/\n' )
            f.write( os.path.basename( file ) + '\n'  )
        f.write( f'{outdir:s}\n' )
        f.write( f'{tfile:s}\n' )
