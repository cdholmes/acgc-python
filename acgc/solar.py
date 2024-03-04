#!/usr/local/bin/env python3
'''Module for calculating solar position and TOA insolation

The functions here are are vectorized and generally broadcast over xarray dimensions,
making this program faster than PySolar and pvlib. Calculations here use orbital parameters
from the NOAA Solar Position Calculator, following Jean Meeus Astronomical Algorithms, unless
the "fast" keyword is used. These accurate calculations are suitable for years -2000 to +3000.
The "fast" calculations have lower accuracy orbital parameters and coarser approximation 
for the equation of time. All calculations here use geocentric solar position, neglecting the 
parallax effect of viewing the sun from different points on Earth (i.e. topocentric vs. 
geocentric in NREL SPA algorithm).

Accuracy:
The NREL SPA algorithm in pvlib is used as an accurate reference.
The maximum error in solar zenith angle computed here is 0.02° over 1900-2100.
The maximum error in overall solar angular position is 0.022°.
Large apparent differences in azimuth alone can occur when the sun is near zenith or nadir,
where a small angular displacement results in a large azimuthal change.

The "fast" calculations have typical errors of ~0.2°.

NOAA Solar Calculator
https://gml.noaa.gov/grad/solcalc/calcdetails.html
'''

from collections import namedtuple
import warnings
import numpy as np
import pandas as pd
import xarray as xr

def insolation_toa( lat, lon, datetime, solar_pos=None, **kwargs ):
    '''Insolation at top of the atmosphere, accounting for solar zenith angle
    
    Parameters
    ----------
    lat : float or ndarray
        latitude in degrees
    lon : float or ndarray
        longitudes in degrees
    datetime : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`
    **kwargs passed to `solar_zenith_angle`
    
    Returns
    -------
    Insolation : float
        radiation flux density accounting for solar zenith angle, W/m2
    '''

    if solar_pos is None:
        solar_pos = solar_position( datetime )

    S = solar_constant(datetime, solar_pos=solar_pos)
    sza = solar_zenith_angle(lat, lon, datetime, **kwargs, solar_pos=solar_pos )

    return S * np.cos(sza)

def solar_azimuth_angle( lat, lon, datetime, solar_pos=None ):
    '''Solar azimuth angle (degrees) for a latitude, longitude, date and time
    
    SAA is degrees clockwise from north.
    
    Parameters
    ----------
    lat : float or ndarray
        latitude in degrees
    lon : float or ndarray
        longitudes in degrees
    datetime : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`

    Returns
    -------
    saa : float or ndarray
        solar azimuth angle in degrees (clockwise from north)
    '''
    # Convert to pandas Timestamp, if needed
    datetime = _to_datetime(datetime)

    # Subsolar point, latitude longitude, degrees
    solar_lat = solar_latitude( datetime, solar_pos=solar_pos )
    solar_lon = solar_longitude( datetime, solar_pos=solar_pos )

    # Vector pointing toward sun
    x = np.cos( solar_lat * pi180 ) * np.sin( (solar_lon - lon) * pi180 )
    y = np.cos( lat*pi180 ) * np.sin( solar_lat*pi180 ) \
        - np.sin( lat*pi180 ) * np.cos( solar_lat*pi180 ) \
            * np.cos( (solar_lon - lon) * pi180 )

    # Azimuth angle from north, degrees
    saa = np.arctan2( x, y ) / pi180

    # Change range [-180,180] to [0,360]
    return np.mod( saa+360, 360 )

def solar_elevation_angle( lat, lon, datetime, alt=0,
                        refraction=False, temperature=10., pressure=101325.,
                        solar_pos=None ):
    '''Solar elevation angle (degrees) above the horizon

    The altitude parameter should be the vertical distance 
    above the surrounding terrain that defines the horizon,
    not necessarily the altitude above sea level or the altitude above ground level.
    For example, on a mountain peak that is 4000 m above sea level and 
    1500 m above the surrounding plateau, the relevant altitude is 1500 m.
    For an observer on the plateau, the relevant altitude is 0 m.

    See documentation for `solar_zenith_angle` and `horizon_zenith_angle`.

    Parameters
    ----------
    lat : float or ndarray
        latitude in degrees
    lon : float or ndarray
        longitudes in degrees
    datetime : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    alt : float or ndarray (default=0)
        altitude above surrounding terrain that defines the horizon, meters
    refraction : bool (default=False)
        specifies whether to account for atmospheric refraction
    temperature : float or ndarray (default=10)
        surface atmospheric temperature (Celsius), only used for refraction calculation
    pressure : float or ndarray (default=101325)
        surface atmospheric pressure (Pa), only used for refraction calculation
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`
    
    Returns
    -------
    sea : float or ndarray
        solar elevation angle in degrees at the designated locations and times
        If refraction=False, this is the true solar elevation angle
        If refraction=True, this is the apparent solar elevation angle
    
    '''

    if refraction and np.any(alt):
        warnings.warn( 'Atmospheric refraction is calculated for surface conditions, '
                    + 'but an altitude above the surface was specified',
                     category=UserWarning,
                     stacklevel=2 )

    sea = horizon_zenith_angle( alt ) \
         - solar_zenith_angle( lat, lon, datetime, refraction, temperature, pressure, 
                               solar_pos=solar_pos )

    return sea

def solar_zenith_angle( lat, lon, datetime,
                        refraction=False, temperature=10., pressure=101325.,
                        solar_pos=None ):
    '''Solar zenith angle (degrees) for a given latitude, longitude, date and time.
    
    Accounts for equation of time and (optionally) for atmospheric refraction.
    Altitude of the observer is not accounted for, which can be important when the sun 
    is near the horizon. 
    
    Results are accurate to tenths of a degree, except where altitude is important
    (< 20 degrees solar elevation)

    Parameters
    ----------
    lat : float or ndarray
        latitude in degrees
    lon : float or ndarray
        longitudes in degrees
    datetime : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    refraction : bool (default=False)
        specifies whether to account for atmospheric refraction
    temperature : float or ndarray (default=10)
        surface atmospheric temperature (Celsius), only used for refraction calculation
    pressure : float or ndarray (default=101325)
        surface atmospheric pressure (Pa), only used for refraction calculation
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`
    
    Returns
    -------
    sza : float or ndarray
        solar zenith angle in degrees at the designated locations and times
        If refraction=False, this is the true solar zenith angle
        If refraction=True, this is the apparent solar zenith angle
    '''
    # Convert to pandas Timestamp, if needed
    datetime = _to_datetime(datetime)

    # Solar declination, degrees
    if solar_pos is None:
        dec = solar_declination( datetime )
    else:
        dec = solar_pos.declination

    # Hour angle, degrees
    Ha = solar_hour_angle( lon, datetime, solar_pos )

    # True solar zenith angle, radians
    sza = np.arccos( np.sin(lat*pi180) * np.sin(dec*pi180) + \
          np.cos(lat*pi180) * np.cos(dec*pi180) * np.cos(Ha*pi180) )

    # Convert radians -> degrees
    sza /= pi180

    if refraction:
        # Subtract refraction angle (degrees) from zenith angle.
        # SZA is always smaller due to refraction.
        sza -= refraction_angle( 90-sza, pressure, temperature )

    return sza

def sunrise_time( *args, **kwargs ):
    '''Compute sunrise time
    
    See `sun_times` for Parameters.'''
    result = sun_times( *args, **kwargs )
    return result[0]

def sunset_time( *args, **kwargs ):
    '''Compute sunset time
    
    See `sun_times` for Parameters.'''
    result = sun_times( *args, **kwargs )
    return result[1]

def day_length( *args, **kwargs ):
    '''Compute length of daylight
    
    See `sun_times` for Parameters.'''
    result = sun_times( *args, **kwargs )
    return result[2]

def solar_noon( *args, **kwargs ):
    '''Compute time of solar noon (meridian transit)
    
    See `sun_times` for Parameters.'''
    result = sun_times( *args, **kwargs )
    return result[3]

def solar_constant( datetime, solar_pos=None ):
    '''Compute solar constant for specific date or dates
    
    Parameters
    ----------
    datetime : datetime-like
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`

    Returns
    -------
    S : float
        Solar direct beam radiation flux density, W/m2
    '''
    if solar_pos is None:
        solar_pos = solar_position( datetime )
    S = 1361/solar_pos.distance**2

    return S

def solar_declination( datetime, fast=False, solar_pos=None ):
    '''Calculate solar declination (degrees) for specified date
        
    Parameters
    ----------
    datetime : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    fast : bool (default=False)
        Specifies using a faster but less accurate calculation
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`

    Returns
    -------
    dec : float
        solar declination in degrees at the specified date
    '''
    # Convert to pandas Timestamp, if needed
    datetime = _to_datetime(datetime)

    # Select the accurate or fast calculation
    accurate = not fast

    if solar_pos is not None:
        dec = solar_pos.declination

    else:
        if accurate:

            # Solar declination, degrees
            dec, junk, junk, junk, junk = solar_position( datetime )

        else:
            # The fast method implements
            # Eq. 9.68-9.72 from M.Z. Jacobson, Fundamentals of Atmospheric Modeling

            # Number of days since beginning of 2000
            NJD = datetime - np.datetime64('2000-01-01')
            try:
                NJD = NJD.dt.days
            except AttributeError:
                NJD = NJD.days

            # Obliquity, degrees
            ob = 23.439 - 4e-7 * NJD

            # Parameters for ecliptic, degrees
            gm = 357.528 + 0.9856003 * NJD
            lm = 280.460 + 0.9856474 * NJD

            # Ecliptic longitude of sun, degrees
            ec = lm + 1.915 * np.sin( gm * pi180 ) + 0.020 * np.sin( 2 * gm * pi180 )

            #Solar declination, degrees
            dec = np.arcsin( np.sin( ob * pi180 ) * np.sin( ec * pi180 ) ) / pi180

    return dec

def solar_latitude( *args, **kwargs ):
    '''Latitude of the subsolar point
    
    Parameters
    ----------
    datetime : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    fast : bool
        see `solar_declination`
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`
        
    Returns
    -------
    latitude : float
        degrees of latitude
    '''
    return solar_declination( *args, **kwargs )

def solar_longitude( datetime, solar_pos=None ):
    '''Longitude of the subsolar point, degrees
    
    Parameters
    ----------
    datetimeUTC : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`
    
    Returns
    -------
    longitude : float
        degrees of longitude
    '''
    # Convert to pandas Timestamp, if needed
    datetimeUTC, tz_in = _to_datetime_utc(datetime)

    # Longitude of subsolar point, degrees
    # Equation of time will be added below
    try:
        # Treat as xarray.DataArray or pandas.Series
        solar_lon = - 15 * ( datetimeUTC.dt.hour +
                          datetimeUTC.dt.minute / 60 +
                          datetimeUTC.dt.second / 3600 - 12 )
    except AttributeError:
        solar_lon = - 15 * ( datetimeUTC.hour +
                          datetimeUTC.minute / 60 +
                          datetimeUTC.second / 3600 - 12 )

    # Add equation of time to the solar longitude, degrees
    if solar_pos is None:
        eot = equation_of_time( datetimeUTC, degrees=True )
    else:
        eot = solar_pos.equation_of_time
    solar_lon -= eot

    return solar_lon

def solar_hour_angle( lon, datetime, solar_pos=None ):
    '''Solar hour angle (degrees) for specified longitude, date and time

    Hour angle is the angular displacement of the sun from the local meridian.
    It is zero at local noon, negative in the morning, and positive is afternoon.
    
    Parameters
    ----------
    lon : float
        longitude in degrees east
    datetimeUTC : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`
    
    Returns
    -------
    ha : float
        hour angle in degrees at the specified location and time
    '''

    # Subsolar longitude, degrees
    solar_lon = solar_longitude(datetime, solar_pos )

    # Hour angle, degrees
    Ha = lon - solar_lon

    return Ha

def equation_of_time( datetime, degrees=False, fast=False ):
    '''Equation of time for specified date
    
    Accounts for the solar day being slightly different from 24 hours

    Parameters
    ----------
    datetime : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    degrees : bool (default=False)
        If True, then return value in compass degrees
        If False, then return value in minutes of an hour
    fast : bool (default=False)
        specifies whether to use a faster, but less accurate calculation
        
    Returns
    -------
    eot : float
        equation of time on the specified date, degrees or minutes
    '''
    # Convert to pandas Timestamp, if needed
    datetime = _to_datetime(datetime)

    # Determine whether to use the fast or accurate calculation
    accurate = not fast

    if accurate:

        # Equation of time, minutes
        junk, junk, eot, junk, junk = solar_position( datetime )

    else:
        # Implements the "alternative equation" from Wikipedia, derived from
        # https://web.archive.org/web/20120323231813/http://www.green-life-innovators.org/tiki-index.php?page=The%2BLatitude%2Band%2BLongitude%2Bof%2Bthe%2BSun%2Bby%2BDavid%2BWilliams
        # When compared to the NREL SPA algorithm, differences reach are up to about 0.5 minute.
        # Note: Leap years are not accounted for.

        # Equation of time, accounts for the solar day differing slightly from 24 hr
        try:
            doy = datetime.dt.dayofyear
        except AttributeError:
            doy = datetime.dayofyear
        W = 360 / 365.24
        A = W * (doy+10)
        B = A + 1.914 * np.sin( W * (doy-2) * pi180 )
        C = ( A - np.arctan2( np.tan(B*pi180), np.cos(23.44*pi180) ) / pi180 ) / 180

        # Equation of time in minutes of an hour (1440 minutes per day)
        eot = 720 * ( C - np.round(C) )

    # Equation of time, minutes -> degrees (360 degrees per day)
    if degrees:
        eot = eot / 60 * 360 / 24

    return eot

def solar_position( datetime ):
    '''Compute position of sun (declination, right ascension, equation of time, distance)) on specified date
    
    Calculations follow the NOAA solar calculator spreadsheet
    Applicable to years 1900-2100.

    Parameters
    ----------
    date : datetime-like or str
        date and time. Include time zone or UTC will be assumed
    
    Returns
    -------
    result : `SolarPositionResults`
        declination, right ascension, equation of time, Earth-sun distance, datetime
    '''
    # Ensure time is Timestamp in UTC
    datetimeUTC, tz_in = _to_datetime_utc(datetime)

    # Raise warning if any dates are outside date range
    # recommended for orbital parameters used here
    if np.logical_or( np.any( datetimeUTC < np.datetime64('1900-01-01') ),
                      np.any( datetimeUTC > np.datetime64('2100-01-01') ) ):
        warnings.warn('Solar position accuracy declines for dates outside 1900-2100', \
                      RuntimeWarning )

    # Number of days since 1 Jan 2000
    NJD = datetimeUTC - np.datetime64('2000-01-01')
    try:
        NJD = NJD.dt.days \
            + NJD.dt.seconds / 86400.
    except AttributeError:
        NJD = NJD.days \
            + NJD.seconds / 86400.

    # Julian day (since 12:00UTC 1 Jan 4713 BCE)
    NJD += 2451544.50

    # Julian century
    JC = (NJD-2451545)/36525

    # Earth orbital eccentricity, unitless
    ec = 0.016708634 - JC*( 0.000042037 + 0.0000001267*JC )

    # Earth mean orbital obliquity, degree
    mean_ob = 23 + ( 26 + ( (21.448
                             - JC * (46.815
                                    + JC * (0.00059 - JC * 0.001813) ) ) )/60 )/60

    # Earth true orbital obliquity, corrected for nutation, degree 
    ob = mean_ob + 0.00256 * np.cos( (125.04 - 1934.136*JC ) * pi180 )

    # Sun Mean ecliptic longitude, degree
    mean_ec_lon = np.mod( 280.46646 + JC*( 36000.76983 + JC*0.0003032 ), 360 )

    # Sun Mean anomaly, degree
    mean_anom = 357.52911 + JC*( 35999.05029 - 0.0001537*JC )

    # Sun Equation of center, degree
    eq_center = np.sin(mean_anom*pi180) * (1.914602 - JC*( 0.004817 + 0.000014*JC )) \
                    + np.sin(2*mean_anom*pi180) * (0.019993 - 0.000101*JC) \
                    + np.sin(3*mean_anom*pi180) * 0.000289

    # Sun True ecliptic longitude, degrees
    true_ec_lon = mean_ec_lon + eq_center

    # Sun True anomaly, degree
    true_anom = mean_anom + eq_center

    # Earth-Sun distance, AU
    distance = (1.000001018 * (1-ec**2) ) / (1 + ec * np.cos( true_anom * pi180 ))

    # Sun Apparent ecliptic longitude, corrected for nutation, degrees
    ec_lon = true_ec_lon - 0.00569 - 0.00478 * np.sin( (125.04 - 1934.136*JC ) * pi180)

    # Sun Right ascension, deg
    right_ascension = np.arctan2( np.cos(ob*pi180) * np.sin(ec_lon*pi180),
                                  np.cos(ec_lon*pi180) ) / pi180

    # Sun Declination, deg
    declination = np.arcsin( np.sin(ob*pi180) * np.sin(ec_lon*pi180) ) / pi180

    # var y
    vary = np.tan( ob/2 * pi180 )**2

    # Equation of time, minutes
    eot = vary * np.sin( 2 * mean_ec_lon * pi180) \
        - 2 * ec * np.sin( mean_anom * pi180 ) \
        + 4 * ec * vary * np.sin( mean_anom * pi180 ) * np.cos( 2 * mean_ec_lon * pi180) \
        - 0.5 * vary**2 * np.sin( 4 * mean_ec_lon * pi180) \
        - 1.25 * ec**2 * np.sin( 2 * mean_anom * pi180)
    eot = eot * 4 / pi180

    # Bundle results
    result = SolarPositionResults(declination,
                                  right_ascension,
                                  eot,
                                  distance,
                                  datetimeUTC)

    return result

def sun_times( lat, lon, datetime, tz_out=None, sza_sunrise=90.833,
              fast=False, solar_pos=None ):
    '''Compute times of sunrise, sunset, solar noon, and day length
    
    Common options for solar zenith angle at sunrise
    1. 90.833 for first edge of sun rising, typical (0.567°) refraction (default)
    2. 90.267 for first edge of sun rising, no refraction
    3. 90 degrees for center of sun rising, no refraction

    Note: horizon_zenith_angle can be used to compute a more accurate horizon location
    for sites at elevation.
    
    Parameters
    ----------
    lat : float or ndarray
        latitude in degrees
    lon : float or ndarray
        longitudes in degrees
    datetime : datetime-like or str
        datetime, provide a time zone or UTC will be assumed 
    tz_out : str, pytz.timezone, datetime.tzinfo, optional
        timezone to be used for output times. 
        If None is provided, then result will be in same time zone as input or UTC
    sza_sunrise : float (default=90.833)
        Solar zenith angle at which sunrise and sunset are calculated, degrees
    fast : bool (default=False)
        Select a faster but less accurate calculation
    solar_pos : `SolarPositionResults`, optional
        solar position parameters from a prior call to `solar_position`

    Returns
    -------
    result : `SunTimesResults`
        times of sunrise, sunset, solar noon, and day length
    '''
    # Convert to pandas Timestamp in UTC, if needed
    datetimeUTC, tz_in = _to_datetime_utc(datetime)

    # If no output timezone is specified, use the input time zone
    if tz_out is None:
        tz_out = tz_in

    # Select fast or accurate calculation
    accurate = not fast

    # Solar declination (degrees) and equation of time (minutes)
    if solar_pos is not None:
        dec = solar_pos.declination
        eot = solar_pos.eot
    else:
        if accurate:
            dec, junk, eot, junk, junk = solar_position( datetimeUTC )
        else:
            dec = solar_declination( datetimeUTC )
            eot = equation_of_time( datetimeUTC )

    # Sunrise hour angle, degree
    # Degrees east of the local meridian where sun rises
    ha_sunrise = np.arccos( np.cos(sza_sunrise*pi180) /
                           (np.cos(lat*pi180) * np.cos(dec*pi180))
                           - np.tan(lat*pi180)*np.tan(dec*pi180) ) / pi180

    # Solar noon, local standard time, day fraction
    solar_noon = (720 - 4*lon - eot ) / 1440

    # Sunrise and sunset, local standard time, day fraction
    t_sunrise = solar_noon - 4 * ha_sunrise / 1440
    t_sunset  = solar_noon + 4 * ha_sunrise / 1440

    # Midnight UTC
    # datetimeUTC is in UTC but time-zone-naive
    try:
        # Series time objects
        dateUTC = datetimeUTC.dt.floor('D')
    except AttributeError:
        # Scalar time objects
        dateUTC = datetimeUTC.floor('D')

    # Convert day fraction -> date time
    solar_noon = dateUTC + solar_noon * pd.Timedelta( 1, 'day' )
    t_sunrise  = dateUTC + t_sunrise  * pd.Timedelta( 1, 'day' )
    t_sunset   = dateUTC + t_sunset   * pd.Timedelta( 1, 'day' )

    # Convert to output timezone, if any is provided
    if tz_out is not None:
        if isinstance(solar_noon,(xr.DataArray,np.ndarray)) or \
            isinstance(t_sunrise,(xr.DataArray,np.ndarray)):
            # These types don't localize tz, but we can add offset to the tz-naive time
            if hasattr(datetimeUTC,'tz_localize'):
                # For scalar datetime, there is a single time offset, which we can add
                utcoffset = np.timedelta64( datetimeUTC.tz_localize(tz_out).utcoffset() )
                solar_noon += utcoffset
                t_sunrise  += utcoffset
                t_sunset   += utcoffset
            else:
                # For Series datetime, there are potentially multiple offsets. We can only add one
                unique_datetimeUTC = pd.DatetimeIndex(np.unique(datetimeUTC))
                unique_utcoffsets = np.unique( unique_datetimeUTC.tz_localize('UTC') \
                                        - unique_datetimeUTC.tz_localize(tz_out) )
                if len(unique_utcoffsets)==1:
                    utcoffset = unique_utcoffsets[0]
                    solar_noon += utcoffset
                    t_sunrise  += utcoffset
                    t_sunset   += utcoffset
                else:
                    # We might be able to handle multiple offsets if we can
                    # determine which dimension of `solar_noon` and `t_sunrise`
                    # is the time dimension.
                    raise ValueError('Multiple timezone offsets not supported. '
                                     +'Request output in UTC or reduce number of input times.')
        else:
            try:
                # Series time objects
                solar_noon = solar_noon.dt.tz_localize('UTC')\
                                    .dt.tz_convert(tz_out)
                t_sunrise  = t_sunrise.dt.tz_localize('UTC')\
                                    .dt.tz_convert(tz_out)
                t_sunset   = t_sunset.dt.tz_localize('UTC')\
                                    .dt.tz_convert(tz_out)
            except AttributeError:
                # Scale time objects
                solar_noon = solar_noon.tz_localize('UTC')\
                                    .tz_convert(tz_out)
                t_sunrise  = t_sunrise.tz_localize('UTC')\
                                    .tz_convert(tz_out)
                t_sunset   = t_sunset.tz_localize('UTC')\
                                    .tz_convert(tz_out)

    # Sunlight duration, minutes
    day_length = 8 * ha_sunrise * pd.Timedelta(1, 'minute')

    result = SunTimesResults(t_sunrise, t_sunset, day_length, solar_noon, datetimeUTC)

    return result

def horizon_zenith_angle( alt ):
    '''Angle from the zenith to the horizon
    
    The horizon is the locii of points where a line from the 
    observation location to the ellipsoid is tangent to the ellipsoid surface.
    
    The altitude parameter should be the vertical distance 
    above the surrounding terrain that defines the horizon,
    not necessarily the altitude above sea level or the altitude above ground level.
    For example, on a mountain peak that is 4000 m above sea level and 
    1500 m above the surrounding plateau, the relevant altitude is 1500 m.
    For an observer on the plateau, the relevant altitude is 0 m.

    The implementation below assumes a spherical Earth.
    Results using the WGS84 ellipsoid (see commented code below)
    differ from the spherical case by << 1°. Terrain,
    which is neglected here, has a larger effect on the horizon
    location, so the simpler spherical calculation is appropriate. 

    Parameters
    ----------
    lat : float or ndarray
        latitude in degrees
    alt : float or ndarray
        altitude above surrounding terrain that defines the horizon, meters
        
    Returns
    -------
    hza : float or ndarray
        horizon zenith angle in degrees
    '''

    # WGS84 ellipsoid parameters
    # semi-major radius, m
    r_earth = 6378137.0
    # ellipsoidal flattening, unitless
    f = 1/298.257223563

    # Horizon zenith angle, degrees (spherical earth)
    hza = 180 - np.arcsin( r_earth / ( r_earth + alt ) ) / pi180

    ## Ellipsoidal Earth
    # # Eccentricity of ellipsoid
    # ecc = f * (2-f)
    # # Local (i.e. prime vertical) radius of curvature at latitude
    # N = r_earth / np.sqrt( 1 - ecc**2 * np.sin(lat*pi180)**2 )
    # # Horizon zenith angle, degrees
    # hza = 180 - np.arcsin( N / (N+alt) ) / pi180

    return hza

def refraction_angle( true_elevation_angle, pressure=101325., temperature_celsius=10. ):
    '''Atmospheric refraction angle for light passing through Earth's atmosphere

    The apparent locations in the sky of objects outsides Earth's atmosphere 
    differs from their true locations due to atmospheric refraction. 
    (e.g. sun and moon when they rise and set)
    The apparent elevation of an object above the horizon is
    apparent elevation angle = (true elevation angle) + (refraction angle)
    
    The equations here are from Saemundsson/Bennett, whose calculations use
    a typical vertical profile of atmospheric density (i.e. temperature and pressure).
    The profiles can be rescaled to a particular surface temperature and pressure
    to approximately account for varying atmospheric conditions.
    Accurate refraction calculations should use fully specified vertical profile
    of temperature and pressure, which cannot be done here.

    Parameters
    ----------
    true_elevation_angle : float
        degrees above horizon of sun or other object
    pressure : float (default=101325)
        surface atmospheric pressure (Pa)
    temperature_celsius : float (default=10)
        surface atmospheric temperature (C)

    Returns
    -------
    angle : float
        refraction angle in degrees. Value is zero when apparent elevation is below horizon
    '''
    # Refraction angle, arcminutes
    R = 1.02 / np.tan( ( true_elevation_angle + 10.3 / (true_elevation_angle + 5.11) ) * pi180 )
    # Account for temperature and pressure, arcminutes
    R = R * pressure / 101325 * 283 / ( 273 + temperature_celsius )
    # Convert arcminutes -> degrees
    R /= 60

    # Result must be positive
    R = np.maximum(R,0)

    # Refraction defined only when the apparent elevation angle is positive
    # Set refraction to zero when the apparent elevation is below horizon
    refraction_angle = np.where( true_elevation_angle + R <= 0, 0, R)

    return refraction_angle

def _to_datetime(time_in):
    '''Convert to Timestamp, Series of datetime64, or DataArray of datetime64 
    
    Parameters
    ----------
    time_in : datetime-like or array
        time to be converted

    Returns
    -------
    time_out : pandas.Timestamp, pandas.Series of datetime64, DataArray of datetime64
    '''
    if hasattr(time_in,'dt'):
        time_out = time_in
    # elif isinstance(time_in, pd.DatetimeIndex ):
    #     # Unnecessary; DatetimeIndex will work fine in the else cases
    #     time_out = pd.Series(time_in)
    #     tz = time_out.dt.tz
    else:
        try:
            # Convert list of times
            time_out = pd.Series(pd.DatetimeIndex(time_in))
        except TypeError:
            # Single datetime or str
            time_out = pd.Timestamp(time_in)

    return time_out

def _to_datetime_utc( datetime_in ):
    '''Convert to Timestamp, Series of datetime64, or DataArray of datetime64 and in UTC
    
    Parameters
    ----------
    datetime_in : datetime-like
        date and time to be converted
        
    Returns
    -------
    datetimeUTC : Timestamp, Series, or DataArray
        date and time in UTC but tz-naive
    tz_in : datetime.timezone
        timezone of datetime_in
    '''

    # Ensure input is a timestamp
    datetime_in = _to_datetime( datetime_in )

    # Convert to UTC, then strip timezone
    try:
        if hasattr(datetime_in,'dt'):
            # Pandas Series objects
            tz_in = datetime_in.dt.tz
            datetimeUTC = datetime_in.dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            # Scalar time objects
            tz_in = datetime_in.tzinfo
            datetimeUTC = datetime_in.tz_convert('UTC').tz_localize(None)
    except (TypeError, AttributeError):
        # tz-naive time objects: Timestamp, Series, (all) DataArrays
        # No timezone info, so assume it is already UTC
        warnings.warn('Time does not contain timezone. UTC will be assumed',RuntimeWarning)
        datetimeUTC = datetime_in
        tz_in = None

    return datetimeUTC, tz_in

SolarPositionResults = namedtuple('SolarPositionResults',
                    'declination right_ascension equation_of_time distance datetimeUTC')
'''Namedtuple containing results of `solar_position`

All values are geocentric, strictly accurate for the center of the Earth, not a point on 
Earth's surface. The parallax angle from Earth's center to surface is 4e-5 degrees.

Attributes
----------
declination : float
    position of the sun relative to Earth's equatorial plane, degrees
right_ascension : float
    position of the sun along Earth's equatorial plane relative to the vernal equinox, degrees
equation_of_time : float
    equation of time (minutes) between mean solar time and true solar time.
    Divide by 4 minutes per degree to obtain equation of time in degrees.
distance : float
    Earth-sun distance in AU (1 AU = 1.495978707e11 m)
datetimeUTC : Timestamp, Series, or DataArray of numpy.datetime64
    date and time input for calculations, UTC
'''

SunTimesResults = namedtuple('SunTimesResults',
                    'sunrise sunset day_length solar_noon datetimeUTC')
'''Namedtuple containing results of `sun_times`

Attributes
----------
sunrise : pandas.DatetimeIndex
    sunrise time, UTC if not specified otherwise
sunset : pandas.DatetimeIndex
    sunset time, UTC if not specified otherwise
day_length : pandas.Timedelta
    duration of daylight
solar_noon : pandas.DatetimeIndex
    time of meridian transit, UTC if not specified otherwise
datetimeUTC : Timestamp, Series, or DataArray of numpy.datetime64
    date and time input for calculations, UTC
'''

pi180 = np.pi / 180
'''Constant $\pi/180$'''

# Aliases for functions
sza = solar_zenith_angle
'''Alias for `solar_zenith_angle`'''
saa = solar_azimuth_angle
'''Alias for `solar_azimuth_angle`'''
sea = solar_elevation_angle
'''Alias for `solar_elevation_angle`'''

# Deprecated aliases
equationOfTime = equation_of_time
'''Alias for `equation_of_time`'''
solarDeclination = solar_declination
'''Alias for `solar_declination`'''
