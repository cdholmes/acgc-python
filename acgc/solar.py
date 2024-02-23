#!/usr/local/bin/env python3
'''Module for calculating solar position (zenith angle, elevation, azimuth)

The functions here are are vectorized and generally broadcast over xarray dimensions,
making this program faster than PySolar. Calculations assume spherical Earth (not ellipsoidal). 
Results should be accurate to < 0.2 degree, which is less than radius of the sun.
Other modules (e.g. pvlib) should be used for high-precision calculations.

'''

import warnings
import numpy as np
import pandas as pd

pi180 = np.pi / 180

def solar_azimuth_angle( lat, lon, datetimeUTC ):
    '''Solar azimuth angle (degrees) for a latitude, longitude, date and time
    
    SAA is degrees clockwise from north.
    
    Parameters
    ----------
    lat : float or ndarray
        latitude in degrees
    lon : float or ndarray
        longitudes in degrees
    datetimeUTC : datetime-like or str
        date and time in UTC

    Returns
    -------
    saa : float or ndarray
        solar azimuth angle in degrees (clockwise from north)
    '''
    # Convert to pandas Timestamp, if needed
    datetimeUTC = _to_timestamp(datetimeUTC)

    # Subsolar point, latitude longitude, degrees
    solar_lat = solar_latitude( datetimeUTC )
    solar_lon = solar_longitude( datetimeUTC )

    # Vector pointing toward sun
    x = np.cos( solar_lat * pi180 ) * np.sin( (solar_lon - lon) * pi180 )
    y = np.cos( lat*pi180 ) * np.sin( solar_lat*pi180 ) \
        - np.sin( lat*pi180 ) * np.cos( solar_lat*pi180 ) \
            * np.cos( (solar_lon - lon) * pi180 )

    # Azimuth angle from north, degrees
    saa = np.arctan2( x, y ) / pi180

    # Change range [-180,180] to [0,360]
    return np.mod( saa+360, 360 )

def solar_elevation_angle( lat, lon, alt, datetimeUTC,
                       refraction=False, temperature=10., pressure=101325. ):
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
    alt : float or ndarray
        altitude above surrounding terrain that defines the horizon, meters
    datetimeUTC : datetime-like or str
        date and time in UTC
    refraction : bool, optional (default=False)
        specifies whether to account for atmospheric refraction
    temperature : float or ndarray, optional (default=10)
        surface atmospheric temperature (Celsius), only used for refraction calculation
    pressure : float or ndarray, optional (default=101325)
        surface atmospheric pressure (Pa), only used for refraction calculation
    
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
         - solar_zenith_angle( lat, lon, datetimeUTC, refraction, temperature, pressure )

    return sea

def solar_zenith_angle( lat, lon, datetimeUTC, 
                       refraction=False, temperature=10., pressure=101325. ):
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
    datetimeUTC : datetime-like or str
        date and time in UTC
    refraction : bool, optional (default=False)
        specifies whether to account for atmospheric refraction
    temperature : float or ndarray, optional (default=10)
        surface atmospheric temperature (Celsius), only used for refraction calculation
    pressure : float or ndarray, optional (default=101325)
        surface atmospheric pressure (Pa), only used for refraction calculation
    
    Returns
    -------
    sza : float or ndarray
        solar zenith angle in degrees at the designated locations and times
        If refraction=False, this is the true solar zenith angle
        If refraction=True, this is the apparent solar zenith angle
    '''
    # Convert to pandas Timestamp, if needed
    datetimeUTC = _to_timestamp(datetimeUTC)

    # Solar declination, degrees
    dec = solar_declination( datetimeUTC )

    # Hour angle, degrees
    Ha = solar_hour_angle( lon, datetimeUTC )

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

def solar_declination( date ):
    '''Calculate solar declination (degrees) for specified date
    
    Implements Eq. 9.68-9.72 from M.Z. Jacobson, Fundamentals of Atmospheric Modeling
    
    Parameters
    ----------
    date : datetime-like or str
        date for calculation

    Returns
    -------
    dec : float
        solar declination in degrees at the specified date
    '''
    # Convert to pandas Timestamp, if needed
    date = _to_timestamp(date)

     # Number of days since beginning of 2000
    NJD = date - np.datetime64('2000-01-01')
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

def solar_latitude( datetimeUTC ):
    '''Latitude of the subsolar point
    
    Parameters
    ----------
    datetimeUTC : datetime-like or str
        date and time, must be UTC
    
    Returns
    -------
    latitude : float
        degrees of latitude
    '''
    return solar_declination( datetimeUTC )

def solar_longitude( datetimeUTC ):
    '''Longitude of the subsolar point, degrees
    
    Parameters
    ----------
    datetimeUTC : datetime-like or str
        date and time, must be UTC
    
    Returns
    -------
    longitude : float
        degrees of longitude
    '''
    # Convert to pandas Timestamp, if needed
    datetimeUTC = _to_timestamp(datetimeUTC)

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
    solar_lon -= equation_of_time( datetimeUTC, degrees=True )

    return solar_lon

def solar_hour_angle( lon, datetimeUTC ):
    '''Solar hour angle (degrees) for specified longitude, date and time

    Hour angle is the angular displacement of the sun from the local meridian.
    It is zero at local noon, negative in the morning, and positive is afternoon.
    
    Parameters
    ----------
    lon : float
        longitude in degrees east
    datetimeUTC : datetime-like or str
        date and time, must be UTC
    
    Returns
    -------
    ha : float
        hour angle in degrees at the specified location and time
    '''

    # Subsolar longitude, degrees
    solar_lon = solar_longitude(datetimeUTC)

    # Hour angle, degrees
    Ha = lon - solar_lon

    return Ha

def equation_of_time( date, degrees=False ):
    '''Equation of time for specified date
    
    Implements the "alternative equation" from Wikipedia, derived from
    https://web.archive.org/web/20120323231813/http://www.green-life-innovators.org/tiki-index.php?page=The%2BLatitude%2Band%2BLongitude%2Bof%2Bthe%2BSun%2Bby%2BDavid%2BWilliams
    Results checked against NOAA solar calculator and agree within 10 seconds.
    
    Note: Leap years are not accounted for.

    Parameters
    ----------
    date : datetime-like or str
        date UTC
    degrees : bool (default=False)
        If True, then return value in compass degrees
        If False, then return value in minutes of an hour
        
    Returns
    -------
    eot : float
        equation of time on the specified date, degrees or minutes
    '''
    # Convert to pandas Timestamp, if needed
    date = _to_timestamp(date)

    # Equation of time, accounts for the solar day differing slightly from 24 hr
    try:
        doy = date.dt.dayofyear
    except AttributeError:
        doy = date.dayofyear
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

def _to_timestamp(time_in):
    '''Convert input to Pandas Timestamp or Series of datetime64 
    
    Parameters
    ----------
    time_in : datetime-like or array
        time to be converted

    Returns
    -------
    time_out : pandas.Timestamp or pandas.Series of datetime64
    '''
    if hasattr(time_in,'dt'):
        time_out = time_in
    elif isinstance(time_in, pd.DatetimeIndex ):
        time_out = pd.Series(time_in)
    else:
        try:
            # Convert list of times
            time_out = pd.Series(pd.DatetimeIndex(time_in))
        except TypeError:
            # Single datetime or str
            time_out = pd.Timestamp(time_in)

    return time_out

# Aliases for functions
sza = solar_zenith_angle
saa = solar_azimuth_angle
sea = solar_elevation_angle
# Additional aliases for backwards compatibility
equationOfTime = equation_of_time
solarDeclination = solar_declination