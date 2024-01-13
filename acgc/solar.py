#!/usr/local/bin/env python3
''' Module to calculate solar zenith angle, solar declination, and equation of time 
Results should be accurate to < 0.1 degree, but other modules should be used for 
high-precision calculations.

C.D. Holmes 9 Nov 2018
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
    datetimeUTC : pandas.Timestamp, datetime, or str
        date and time in UTC

    Returns
    -------
    saa : float or ndarray
        solar azimuth angle in degrees (clockwise from north)
    '''
    # Convert to pandas Timestamp, if needed
    datetimeUTC = _to_timestamp(datetimeUTC)

    # Solar declination, degrees
    dec = solar_declination( datetimeUTC )

    # Hour angle, degrees
    Ha = solar_hour_angle( lon, datetimeUTC )

    # Solar zenith angle, degrees
    # Use true sza, without refraction
    zen = sza( lat, lon, datetimeUTC, refraction=False )

    # Solar azimuth angle, degrees
    saa = np.arcsin( -np.sin( Ha*pi180 ) * np.cos( dec*pi180 ) /
            np.sin( zen*pi180 ) ) / pi180

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
    datetimeUTC : pandas.Timestamp, datetime, or str
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
    datetimeUTC : pandas.Timestamp, datetime, or str
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

def equation_of_time( date ):
    '''Equation of time (degrees) for specified date
    
    Implements the "alternative equation" from Wikipedia, derived from
    https://web.archive.org/web/20120323231813/http://www.green-life-innovators.org/tiki-index.php?page=The%2BLatitude%2Band%2BLongitude%2Bof%2Bthe%2BSun%2Bby%2BDavid%2BWilliams
    Results checked against NOAA solar calculator and agree within 10 seconds.
    
    Argument
    --------
    date : pandas.Timestamp, date, or datetime
        date for calculation

    Returns
    -------
    eot : float
        equation of time in degrees on the specified date
    '''
    # Convert to pandas Timestamp, if needed
    date = _to_timestamp(date)

    # Equation of time, accounts for the solar day differing slightly from 24 hr
    doy = date.dayofyear
    W = 360 / 365.24
    A = W * (doy+10)
    B = A + 1.914 * np.sin( W * (doy-2) * pi180 )
    C = ( A - np.arctan2( np.tan(B*pi180), np.cos(23.44*pi180) ) / pi180 ) / 180

    # Equation of time in minutes of an hour
    eotmin = 720 * ( C - np.round(C) )

    # Equation of time, minutes -> degrees
    eot = eotmin / 60 * 360 / 24

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

def solar_declination( date ):
    '''Calculate solar declination (degrees) for specified date
    
    Implements Eq. 9.68-9.72 from M.Z. Jacobson, Fundamentals of Atmospheric Modeling
    
    Argument
    --------
    date : pandas.Timestamp, date, datetime, or str
        date for calculation

    Returns
    -------
    dec : float
        solar declination in degrees at the specified date
    '''

    # Convert to pandas Timestamp, if needed
    date = _to_timestamp(date)

     # Number of days since beginning of 2000
    NJD = np.floor( date.to_julian_date() - pd.Timestamp(2000,1,1).to_julian_date() )

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

def solar_hour_angle( lon, datetimeUTC ):
    '''Solar hour angle (degrees) for specified longitude, date and time

    Hour angle is the angular displacement of the sun from the local meridian.
    It is zero at local noon, negative in the morning, and positive is afternoon.
    
    Parameters
    ----------
    lon : float
        longitude in degrees east
    datetimeUTC : pandas.Timestamp or datetime
        date and time for calculation, must be UTC
    
    Returns
    -------
    ha : float
        hour angle in degrees at the specified location and time
    '''

    # Convert to pandas Timestamp, if needed
    datetimeUTC = _to_timestamp(datetimeUTC)

    # Hour angle for mean solar time.
    # Actual solar position has a small offset given by the equation of time (below)
    Ha = ( datetimeUTC.hour + datetimeUTC.minute / 60 - 12 ) * 15 + lon

    # Add equation of time to the hour angle, degrees
    Ha += equation_of_time( datetimeUTC )

    return Ha

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
    '''Convert input to Pandas Timestamp or DatetimeIndex
    
    Arguments
    ---------
    time_in : datetime-like or array
        time to be converted

    Returns
    -------
    time_out : pandas.DatetimeIndex or pandas.Timestamp
        result will be a DatetimeIndex, if possible, and Timestamp otherwise
    '''
    if not isinstance(time_in, (pd.Timestamp,pd.DatetimeIndex) ):
        try:
            time_out = pd.DatetimeIndex(time_in)
        except TypeError:
            time_in = pd.Timestamp(time_in)
    else:
        time_out = time_in
    return time_out

# Aliases for functions
sza = solar_zenith_angle
saa = solar_azimuth_angle
sea = solar_elevation_angle
# Additional aliases for backwards compatibility
equationOfTime = equation_of_time
solarDeclination = solar_declination