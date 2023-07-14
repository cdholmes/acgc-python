#!/usr/local/bin/env python3
''' Module to calculate solar zenith angle, solar declination, and equation of time 
Results should be accurate to < 0.1 degree, but other modules should be used for 
high-precision calculations.

C.D. Holmes 9 Nov 2018
'''

import numpy as np
import pandas as pd
import datetime

pi180 = np.pi / 180

def solarDeclination( date ):
    # Calculate solar declination (degrees) based on date
    # Implements Eq. 9.68-9.72 from M.Z. Jacobson, Fundamentals of Atmospheric Modeling
    #
    # Inputs:
    # date : a pandas.Timestamp object
    #
    # C.D. Holmes - 9 Nov 2018 - Initial version

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

def equationOfTime( date ):
    # Calculate equation of time (degrees) based on date
    # Implements the "alternative equation" from Wikipedia, derived from
    # https://web.archive.org/web/20120323231813/http://www.green-life-innovators.org/tiki-index.php?page=The%2BLatitude%2Band%2BLongitude%2Bof%2Bthe%2BSun%2Bby%2BDavid%2BWilliams
    # Results checked against NOAA solar calculator and agree within 10 seconds.
    #
    # Inputs:
    # date : a pandas.Timestamp object
    #
    # C.D. Holmes - 9 Nov 2018 - Initial version

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

def hour_angle( lon, datetimeUTC ):
    # Compute hour angle, degrees
    #
    # Hour angle is the angular displacement of the sun from the local meridian.
    # It is zero at local noon, negative in the morning, and positive is afternoon.

    # This is for mean solar time. Actual solar position has a small offset given by the equation of time (below)
    Ha = ( ( datetimeUTC.hour + datetimeUTC.minute / 60 - 12 ) * 15 + lon ) 

    # Add equation of time to the hour angle, degrees
    Ha += equationOfTime( datetimeUTC )

    return Ha
    
def refraction_angle( true_elevation_angle, pressure=101325., temperature_celsius=10. ):
    '''Atmospheric refraction angle
    Equation from Saemundsson/Bennett

    Inputs:
    true_elevation_angle : degrees above horizon
    pressure            : surface atmospheric pressure (Pa)
    temperature_celsius : surface atmospheric temperature (C)

    Output
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


def sza( lat, lon, datetimeUTC, refraction=False, temperature=10., pressure=101325. ):
    # Calculate solar zenith angle for a given latitude, longitude, date and time.
    # Accounts for equation of time, but not for elevation or atmospheric refraction,
    # which are important when the sun is near the horizon.
    # Results are accurate to tenths of a degree, except near the horizon (< 20 degrees solar elevation)
    #
    # Inputs:
    # lat  : scalar or array of latitudes, degrees
    # lon  : scalar or array of longitudes, degrees
    # datetimeUTC : date and time in UTC, can be a pandas.Timestamp or datetime object
    # refraction  : boolean specifying whether to account for atmospheric refraction
    # temperature : surface atmospheric temperature (Celsius), only used for refraction calculation
    # pressure    : surface atmospheric pressure (Pa), only used for refraction calculation
    #  
    #
    # C.D. Holmes - 9 Nov 2018 - Initial version
    
    # Convert to Timestamp, if necessary
    if (type(datetimeUTC) is datetime.datetime ):
        datetimeUTC = pd.Timestamp( datetimeUTC )

    # Solar declination, degrees 
    dec = solarDeclination( datetimeUTC )    

    # Hour angle, degrees
    Ha = hour_angle( lon, datetimeUTC )

    # True solar zenith angle, radians
    sza = np.arccos( np.sin(lat*pi180) * np.sin(dec*pi180) + \
          np.cos(lat*pi180) * np.cos(dec*pi180) * np.cos(Ha*pi180) )

    # Convert radians -> degrees
    sza /= pi180

    if refraction: 
        # Subtract refraction angle (degrees) from zenith angle (SZA is always smaller due to refraction)
        sza -= refraction_angle( 90-sza, pressure, temperature ) 

    return sza

def saa( lat, lon, datetimeUTC ):
    '''Solar azimuth angle (degrees) for a latitude, longitude, date and time
    
    SAA is degrees clockwise from north.
    Inputs:
    lat  : scalar or array of latitudes, degrees
    lon  : scalar or array of longitudes, degrees
    datetimeUTC : date and time in UTC, can be a pandas.Timestamp or datetime object

    C.D. Holmes - 13 Jan 2023 - Initial version
    '''

    # Solar declination, degrees 
    dec = solarDeclination( datetimeUTC )   

    # Hour angle, degrees
    Ha = hour_angle( lon, datetimeUTC )

    # Solar zenith angle, degrees
    # Use true sza, without refraction
    zen = sza( lat, lon, datetimeUTC, refraction=False )

    # Solar azimuth angle, degrees
    saa = np.arcsin( -np.sin( Ha*pi180 ) * np.cos( dec*pi180 ) / 
            np.sin( zen*pi180 ) ) / pi180

    # Change range [-180,180] to [0,360]
    return np.mod( saa+360, 360 )


