#!/usr/local/bin/env python3

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

    # Ecliptic longitude of sun
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

def sza( lat, lon, datetimeUTC ):
    # Calculate solar zenith angle for a given latitude, longitude, date and time.
    # Accounts for equation of time, but not for elevation or atmospheric refraction,
    # which are important when the sun is near the horizon.
    # Results are accurate to tenths of a degree, except near the horizon (< 20 degrees solar elevation)
    #
    # Inputs:
    # lat  : scalar or array of latitudes, degrees
    # lon  : scalar or array of longitudes, degrees
    # datetimeUTC : date and time in UTC, can be a pandas.Timestamp or datetime object
    #
    # C.D. Holmes - 9 Nov 2018 - Initial version
    
    # Convert to Timestamp, if necessary
    if (type(datetimeUTC) is datetime.datetime ):
        datetimeUTC = pd.Timestamp( datetimeUTC )

    # Calculate declination and equation of time, degrees 
    dec = solarDeclination( datetimeUTC )    
    eot = equationOfTime(   datetimeUTC )

    # Hour angle, degrees
    # This is for mean solar time. Actual solar position has a small offset given by the equation of time (below)
    Ha = ( ( datetimeUTC.hour + datetimeUTC.minute / 60 - 12 ) * 15 + lon ) 

    # Add equation of time to the hour angle
    Ha += eot

    # Solar zenith angle, radians
    sza = np.arccos( np.sin(lat*pi180) * np.sin(dec*pi180) + np.cos(lat*pi180) * np.cos(dec*pi180) * np.cos(Ha*pi180) )

    # Convert radians -> degrees
    sza /= pi180

    return sza 




