#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of functions for manipulating times and dates

This module is outdated. These functions can be accomplished with pandas.Timestamp.

@author: cdholmes
"""

import time
from datetime import datetime as dt
from datetime import timedelta as td

import numpy as np
import pandas as pd


def _seconds_since_epoch(datetime):
    ''''Compute seconds since start of Unix epoch
    
    Epoch begins January 1, 1970, 00:00:00 (UTC)
    
    Parameter
    ---------
    datetime : datetime.datetime
        date and time
    
    Returns
    -------
    sec : float
        seconds since
    '''
    return time.mktime(datetime.timetuple())
_sinceEpoch = _seconds_since_epoch

def date2yearf(datetime):
    '''Convert datetime to year with fraction

    Parameters
    ----------
    datetime : datetime.date or datetime.datetime or list/array of same
        date and time
    
    Returns
    -------
    yearfrac : float or list
        year and fraction corresponding to input dates. e.g. 2005.156
    '''

    yearfrac = []
    for d in np.array(datetime):
        # Convert numpy datetime64 to datetime
        d = pd.to_datetime(d)
        # get the year
        year = d.year
        startOfThisYear = dt(year=year, month=1, day=1)
        startOfNextYear = dt(year=year+1, month=1, day=1)
        yearElapsed = _sinceEpoch(d) - _sinceEpoch(startOfThisYear)
        yearDuration = _sinceEpoch(startOfNextYear) - _sinceEpoch(startOfThisYear)
        fraction = yearElapsed/yearDuration

        yearfrac.append(d.year + fraction)

    # if result has only one value, then convert to scalar
    if len(yearfrac)==1:
        yearfrac = yearfrac[0]

    return yearfrac

def yearf2date(yearf,roundSec=True):
    '''Convert year with fraction to datetime 

    Parameters
    ----------
    yearfrac : float or list of float
        year and fraction, e.g. 2005.156
    roundSec : bool (default=True)
        specify if output should be rounded to the nearest second
        
    Returns
    -------
    date : datetime.datetime or list of same
        date and time corresponding to input
    '''

    s = _sinceEpoch

    date = []
    for d in [yearf]:
        y = np.int(d)
        startOfThisYear = dt(year=y,month=1,day=1)
        startOfNextYear = dt(year=y+1, month=1, day=1)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        secElapsed = yearDuration * np.mod(d,1)

        if roundSec:
            secElapsed = np.round(secElapsed)
        date.append(startOfThisYear + td(seconds=secElapsed) )

    # if result has only one value, then convert to scalar
    if len(date)==1:
        date = date[0]

    return date

def date2doyf(datetime):

    '''Convert datetime to day of year with fraction

    Parameters
    ----------
    datetime : datetime.date or datetime.datetime or list of same
        date and time
    
    Returns
    -------
    doyfrac : float or list
        day of year and fraction corresponding to input dates. e.g. 100.156
    '''

    s = _sinceEpoch

    doy = []
    for d in [datetime]:

        year = d.year
        month = d.month
        day = d.day

        startOfThisDay = dt(year=year,month=month,day=day)
        dayElapsed = s(d) - s(startOfThisDay)
        fraction = dayElapsed/86400.

        doy.append(d.timetuple().tm_yday + fraction)

     # if result has only one value, then convert to scalar
    if len(doy)==1:
        doy = doy[0]

    return doy

def doyf2date(doyf,year,roundSec=True):
    '''Convert day of year with fraction to datetime 

    Parameters
    ----------
    doyf : float or list of float
        day of year and fraction, e.g. 100.156
    year : float or list of float
        year
    roundSec : bool (default=True)
        specify if output should be rounded to the nearest second
        
    Returns
    -------
    date : datetime.datetime or list of same
        date and time corresponding to input
    '''

    if len(year)==1:
        year = np.zeros_like(doyf)+year

    date = []
    for i,d in enumerate(doyf):
        startOfThisYear = dt(year=year[i],month=1,day=1)
        secElapsed = (d-1)*86400

        if roundSec:
            secElapsed = np.round(secElapsed)
        d = td(seconds=secElapsed)
        date.append( startOfThisYear + td(seconds=secElapsed) )

    # if result has only one value, then convert to scalar
    if len(date)==1:
        date = date[0]

    return date
