# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:33:52 2015

@author: cdholmes
"""
def sinceEpoch(date): 
    '''' Returns seconds since epoch 
    input should be a datetime object 
    '''
    import time
    return time.mktime(date.timetuple())
    
def date2yearf(date):
    ''' Convert datetime or array of datetime objects to year with fraction
    input can also be a date object
    '''

    import numpy as np
    import pandas as pd
    from datetime import datetime as dt
    

    yearfrac = []
    for d in np.array(date):
        # Convert numpy datetime64 to datetime
        d = pd.to_datetime(d)
        # get the year
        year = d.year
        startOfThisYear = dt(year=year, month=1, day=1)
        startOfNextYear = dt(year=year+1, month=1, day=1)
        yearElapsed = sinceEpoch(d) - sinceEpoch(startOfThisYear)
        yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
        fraction = yearElapsed/yearDuration
            
        yearfrac.append(d.year + fraction)

    # if result has only one value, then convert to scalar   
    if (len(yearfrac)==1):
        yearfrac = yearfrac[0]
        
    return yearfrac

def yearf2date(yearf,roundSec=True):
    ''' Convert fractional year to datetime or array of datetime objects '''

    from datetime import datetime as dt
    from datetime import timedelta as td
    import numpy as np    

    s = sinceEpoch    
    
    date = []
    for d in [yearf]:
        y = np.int(d)
        startOfThisYear = dt(year=y,month=1,day=1)
        startOfNextYear = dt(year=y+1, month=1, day=1)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        secElapsed = yearDuration * np.mod(d,1)
        if (roundSec):
            secElapsed = np.round(secElapsed)
        date.append(startOfThisYear + td(seconds=secElapsed) )

    # if result has only one value, then convert to scalar   
    if (len(date)==1):
        date = date[0]
        
    return date

def date2doyf(date):
    ''' Convert datetime or array of datetime objects to day of year with fraction
    input can also be a date object
    '''

    #import numpy as np
    from datetime import datetime as dt
    
    s = sinceEpoch

    doy = []
    for d in [date]:

        year = d.year
        month = d.month
        day = d.day
        
        startOfThisDay = dt(year=year,month=month,day=day)
        dayElapsed = s(d) - s(startOfThisDay)
        fraction = dayElapsed/86400.
        
        doy.append(d.timetuple().tm_yday + fraction)
            
     # if result has only one value, then convert to scalar   
    if (len(doy)==1):
        doy = doy[0]
        
    return doy

def doyf2date(doyf,year,roundSec=True):
    ''' Convert day of year and year to datetime or array of datetime objects '''

    import numpy as np
    from datetime import datetime as dt
    from datetime import timedelta as td

    if (len(year)==1):
        year = np.zeros_like(doyf)+year

    date = []
    for i,d in enumerate(doyf):
        startOfThisYear = dt(year=year[i],month=1,day=1)
        secElapsed = (d-1)*86400
        if (roundSec):
            secElapsed = np.round(secElapsed)
        d = td(seconds=secElapsed)
        date.append( startOfThisYear + td(seconds=secElapsed) )

    # if result has only one value, then convert to scalar   
    if (len(date)==1):
        date = date[0]
    
    return date





