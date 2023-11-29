# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:12:41 2015

@author: cdholmes
"""

import datetime as dt
import numpy as np
import pandas
import matplotlib.pyplot as plt
from acgc.stats import tapply


file = 'c2h6_brw_surface-flask_1_arl_event.txt'

# Both read_fwf and read_csv can read fixed-width files in many cases
#data = pandas.read_fwf('c2h6_brw_surface-flask_1_arl_event.txt', comment='#', delimiter=' ',na_values='-999.990',
#                       names=['siteID','year','month','day','hour','minute','second','sampleID','method','chemformula','labname','value','uncertainty','flag','instrument','labyear','labmonth','labday','labhour','labminute','labsecond','latitude','longitude','altitude','eventnum'])

data = pandas.read_csv('c2h6_brw_surface-flask_1_arl_event.txt', comment='#', delim_whitespace=True, na_values='-999.990',
                       names=['siteID','year','month','day','hour','minute','second','sampleID','method','chemformula','labname','value','uncertainty','flag','instrument','labyear','labmonth','labday','labhour','labminute','labsecond','latitude','longitude','altitude','eventnum'])

# construct a unique number for each sample collection time
# consider all samples collected within an hour to be replicates
timeid = data['year']*1e6 + data['month']*1e4 + data['day']*1e2 + data['hour']
timeid = np.int64(timeid.values)

# calculate the mean and standard deviation of replicate samples, 
# which have the same timeid
# This should be rewritten to use pandas.groupby
valuemean, timeiduniq = tapply(data['value'].values,timeid,np.nanmean)
valuestd, _ = tapply(data['value'].values,timeid,np.nanstd)

# array to store time
time = np.zeros_like(valuemean)

# loop over times
for i in range(len(timeiduniq)):
    
    # time structure for i-th observation    
    tt = dt.datetime.strptime(str(timeiduniq[i]),'%Y%m%d%H')

    # day of year for i-th observations (starts at 0 on Jan 1)    
    doy = tt.timetuple().tm_yday-1 
    
    # number of days in year
    doy1 = dt.datetime(tt.year,12,31).timetuple().tm_yday    

    # time of i-th observation, as a year and fraction    
    time[i] = tt.year + np.float(doy)/doy1 + tt.hour/24/doy1
    print(i,timeiduniq[i],tt,time[i])

plt.clf()
#plt.plot(data['value']) # plot raw data, replicates included

# plot time series of observations
plt.subplot(2,1,1)
plt.plot(time,valuemean)
plt.xlabel('time')
plt.ylabel('C2H6, ppt')
plt.title('Barrow, Alaska')

# plot time series of observations errorbars
plt.subplot(2,1,2)
plt.errorbar(time,valuemean,yerr=valuestd)
plt.xlabel('time')
plt.ylabel('C2H6, ppt')
plt.title('Barrow, Alaska')
