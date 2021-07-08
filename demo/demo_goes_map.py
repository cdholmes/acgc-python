#!/usr/bin/env python3
'''Show GOES geostationary imagery unprojected and projected

see demo_goes_truecolor.py for more examples

Instructions from
https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html
'''

'''
GOES Filename interpretation
OR_ABI-L2-MCMIPC-M3_G16_s20181781922189_e20181781924562_c20181781925075.nc
OR - Indicates the system is operational
ABI - Instrument type
L2 - Level 2 Data
MCMIP - Multichannel Cloud and Moisture Imagery products
c - CONUS file (created every 5 minutes).
M3 - Scan mode
G16 - GOES-16
sYYYYJJJHHMMSSZ - Scan start: 4 digit year, 3 digit day of year (Julian day), hour, minute, second, tenth second
eYYYYJJJHHMMSSZ - Scan end
cYYYYJJJHHMMSSZ - File Creation .nc - NetCDF file extension
'''
#%%
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy
import xarray as xr

# Open the file with xarray.
# The opened file is assigned to "C" for the CONUS domain.

FILE = ('http://ramadda-jetstream.unidata.ucar.edu/repository/opendap'
        '/4ef52e10-a7da-4405-bff4-e48f68bb6ba2/entry.das#fillmismatch')

#FILE='/Users/cdholmes/Documents/ResearchProjects/GOES-TrueColor/OR_ABI-L2-MCMIPC-M6_G17_s20192332001196_e20192332003575_c20192332004099.nc'

C = xr.open_dataset(FILE)

# Scan's start time, converted to datetime object
scan_start = datetime.strptime(C.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')

# Scan's end time, converted to datetime object
scan_end = datetime.strptime(C.time_coverage_end, '%Y-%m-%dT%H:%M:%S.%fZ')

# File creation time, convert to datetime object
file_created = datetime.strptime(C.date_created, '%Y-%m-%dT%H:%M:%S.%fZ')

# The 't' variable is the scan's midpoint time
midpoint = str(C['t'].data)[:-8]
scan_mid = datetime.strptime(midpoint, '%Y-%m-%dT%H:%M:%S.%f')

print('Scan Start    : {}'.format(scan_start))
print('Scan midpoint : {}'.format(scan_mid))
print('Scan End      : {}'.format(scan_end))
print('File Created  : {}'.format(file_created))
print('Scan Duration : {:.2f} minutes'.format((scan_end-scan_start).seconds/60))

# We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
dat = C.metpy.parse_cf('CMI_C02')

# GOES projection
goesproj = dat.metpy.cartopy_crs

# We also need the x (north/south) and y (east/west) axis sweep of the ABI data
x = dat.x
y = dat.y

# Red channel is band 2
R   = C['CMI_C02'].data

fig = plt.figure(figsize=(15, 12))

# Create axis with Geostationary projection
ax1 = fig.add_subplot(1, 2, 1, projection=goesproj)

# Add the RGB image to the figure. The data is in the same projection as the
# axis we just created.
ax1.imshow(R, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()), transform=goesproj)

# Add Coastlines and States
ax1.coastlines(resolution='10m', color='black', linewidth=0.25)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.25)

ax1.set_title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
ax1.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')


# Generate an Cartopy projection
lc = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5,
                                                                        38.5))

ax2 = fig.add_subplot(1, 2, 2, projection=lc)
ax2.set_extent([-130, -65, 15, 55], crs=ccrs.PlateCarree())

ax2.imshow(R, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=goesproj,
          interpolation='none')
ax2.coastlines(resolution='10m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)

ax2.set_title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
ax2.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')


plt.show()
