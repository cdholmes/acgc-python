# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:48:05 2015

@author: cdholmes
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import numpy as np

# Create artificial data
n = 20
lat  = np.linspace(-90,90,n+1)
lon  = np.linspace(-180,180,2*n+1)
lon2d, lat2d = np.meshgrid( lon,lat)
data = np.exp( -lat2d**2/90**2 -lon2d**2/120**2 )

# Open a figure
plt.figure(1)
plt.clf()

# data coordinates: Use PlateCarree for data in lat-lon coordinates
datacoord = ccrs.PlateCarree()

# Map projection for Global 
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax = plt.axes(projection=ccrs.Robinson())

# Map projection for North Pole
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())

# Map projection for CONUS 
ax = plt.axes(projection=ccrs.LambertConformal())
# Set lat-lon of map edges. These are reasonable for CONUS
ax.set_extent([-120, -73, 25, 50], ccrs.PlateCarree())



# Add your data
plt.pcolormesh(lon, lat, data, transform=datacoord)

# Add a colorbar 
plt.colorbar()

# Add coasts and states; Use ‘110m’ or ‘50m’ resolution
ax.coastlines(resolution='110m')
ax.add_feature(cfeature.STATES.with_scale('110m'))



