# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:48:05 2015

@author: cdholmes
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import numpy as np

#########################################################
# Create artificial gridded data

# Number of points in x and y directions
n = 20

# Edges of the latitude-longitude grid
latedge  = np.linspace(-90,90,n+1)
lonedge  = np.linspace(-180,180,2*n+1)
latedge[0] = -89.9999 # Offset south pole to avoid singularity in LambertConformal projection
lon2de, lat2de = np.meshgrid( lonedge, latedge )

# Centers of the latitude-longitude grid
lat      = np.convolve(latedge, [0.5,0.5], 'valid' )
lon      = np.convolve(lonedge, [0.5,0.5], 'valid' )
lon2d, lat2d = np.meshgrid( lon, lat )

# Artificial data
data = np.exp( -lat2d**2/90**2 -lon2d**2/120**2 )

#########################################################
# Create artificial point data

# number of points
npoint = 50

# random latitude longitude and value
plat  = np.random.rand(npoint) * 25 + 25
plon  = np.random.rand(npoint) * 50 - 120
pdata = np.random.rand(npoint) 

##########################################################
# Set up map and plot data

# Open a figure
plt.figure(1)
plt.clf()

# data coordinates: Use Geodetic (or PlateCarree) for data in lat-lon coordinates
datacoord = ccrs.Geodetic()
datacoord = ccrs.PlateCarree()

# Map projection for Global 
ax = plt.axes(projection=ccrs.PlateCarree())
ax = plt.axes(projection=ccrs.Robinson())
ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine())

# Map projection for North Pole
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())

# Map projection for CONUS 
ax = plt.axes(projection=ccrs.LambertConformal())
#ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96,central_latitude=39,standard_parallels=(33,45)))
# Set lat-lon of map edges. These are reasonable for CONUS
ax.set_extent([-120, -73, 25, 50], ccrs.PlateCarree())

# Add your gridded data
plt.pcolormesh(lonedge, latedge, data, transform=datacoord)

# Add your point data
plt.scatter(plon, plat, marker='o', c=pdata, s=50, transform=datacoord)

# Add a colorbar 
plt.colorbar()

# Add coasts and states; Use ‘110m’ or ‘50m’ resolution
ax.coastlines(resolution='110m')
ax.add_feature(cfeature.STATES.with_scale('110m'))

plt.show()
