# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:43:20 2015

@author: cdholmes
"""

#conda install cartopy

import cartopy.crs as ccrs
import cartopy.feature as cfeature 

import matplotlib.pyplot as plt
import numpy as np

# Create artificial point data
# number of points
npoint = 50
# random latitude, longitude and value
plat  = np.random.rand(npoint) * 180 - 90
plon  = np.random.rand(npoint) * 360 - 180
pdata = np.random.rand(npoint) 

# Clear the figure
plt.figure(1)
plt.clf()

# data coordinates: Use PlateCarree for data in lat-lon coordinates
datacoord = ccrs.PlateCarree()

# Projection
projcoord = ccrs.PlateCarree()
projcoord = ccrs.Robinson()

#Setup map projection and continent outlines
ax = plt.axes(projection=projcoord)

# Change map extent, if needed
#ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())

# Add data points
plt.scatter( plon, plat, c=pdata, s=50, marker='o', transform=datacoord )

# draw parallels and meridians, but don't bother labelling them.
ax.gridlines(draw_labels=False)

# Add coasts and states; Use ‘110m’ or ‘50m’ resolution
ax.coastlines(resolution='110m')

# Fill the continents
ax.add_feature(cfeature.LAND, facecolor='0.75')

# add colorbar
plt.colorbar()#location='bottom')
plt.clim(0,1)

# Optional: Convert the points into the projection coordinate system
xyz = projcoord.transform_points( datacoord, plon, plat )
# Optional: Plot the points using projection coordinates; Note: Now we don't need "transform"
x=xyz[:,0]; y=xyz[:,1]
plt.scatter( x, y, s=50, marker='o', facecolor='none', edgecolor='black' )

plt.title('Artificial data')
#plt.show()
