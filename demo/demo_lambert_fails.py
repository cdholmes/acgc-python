#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:01:36 2018

@author: cdholmes
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Artificial data with n latitudes, m longitudes, starting at latitude lat1 and ending at lat2 
def create_data(n,m,lat1,lat2):
    # edges of the latitude-longitude grid
    latedge = np.linspace(lat1,lat2,n+1)
    lonedge = np.linspace(-180,180,m+1)
    
    # Centers of the latitude-longitude grid
    lat      = np.convolve(latedge, [0.5,0.5], 'valid' )
    lon      = np.convolve(lonedge, [0.5,0.5], 'valid' )
    lon2d, lat2d = np.meshgrid( lon, lat )

    # Artificial data
    data = np.exp( -lat2d**2/90**2 -lon2d**2/120**2 )

    return latedge, lonedge, data    


##########################################################
# Set up map and plot data

# Open a figure
plt.figure(1)
plt.clf()

# number of points
n = 20
m = 40

# data coordinates: Use PlateCarree for data in lat-lon coordinates
datacoord = ccrs.PlateCarree()

##########################################################
# Latitude range 90S-90N; This FAILS

# Data from 90S to 90N
latedge, lonedge, data = create_data(n,m,-90,90)

# Map projection 
ax = plt.subplot(1,3,1, projection=ccrs.LambertConformal())
ax.coastlines()

# Add data
plt.pcolormesh(lonedge,latedge,data,transform=datacoord)

plt.title('LambertConformal FAILS')

##########################################################
# Same data, different projection


# Map projection 
ax = plt.subplot(1,3,2, projection=ccrs.AlbersEqualArea())
ax.coastlines()

# Add data
plt.pcolormesh(lonedge,latedge,data,transform=datacoord)

plt.title('Albers WORKS (same data)')

##########################################################
# Latitude range 89.999S-90N; This WORKS

# Data from 90S to 90N
latedge, lonedge, data = create_data(n,m,-89.999,90)

# Map projection 
ax = plt.subplot(1,3,3, projection=ccrs.LambertConformal())
ax.coastlines()

plt.pcolormesh(lonedge,latedge,data,transform=datacoord)

plt.title('LambertConformal WORKS (limit 89.999S)')

plt.show()

#plt.savefig('test.png')