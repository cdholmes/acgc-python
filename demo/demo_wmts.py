#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example using Web Map Tile Service to access and plot Earth imagery from NASA GIBS

Created on Thu Nov  7 17:13:47 2019

@author: cdholmes
"""

import matplotlib.pyplot as plt
from owslib.wmts import WebMapTileService
import cartopy.crs as ccrs

# NASA GIBS currently seems to return a faulty projection code (2022-11-19)
# These lines enable OWS to understand the faulty code
import cartopy.io.ogc_clients as ogcc
ogcc._URN_TO_CRS['urn:ogc:def:crs:EPSG:6.18:3:3857'] = ccrs.GOOGLE_MERCATOR
ogcc.METERS_PER_UNIT['urn:ogc:def:crs:EPSG:6.18:3:3857'] = 1

# Location of online data
URL = 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/wmts.cgi'
wmts = WebMapTileService(URL)

# Select desired dataset and date
# Additional datasets are listed here: https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+Available+Imagery+Products
layer1 = 'BlueMarble_NextGeneration'
layer2 = 'MODIS_Aqua_CorrectedReflectance_TrueColor'
date   = '2015-07-01'


# Map projection
# Additional options: https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
mapproj = ccrs.EuroPP()

# Clear the figure
plt.clf()

# Create the plot axes with desired projection
fig, axs  = plt.subplots( nrows=1, ncols=2, 
            subplot_kw=dict(projection=mapproj) )

# Add Blue Marble
axs[0].add_wmts( wmts, layer1 ) 

# Add MODIS for specific day
axs[1].add_wmts( wmts, layer2, wmts_kwargs={'time': date} ) 


