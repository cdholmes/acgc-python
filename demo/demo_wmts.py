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

# Location of online data
URL = 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/wmts.cgi'
wmts = WebMapTileService(URL)

# Select desired dataset and date
# Additional datasets are listed here: https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+Available+Imagery+Products
layer = 'MODIS_Aqua_CorrectedReflectance_TrueColor'
date   = '2015-07-01'

# Map projection
# Additional options: https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
mapproj = ccrs.EuroPP()

# Clear the figure
plt.clf()

# Create the plot axes with desired projection
ax  = plt.subplot( 111, projection=mapproj )

# Add the dataset
ax.add_wmts( wmts, layer, wmts_kwargs={'time': date} ) 


