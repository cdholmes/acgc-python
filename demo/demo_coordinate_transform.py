#!/usr/bin/env python3
'''Demonstrate conversion between geodetic coordinates (latitude, longitude) 
and map projection coordinates (x,y)'''

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import numpy  as np
import pandas as pd

# Define a few points in latitude, longitude
points_usa    = pd.DataFrame( {'lat':[40,45],'lon':[-70,-80]} )
points_arctic = pd.DataFrame( {'lat':[80,85],'lon':[10,-90]} )

# Define coordinate reference systems
# Note that for some applications, shape of the Earth should be specified 
geo = ccrs.Geodetic()
nps = ccrs.NorthPolarStereo()

# Convert geodetic coordinates to projection coordinates
points = nps.transform_points(geo,
                            points_arctic['lon'].values,
                            points_arctic['lat'].values )
# Add results to data frame
points_arctic['x'] = points[:,0]
points_arctic['y'] = points[:,1]

#%%
# Plot the points on a map
plt.figure(1)
plt.clf()

# Map projection for North Pole
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())

# Add coasts and states; Use ‘110m’ or ‘50m’ resolution
ax.coastlines(resolution='110m')
ax.add_feature(cfeature.STATES.with_scale('110m'))

# Plot points with geodetic coordinates
plt.scatter(points_arctic['lon'], 
            points_arctic['lat'], 
            marker='v', s=50, 
            transform=ccrs.PlateCarree())

# Plot points with projection coordinates, showing results are same
plt.scatter(points_arctic['x'], 
            points_arctic['y'], 
            marker='^', s=50, 
            transform=nps)

plt.show()
# %%
