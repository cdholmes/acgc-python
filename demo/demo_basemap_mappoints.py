# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:43:20 2015

@author: cdholmes
"""

#conda install basemap

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Setup map projection and continent outlines
plt.clf()
m = Basemap(projection='kav7',lon_0=180)
m.drawmapboundary(fill_color='1')#blue water: '#99ffff')
m.fillcontinents(color='0.8',zorder=0)

# draw parallels and meridians, but don't bother labelling them.
m.drawparallels(np.arange(-90.,99.,30.),labels=np.ones(7,dtype=bool))
m.drawmeridians(np.arange(-180.,180.,60.),labels=np.ones(7,dtype=bool))

# create artificial data
N=50
lat = np.random.rand(N) * 180 - 90
lon = np.random.rand(N) * 360
c2h6 = np.random.rand(N) * 3000

# convert lat-lon to plot coordinates
x, y = m(lon,lat)

# scatterplot; could also use plt.scatter
m.scatter(x, y, marker='o', c=c2h6, s=50)
#m.scatter(lon, lat, latlon=True, marker='o', c=c2h6, s=50)

# add colorbar
m.colorbar(location='bottom')
plt.clim(0,2500)

#plt.title('Locations of %s ARGO floats active between %s and %s' %\
#        (10,date1,date2),fontsize=12)
#plt.show()