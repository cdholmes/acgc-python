#!/usr/bin/env ipython3

import acgc.gctools as gct
import numpy as np 
import acgc.nctools as nct
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Read data from file
f='ssasnow_05x0625.201504010100.nc'
em  = nct.get_nc_var(f,'EmisSALA_Natural')
lat = nct.get_nc_var(f,'lat')
lon = nct.get_nc_var(f,'lon')
area= nct.get_nc_var(f,'AREA')

# Convert kg/m2/s -> kg/s
em = np.squeeze(em) * area

# Edges of input grid
elat = gct.get_lat(res=0.5,edge=True)
elon = gct.get_lon(res=0.5,edge=True)

# Edges of output grid
elat2 = gct.get_lat(res=2,edge=True)
elon2 = gct.get_lon(res=2,edge=True)

# Regrid emissions and area
em2 = gct.regrid2d( np.squeeze(em), elon, elat, elon2, elat2 )
area2 = gct.regrid2d( area, elon, elat, elon2, elat2 )

# Check that the emission total is the same
print(np.sum(em), np.sum(em2))

# Plot data on old and new grids
plt.figure(1)
plt.clf()
plt.subplot(2,1,1)
plt.pcolormesh( elon, elat, em/area, norm=colors.LogNorm() )
plt.colorbar()
clim = plt.gci().get_clim()
plt.subplot(2,1,2)
plt.pcolormesh( elon2, elat2, em2/area2, norm=colors.LogNorm(vmin=clim[0],vmax=clim[1]) )
plt.colorbar()
