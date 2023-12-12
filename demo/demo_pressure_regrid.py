#!/usr/bin/env python3

from . import gctools as gct
import numpy as np
import matplotlib.pyplot as plt

# Range of data: 0 to mx
mx = 10

# Number of centers in the old and new coordinates
n1 = 9
n2 = 4

# Pressure edges in the old and new coordinates
pe1 = np.linspace(0,mx,n1+1)
pe2 = np.linspace(0,mx,n2+1)

# Pressure centers; average the edges
pc1 = (pe1[1:] + pe1[:-1]) / 2
pc2 = (pe2[1:] + pe2[:-1]) / 2

# Pressure thickness of each level; difference the edges
dp1 =  pe1[1:] - pe1[:-1]
dp2 =  pe2[1:] - pe2[:-1]

# Create data with a peak
y1 = np.sin(pc1*np.pi/mx)**6 + pc1/mx

# Use mass-conservative regridding
y2 = gct.regrid_plevels(y1, pe1, pe2, intensive=True )

# Print the sums in the old and new coordinates; should be equal
print('sums', np.sum(y1*dp1), np.sum(y2*dp2))

# Plot the data
plt.clf()
plt.plot(pc1,y1,'o-', label='original')
plt.plot(pc2,y2,'o-', label='interp')
plt.legend()