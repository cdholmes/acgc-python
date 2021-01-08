#!/usr/bin/env python3
'''Example of rotating a plot marker symbol
'''

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt 


# make a markerstyle class 
t = mpl.markers.MarkerStyle(marker='^')
plt.scatter((0), (0), marker=t, s=100)

# Rotate 30 degrees counter-clockwise
t._transform = t.get_transform().rotate_deg(30)
plt.scatter((1), (1), marker=t, s=100)

# Rotate an /additional/ 60 degrees
t._transform = t.get_transform().rotate_deg(60)
plt.scatter((1.5), (1.5), marker=t, s=100)

# Triangle marker rotated 45 degrees counter-clockwise
plt.scatter((2), (2), marker=(3,0,45), s=100,color='black')

# %%
