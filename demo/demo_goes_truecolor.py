#!/usr/bin/env python3
'''Create true color image from GOES 16, 17 ABI

Instructions from
https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html
'''

'''
GOES Filename interpretation
OR_ABI-L2-MCMIPC-M3_G16_s20181781922189_e20181781924562_c20181781925075.nc
OR - Indicates the system is operational
ABI - Instrument type
L2 - Level 2 Data
MCMIP - Multichannel Cloud and Moisture Imagery products
c - CONUS file (created every 5 minutes).
M3 - Scan mode
G16 - GOES-16
sYYYYJJJHHMMSSZ - Scan start: 4 digit year, 3 digit day of year (Julian day), hour, minute, second, tenth second
eYYYYJJJHHMMSSZ - Scan end
cYYYYJJJHHMMSSZ - File Creation .nc - NetCDF file extension
'''
#%%
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy  # noqa: F401
import numpy as np
import xarray

# Open the file with xarray.
# The opened file is assigned to "C" for the CONUS domain.

FILE = ('http://ramadda-jetstream.unidata.ucar.edu/repository/opendap'
        '/4ef52e10-a7da-4405-bff4-e48f68bb6ba2/entry.das#fillmismatch')

#FILE='/Users/cdholmes/Documents/ResearchProjects/GOES-TrueColor/OR_ABI-L2-MCMIPC-M6_G17_s20192332001196_e20192332003575_c20192332004099.nc'

C = xarray.open_dataset(FILE)

# Scan's start time, converted to datetime object
scan_start = datetime.strptime(C.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')

# Scan's end time, converted to datetime object
scan_end = datetime.strptime(C.time_coverage_end, '%Y-%m-%dT%H:%M:%S.%fZ')

# File creation time, convert to datetime object
file_created = datetime.strptime(C.date_created, '%Y-%m-%dT%H:%M:%S.%fZ')

# The 't' variable is the scan's midpoint time
midpoint = str(C['t'].data)[:-8]
scan_mid = datetime.strptime(midpoint, '%Y-%m-%dT%H:%M:%S.%f')

print('Scan Start    : {}'.format(scan_start))
print('Scan midpoint : {}'.format(scan_mid))
print('Scan End      : {}'.format(scan_end))
print('File Created  : {}'.format(file_created))
print('Scan Duration : {:.2f} minutes'.format((scan_end-scan_start).seconds/60))

# Confirm that each band is the wavelength we are interested in
for band in [2, 3, 1]:
    print('{} is {:.2f} {}'.format(
        C['band_wavelength_C{:02d}'.format(band)].long_name,
        float(C['band_wavelength_C{:02d}'.format(band)][0]),
        C['band_wavelength_C{:02d}'.format(band)].units))

#%%
# Load the three channels into appropriate R, G, and B variables
R   = C['CMI_C02'].data
NIR = C['CMI_C03'].data
B   = C['CMI_C01'].data

# Apply range limits for each channel. RGB values must be between 0 and 1
R   = np.clip(R, 0, 1)
NIR = np.clip(NIR, 0, 1)
B   = np.clip(B, 0, 1)

# Apply a gamma correction to the image to correct ABI detector brightness
gamma = 1#2.2
R   = np.power(R,   1/gamma)
NIR = np.power(NIR, 1/gamma)
B   = np.power(B,   1/gamma)

# Calculate the "True" Green
# Suggested by UniData
G = 0.45 * R + 0.1 * NIR + 0.45 * B
# CIMSS blend
G = 0.48358168 * R + 0.06038137 * NIR + 0.45706946 * B
# College of Dupage
G = 0.41 * R + 0.13 * NIR + 0.46 * B
G = np.clip(G, 0, 1)  # apply limits again, just in case.

# Sqrt adjustment
R = np.sqrt(R)
G = np.sqrt(G)
B = np.sqrt(B)
NIR=np.sqrt(NIR)

# Contrast enhancement, from CIMSS True color AWIPS 
def enhcontrast(img):
    max_value = 1.0
    acont = (255.0 / 10.0) / 255.0
    amax = (255.0 + 4.0) / 255.0
    amid = 1.0 / 2.0
    afact = (amax * (acont + max_value) / (max_value * (amax - acont)))
    aband = (afact * (img - amid) + amid)
    aband[aband <= 10 / 255.0] = 0
    aband[aband >= 1.0] = 1.0
    return aband    
R = enhcontrast(R)
G = enhcontrast(G)
B = enhcontrast(B)
NIR = enhcontrast(NIR)

fig, ([ax1, ax2, ax3, ax4]) = plt.subplots(1, 4, figsize=(16, 3))

ax1.imshow(R, cmap='Reds', vmax=1, vmin=0)
ax1.set_title('Red', fontweight='bold')
ax1.axis('off')

ax2.imshow(NIR, cmap='Greens', vmax=1, vmin=0)
ax2.set_title('Veggie', fontweight='bold')
ax2.axis('off')

ax3.imshow(G, cmap='Greens', vmax=1, vmin=0)
ax3.set_title('"True" Green', fontweight='bold')
ax3.axis('off')

ax4.imshow(B, cmap='Blues', vmax=1, vmin=0)
ax4.set_title('Blue', fontweight='bold')
ax4.axis('off')

plt.subplots_adjust(wspace=.02)

# The RGB array with the raw veggie band
RGB_veggie = np.dstack([R, NIR, B])

# The RGB array for the true color image
RGB = np.dstack([R, G, B])

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# The RGB using the raw veggie band
ax1.imshow(RGB_veggie)
ax1.set_title('GOES-16 RGB Raw Veggie', fontweight='bold', loc='left',
              fontsize=12)
ax1.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
              loc='right')
ax1.axis('off')

# The RGB for the true color image
ax2.imshow(RGB)
ax2.set_title('GOES-16 RGB True Color', fontweight='bold', loc='left',
              fontsize=12)
ax2.set_title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')),
              loc='right')
ax2.axis('off')



# %%
# We'll use the `CMI_C02` variable as a 'hook' to get the CF metadata.
dat = C.metpy.parse_cf('CMI_C02')

geos = dat.metpy.cartopy_crs

# We also need the x (north/south) and y (east/west) axis sweep of the ABI data
x = dat.x
y = dat.y

fig = plt.figure(figsize=(15, 12))

# Create axis with Geostationary projection
ax = fig.add_subplot(1, 1, 1, projection=geos)

# Add the RGB image to the figure. The data is in the same projection as the
# axis we just created.
ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()), transform=geos)

# Add Coastlines and States
ax.coastlines(resolution='10m', color='black', linewidth=0.25)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.25)

plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')

plt.show()

#%%

fig = plt.figure(figsize=(15, 12))

# Generate an Cartopy projection
lc = ccrs.LambertConformal(central_longitude=-97.5, standard_parallels=(38.5,
                                                                        38.5))

ax = fig.add_subplot(1, 1, 1, projection=lc)
ax.set_extent([-135, -60, 10, 65], crs=ccrs.PlateCarree())

ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos,
          interpolation='none')
ax.coastlines(resolution='10m', color='black', linewidth=0.5)
#ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)

plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')

plt.show()


# %%
# Zoom in on Utah
fig = plt.figure(figsize=(8, 8))

pc = ccrs.PlateCarree()

ax = fig.add_subplot(1, 1, 1, projection=pc)
ax.set_extent([-114.75, -108.25, 36, 43], crs=pc)

ax.imshow(RGB, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos,
          interpolation='none')

ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(ccrs.cartopy.feature.STATES)

plt.title('GOES-16 True Color', loc='left', fontweight='bold', fontsize=15)
plt.title('{}'.format(scan_start.strftime('%d %B %Y %H:%M UTC ')), loc='right')

plt.show()

#%%

# Day-night example
FILE = ('http://ramadda-jetstream.unidata.ucar.edu/repository/opendap'
        '/85da3304-b910-472b-aedf-a6d8c1148131/entry.das#fillmismatch')
C = xarray.open_dataset(FILE)

# Scan's start time, converted to datetime object
scan_start = datetime.strptime(C.time_coverage_start, '%Y-%m-%dT%H:%M:%S.%fZ')

# Create the RGB like we did before

# Load the three channels into appropriate R, G, and B
R   = C['CMI_C02'].data
NIR = C['CMI_C03'].data
B   = C['CMI_C01'].data

# Apply range limits for each channel. RGB values must be between 0 and 1
R   = np.clip(R,   0, 1)
NIR = np.clip(NIR, 0, 1)
B   = np.clip(B,   0, 1)

# Apply the gamma correction
gamma = 2.2
R   = np.power(R,   1/gamma)
NIR = np.power(NIR, 1/gamma)
B   = np.power(B,   1/gamma)

# Calculate the "True" Green
G = 0.45 * R + 0.1 * NIR + 0.45 * B
G = np.clip(G, 0, 1)

# The final RGB array :)
RGB = np.dstack([R, G, B])


cleanIR = C['CMI_C13'].data

# Normalize the channel between a range.
#       cleanIR = (cleanIR-minimumValue)/(maximumValue-minimumValue)
cleanIR = (cleanIR-90)/(313-90)

# Apply range limits to make sure values are between 0 and 1
cleanIR = np.clip(cleanIR, 0, 1)

# Invert colors so that cold clouds are white
cleanIR = 1 - cleanIR

# Lessen the brightness of the coldest clouds so they don't appear so bright
# when we overlay it on the true color image.
cleanIR = cleanIR/1.4

# Yes, we still need 3 channels as RGB values. This will be a grey image.
RGB_cleanIR = np.dstack([cleanIR, cleanIR, cleanIR])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.set_title('True Color', fontweight='bold')
ax1.imshow(RGB)
ax1.axis('off')

ax2.set_title('Clean IR', fontweight='bold')
ax2.imshow(RGB_cleanIR)
ax2.axis('off')

plt.show()

#%%
# Combine day and night images, use maximum

# Maximize the RGB values between the True Color Image and Clean IR image
RGB_ColorIR = np.dstack([np.maximum(R, cleanIR), np.maximum(G, cleanIR),
                         np.maximum(B, cleanIR)])

fig = plt.figure(figsize=(15, 12))

ax = fig.add_subplot(1, 1, 1, projection=geos)

ax.imshow(RGB_ColorIR, origin='upper',
          extent=(x.min(), x.max(), y.min(), y.max()),
          transform=geos)

ax.coastlines(resolution='50m', color='black', linewidth=2)
ax.add_feature(ccrs.cartopy.feature.STATES)

plt.title('GOES-16 True Color and Night IR', loc='left', fontweight='bold',
          fontsize=15)
plt.title('{}'.format(scan_start.strftime('%H:%M UTC %d %B %Y'), loc='right'),
          loc='right')

plt.show()

# %%
