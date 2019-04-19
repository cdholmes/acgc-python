# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 17:23:09 2015

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# time interval, days
dt = 1
# sampling frequency, 
fs = 1/dt

# Nyquist frequency, 1/days
nyquist = fs/2

# number of data points
N = 365*5

# time variable
t = np.arange(N)*dt

## Create a synthetic dataset
# frequency for annual cycle
f1 = 1/365
# frequency for weekly cycle
f2 = 1/7
# time series with annual cycle and weekly cycle
y = np.sin(2*np.pi*f1*t) + 0.2* np.sin(2*np.pi*f2*t)
# Add normaly distributed white noise
y = y + 0.2*np.random.randn(N)


## Design Butterworth IIR filter
# Pass frequencies above 1/10 per day; loss should be < 0.1 dB
# Attenuation below 1/20 should be > 20 dB
on, wn = sig.buttord(1/20/nyquist,1/30/nyquist,0.1,20,analog=False)
# Get filter coefficients
b, a = sig.butter(on, wn, 'high', analog=False)
# calculate attenuation
w, h = sig.freqz(b, a)

## Design a FIR filter
# Pass frequencies above 1/12 per day
# Filter order and cut frequency is chosen ad hoc by looking at attenuation vs. frequency and iterating
bfir=sig.firwin(101,1/12/nyquist,pass_zero=False)
# calculate attenuation
wfir,hfir = sig.freqz(bfir,1)

## Apply filters
# Butterworth IIR
yfilt = sig.lfilter(b,a,y)
# FIR
yfiltfir = sig.lfilter(bfir,1,y)

## Plot
plt.clf()

plt.subplot(4,1,1)
plt.semilogx(w/np.pi*nyquist, 20 * np.log10(abs(h)),'b')
plt.semilogx(wfir/np.pi*nyquist,20 * np.log10(abs(hfir)),'r')
plt.legend(['Butterworth filter','FIR filter'],loc='lower right')
plt.xlabel('Frequency, 1/day')
plt.ylabel('Attenuation, dB')
plt.xlim([1/(N*dt),nyquist])

plt.subplot(4,1,2)
plt.semilogx(w/np.pi*nyquist, np.unwrap(np.angle(h)),'b')
plt.semilogx(wfir/np.pi*nyquist, np.unwrap(np.angle(hfir)),'r')
plt.xlabel('Frequency, 1/day')
plt.ylabel('Phase shift, degree')
plt.xlim([1/(N*dt),nyquist])

plt.subplot(4,1,3)
plt.plot(t,y,'k')
plt.plot(t,yfilt,'b')
plt.plot(t,yfiltfir,'r')
plt.xlabel('time, day')
plt.ylabel('signal')
plt.legend(['Unfiltered','Butterworth filter','FIR filter'])

plt.subplot(4,1,4)
# calculate spectral power density
f,pow = sig.periodogram(y,fs)
f,powfilt = sig.periodogram(yfilt,fs)
f,powfiltfir = sig.periodogram(yfiltfir,fs)
plt.loglog(f,pow,'k')
plt.plot(f,powfilt,'b')
plt.plot(f,powfiltfir,'r')
plt.ylim([1e-6,1e3])
plt.xlim([1/(N*dt),nyquist])
plt.xlabel('frequency, 1/day')
plt.ylabel('power density')
plt.legend(['Unfiltered','Butterworth filter','FIR filter'],loc='upper center')
plt.text(1/7,10,'7 day cycle')
plt.text(1/365,10,'365-day cycle')

plt.tight_layout()
