# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 17:23:09 2015

@author: cdholmes
"""

import numpy as np
import matplotlib.pyplot as p
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
p.clf()

p.subplot(4,1,1)
p.semilogx(w/np.pi*nyquist, 20 * np.log10(abs(h)),'b')
p.semilogx(wfir/np.pi*nyquist,20 * np.log10(abs(hfir)),'r')
p.legend(['Butterworth filter','FIR filter'],loc='lower right')
p.xlabel('Frequency, 1/day')
p.ylabel('Attenuation, dB')
p.xlim([1/(N*dt),nyquist])

p.subplot(4,1,2)
p.plot(w/np.pi*nyquist, np.unwrap(np.angle(h)),'b')
p.plot(wfir/np.pi*nyquist, np.unwrap(np.angle(hfir)),'r')

p.subplot(4,1,3)
p.plot(t,y,'k')
p.plot(t,yfilt,'b')
p.plot(t,yfiltfir,'r')
p.xlabel('time, day')
p.ylabel('signal')
p.legend(['Unfiltered','Butterworth filter','FIR filter'])

p.subplot(4,1,4)
# calculate spectral power density
f,pow = sig.periodogram(y,fs)
f,powfilt = sig.periodogram(yfilt,fs)
f,powfiltfir = sig.periodogram(yfiltfir,fs)
p.loglog(f,pow,'k')
p.plot(f,powfilt,'b')
p.plot(f,powfiltfir,'r')
p.ylim([1e-5,1e3])
p.xlim([1/(N*dt),nyquist])
p.xlabel('frequency, 1/day')
p.ylabel('power density')
p.legend(['Unfiltered','Butterworth filter','FIR filter'],loc='upper center')
p.text(1/7,10,'7 day cycle')
p.text(1/365,10,'365-day cycle')