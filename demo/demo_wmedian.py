#!/usr/bin/env python3

import acgc.stats as ws
import numpy as np

n = 10
a = np.linspace(1,10,n)
w = np.ones(n)
# w[0] = 50
w = np.random.random(n)

# p = np.random.permutation(n)
# a = a[p]
# w = w[p]

# r = ws.wquantile(a,0.25,w)

print('\nExample 0, result should be 3')
print(ws.wquantile( [1,2,3,4,5], 0.5, [1,1,1,1,1]))

print('\nExample 1, result should be 4')
print(ws.wquantile( [1,2,3,4,5], 0.5, [0.15,0.1,0.2,0.3,0.25]))

print('\nExample 2, result should be 2.5')
print(ws.wquantile( [1,2,3,4], 0.5, [1,1,1,1]))
print(np.percentile([1,2,3,4], 50,interpolation='midpoint'))

print('\nExample 3, result should be 2.5')
print(ws.wquantile( [1,2,3,4], 0.5, [0.49,0.01,0.25,0.25]))

print('\nExample 4, result should be 1')
print(ws.wquantile( [0,1,2], 0.5, [1,1,1]))
print(np.quantile([0,1,2], 0.5))
print(np.percentile([0,1,2], 50,interpolation='midpoint'))

print('\n')
print(ws.wquantile([0,1,2], [0.2,0.25,0.3,0.5,0.75], [1,2,1]))

print('last test')
import matplotlib.pyplot as plt
qq = np.linspace(0,1,101)
a = np.array([0,1,2,3,4])
w = np.ones_like(a)
p = np.random.permutation(len(a))
a = a[p]
w = w[p]
plt.figure(1)
plt.clf()
plt.plot(qq,np.percentile(a,qq*100,interpolation='linear'),label='linear')
plt.plot(qq,np.quantile(a,qq),label='qlinear')
plt.plot(qq,ws.wquantile(a,qq,w,interpolation='linear'),label='wlinear')

plt.plot(qq,np.percentile(a,qq*100,interpolation='nearest'),'x',label='nearest')
plt.plot(qq,ws.wquantile(a,qq,w,interpolation='nearest'),'x',label='wnearest')

# plt.plot(qq,np.percentile(a,qq*100,interpolation='midpoint'),'.',label='midpoint')
# plt.plot(qq,ws.wquantile(a,qq,w,interpolation='midpoint'),'.',label='wmidpoint')

# plt.plot(qq,np.percentile(a,qq*100,interpolation='higher'),label='midpoint')
# plt.plot(qq,ws.wquantile(a,qq,w,interpolation='higher'),label='wmidpoint')

plt.plot(qq,ws.wquantile(a,qq,w,interpolation='partition'),'o',label='partition')
plt.legend()
plt.show()

plt.figure(2)
plt.clf()
plt.plot(qq,np.quantile(a,qq),'.',label='q4')
plt.plot(qq,np.quantile(np.repeat(a,10),qq),'.',label='q40')
plt.plot(qq,ws.wquantile(a,qq,w,interpolation='partition'),'o',label='partition')
plt.legend()
plt.show()
