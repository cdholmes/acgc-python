#!/usr/bin/env python3

from acgc import gc
import numpy as np

n1 = 5
n2 = 4
a1 = np.ones(n1)

p1 = np.array( [1000,800,600,400,200,0] )
p2 = np.array( [1000,600,500,250,0])

a2 = gc.regrid_plevels( a1, p1, p2 )

a3 = gc.regrid_plevels( a2, p2, p1 )

print(a2)
print(a3)
