#!/usr/bin/env python3

import acgc.gctools as gct
import numpy as np

n1 = 5
n2 = 4
a1 = np.ones(n1)

p1 = np.array( [1000,800,600,400,200,0] )
p2 = np.array( [1000,600,500,250,0])

a2 = gct.regrid_plevels( a1, p1, p2 )

a3 = gct.regrid_plevels( a2, p2, p1 )

print(a2)
print(a3)