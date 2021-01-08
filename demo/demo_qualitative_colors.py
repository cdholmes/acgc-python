#!/usr/bin/env python3
#%%
import numpy as np
from matplotlib import pyplot as plt

# Synthetic data
n = 30
x = np.arange(n)
y = np.random.randn(n)

# Assign group ID (3 consecutive points in each group)
g = x // 3

plt.scatter(x,y,c=g,cmap='Paired')

cm = plt.cm.get_cmap('tab20')

# %%
