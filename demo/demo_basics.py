# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:45:23 2015

@author: cdholmes
"""

# Define two variables
a = 1
b = 2

# Define a third variable from the first two
c = a + b

# Notice that Python hasn't displayed anything yet, so let's see what value "c" has.
# Display the value
print(c)

# Notice that "=" means "assigment" in programming: 
# evaluate the right-hand side and assign its value to the variable on the left-hand side 
# In python (and most other languages) "c = c + 1" means 
# add 1 to the current value of c, then assign that value to the variable c (which overwrites the previous value)
# In algebra, "c = c + 1" would be false nonsense because there is no number which equals itself plus one.
c = c + 1
print(c)

# Define variables that are strings
d = 'Hello'
e = 'World'

# Combine the strings and print them
print(d+' '+e)

# Math formulas are available from the Numerical Python (numpy) module
# We "import" it in order to use it
import numpy as np

# Let's use some trigonometry functions
f = np.sin(np.pi/2)
g = np.arccos(1)

print(f,g)

# All the variables so far have been "scalars", meaning a single value
# List
h = [1, 2, 3]

# Numpy array
i = np.array([1, 2, 3])