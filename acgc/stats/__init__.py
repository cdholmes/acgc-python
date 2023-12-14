#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''A collection of statistical methods

The contents of all sub-modules are imported into `acgc.stats`. 
For example, `acgc.stats.sma` is the function `acgc.stats.bivariate_lines.sma`
within `acgc.stats.bivariate_lines`.

See documentation within sub-modules.

'''

from .bivariate import *
from .bivariate_lines import *
from .boxcar import *
from .partial_corr import *
from .weighted import *

# Other sub-modules that are not imported because of
# better options or limited applications
# from .tapply import tapply    # use groupby in pandas and xarray
# from .sma_warton_fit import * # Use sma
try:
    # Requires R and rpy2
    from .loess import loess
except ModuleNotFoundError:
    # Skip
    pass

