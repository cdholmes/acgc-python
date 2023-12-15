#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''A collection of statistical methods

The contents of all submodules are imported into `acgc.stats`. 
For example, `acgc.stats.sma` is the function `acgc.stats.bivariate_lines.sma`
within `acgc.stats.bivariate_lines`.

Useful functions include BivariateStatistics, line fitting methods (sma, sen, york), 
weighted statistics (wmean, wmedian, wcov, wcorr, etc.), partial_corr, among others. 
See ``help(acgc.stats)`` for complete list of methods.

See documentation within submodules.
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

