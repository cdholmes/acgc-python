#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Alias for `acgc.stats.bivariate` for backwards compatibility. Deprecated
'''
import warnings
from .stats.bivariate import nmb, nmae, nmbf, nmaef, BivariateStatistics

with warnings.catch_warnings():
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f'Module {__name__:s} is deprecated. Use acgc.stats instead.',
                  DeprecationWarning, stacklevel=2)
