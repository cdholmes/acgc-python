#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Wrapper for acgc.stats for backwards compatibility. Deprecated

Classes:
    BivariateStatistics

Functions:
    nmb
    nmae
    nmbf
    nmaef
'''
import warnings
from .stats.bivariate import nmb, nmae, nmbf, nmaef, BivariateStatistics

with warnings.catch_warnings():
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f'Module {__name__:s} is deprecated. Use acgc.stats instead.',
                  DeprecationWarning, stacklevel=2)
