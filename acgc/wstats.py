#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Wrapper for acgc.stats for backwards compatibility. Deprecated

Functions:
    wcorr
    wcorrcoef
    wcov
    wmean
    wmedian
    wquantile
    wscale
    wstd
    wvar
'''
import warnings
from .stats.weighted import wcorr, wcorrcoef, wcov, wmean, wmedian, wquantile, _wscale, wstd, wvar

with warnings.catch_warnings():
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f'Module {__name__:s} is deprecated. Use acgc.stats instead.',
                  DeprecationWarning, stacklevel=2)
