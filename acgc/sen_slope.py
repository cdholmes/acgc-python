#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Wrapper for acgc.stats for backwards compatibility. Deprecated

Functions:
    sen
    sen_slope
'''
import warnings
from .stats.bivariate_lines import sen, sen_slope

with warnings.catch_warnings():
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f'Module {__name__:s} is deprecated. Use acgc.stats instead.',
                  DeprecationWarning, stacklevel=2)
