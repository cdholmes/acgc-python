#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Alias for `acgc.netcdf` for backwards compatibility. Deprecated
'''
import warnings
from .hysplit import *

with warnings.catch_warnings():
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f'Module {__name__:s} is deprecated. Use acgc.netcdf instead.',
                  DeprecationWarning, stacklevel=2)
