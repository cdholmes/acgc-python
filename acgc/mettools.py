#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Alias for `acgc.met` for backwards compatibility. Deprecated
'''
import warnings
from .hysplit import *

with warnings.catch_warnings():
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f'Module {__name__:s} is deprecated. Use acgc.met instead.',
                  DeprecationWarning, stacklevel=2)
