#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:58:41 2017

@author: cdholmes
"""

import numpy as np
import acgc.nctools as nct


nct.write_geo_ncdf('test.nc',globalAtt={'Author':'Me'},
                   xDim=np.arange(100))
