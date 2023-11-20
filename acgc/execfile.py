# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:38:55 2015

@author: cdholmes
"""

def execfile(filename):
    # Reproduces python 2 "execfile" command in python 3+
    exec(open(filename).read())