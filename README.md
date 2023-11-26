# ACGC

## Overview
The acgc package is a collection of data analysis functions used by the Atmospheric Chemistry and Global Change Research Group (ACGC). Programs are written in Python 3.

## Installation

`pip install git+https://github.com/cdholmes/acgc-python`

You can then use the package via `from acgc import <submodule>`.

## Contents

The acgc package includes the following sub-modules. Import these via `from acgc import <submodule>` or `import acgc.<submodule> as <alias>`. Example: `import acgc.gctools as gct` 

- figstyle.py        
Changes Matplotlib style to make figures closer to publication ready. 
- bstats.py	        
Bivariate statistics
- erroranalysis.py    
Automatic error propagation through complex models
- gctools.py          
Tools for handling GEOS-Chem model output
- hytools.py	        
Tools for running HYSPLIT and reading HYSPLIT output
- icartt.py	        
Tools for reading data in ICARTT format
- igra.py		        
Tools for reading IGRA radiosonde data
- map_scalebar.py	        
Add a length scale bar on a map
- mettools.py	        
Miscelaneous functions for PBL properties
- modetools.py	    
Visualization of eigenmode systems
- partial_corr.py	    
Partial correlation
- sen_slope.py	    
Compute Sen's (robust) slope estimator
- smafit.py           
Standard Major Axis (SMA) line fitting, including robust methods and error analysis
- solar.py	        
Solar zenith angle, declination, equation of time
- time_tools.py       
Functions for calculating with dates. e.g. converting dates to fractional years
- wstats.py           
Weighted statistics. e.g. weighted mean, weighted R2, etc.
- york.py		        
York regression


The following sub-modules are are less commonly used or replaced with other features of pandas, xarray and other packages.
- boxcar.py           
Boxcar (running mean) filters
- boxcarpoly.py	    
Boxcar (running mean) polynomial fit
- execfile.py	        
Reproduces Python 2 "execfile" command in Python 3
- nctools.py          
High-level functions for reading and writing netCDF files
- loess.py	        
LOESS locally weighted least squares fitting
- tapply.py	        
tapply function from R implemented in Python (Use groupby methods in pandas or xarray when possible.)

## Demos
The [`demo`](./demo/) folder contains examples of how to accomplish common data analysis and visualization tasks, including using many of the functions within the `acgc` library.

## Author
This package is written by Christopher Holmes (cdholmes@fsu.edu).
