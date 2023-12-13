# ACGC

## Overview
The acgc package is a collection of data analysis functions used by the Atmospheric Chemistry and Global Change Research Group (ACGC). Programs are written in Python 3.

## Installation

For conda users:

 `conda install -c conda-forge acgc`

For pip users:

 `pip install acgc`

#### For developers
If you plan to modify or improve the acgc package, an editable installation may be better:
`pip install -e git+https://github.com/cdholmes/acgc-python`
Your local files can then be managed with git, including keeping up-to-date with the github source repository (e.g. `git pull`).

#### Classic version
The old version of this package (before conversion to an importable python module) is accessible as the "classic" branch of this repository on github.

## Contents

The acgc package includes the following sub-modules. Import these via `from acgc import <submodule>` or `import acgc.<submodule> as <alias>`. Example: `import acgc.gctools as gct`. 
See `help(acgc.<submodule>)` for complete list of features.

- [figstyle](./demo/demo_figstyle.ipynb)

Changes Matplotlib style to make figures closer to publication ready. 
- [stats](./demo/demo_stats.ipynb)  

Collection of statistical methods. Useful functions include BivariateStatistics, line fitting methods (sma, sen, york), weighted statistics (wmean, wmedian, wcov, wcorr, etc.), partial_corr, among others. See help(acgc.stats) for complete list of methods.
- erroranalysis   
Automatic error propagation through complex models
- gctools       
Tools for handling GEOS-Chem model output
- hytools	        
Tools for running HYSPLIT and reading HYSPLIT output
- icartt	        
Tools for reading data in ICARTT format
- igra		        
Tools for reading IGRA radiosonde data
- map_scalebar	        
Add a length scale bar on a map
- mettools        
Miscelaneous functions for PBL properties
- modetools	    
Visualization of eigenmode systems
- nctools          
High-level functions for reading and writing netCDF files. Legacy code. write_geo_nc is still useful for concisely creating netCDF files, but xarray is better for reading netCDF.
- solar        
Solar zenith angle, declination, equation of time
- time_tools       
Functions for calculating with dates. e.g. converting dates to fractional years. Legacy code. Use pandas.Timestamps or similar for new projects.

## Demos
The [`demo`](./demo/) folder contains examples of how to accomplish common data analysis and visualization tasks, including using many of the functions within the `acgc` library.

