'''Data analysis programs from the ACGC research group

# Installation

For conda users:

 `conda install -c conda-forge acgc`

For pip users:

 `pip install acgc`

### For developers
If you plan to modify or improve the acgc package, an editable installation may be better:
`pip install -e git+https://github.com/cdholmes/acgc-python`
Your local files can then be managed with git, including keeping up-to-date with the 
github source repository (e.g. `git pull`).

<!-- ----------------------- SECTION BREAK ----------------------- -->

# Get started

Submodules within `acgc` contain all the capabilities of the package.
Submodules are imported via `import acgc.<submodule>` or `from acgc import <submodule>`. 

## Better looking figures 

The default appearance of figures from Matplotlib doesn't meet the standards of most 
scientific journals (high-resolution, Helvetica-like font). With the `acgc.figstyle` module,
figures meet these criteria. Simply use `from acgc import figstyle` before creating your figures.

[Example using acgc.figstyle](https://github.com/cdholmes/acgc-python/blob/main/demo/demo_figstyle.ipynb)

## Bivariate statistics 

The `BivariateStatistics` class makes it easy to compute and display a large number of
bivariate statistics, including line fitting and weighted statistics. 
Results can be easily formatted into tables or inset in figures. Use `from acgc.stats import BivariateStatistics`
See `acgc.stats.bivariate.BivariateStatistics` for documentation.

[Example using BivariateStatistics](https://github.com/cdholmes/acgc-python/blob/main/demo/demo_stats.ipynb)

## Standard major axis (SMA) line fitting

SMA line fitting (also called reduced major axis or RMA) quantifies the linear relationship 
between variables in which neither one depends on the other.
It is available via `from acgc.stats import sma`. 

[Example using SMA](https://github.com/cdholmes/acgc-python/blob/main/demo/demo_sma.ipynb)

<!-- ----------------------- SECTION BREAK ----------------------- -->

# Demos
The [demo](https://github.com/cdholmes/acgc-python/blob/main/demo) 
folder contains examples of how to accomplish common data analysis and visualization tasks. 
The examples include uses of the `acgc` library as well as other libraries for 
geospatial data analysis.

<!-- ----------------------- SECTION BREAK ----------------------- -->

# Quick summary of submodules

## Key submodules

- `acgc.figstyle`    
Style settings for matplotlib, for publication-ready figures.
[demo](https://github.com/cdholmes/acgc-python/blob/main/demo/demo_figstyle.ipynb)

- `acgc.stats`    
Collection of statistical methods. 
[demo](https://github.com/cdholmes/acgc-python/blob/main/demo/demo_stats.ipynb) 

## Other submodules

- `acgc.erroranalysis`   
Propagation of error through complex numerical models

- `acgc.geoschem` or `acgc.gc`       
Tools for GEOS-Chem grids (e.g. indexing, remapping, interpolating)

- `acgc.hysplit`	        
Read HYSPLIT output and write HYSPLIT CONTROL files

- `acgc.icartt`	        
Read and write ICARTT format files

- `acgc.igra`		        
Read IGRA radiosonde data files

- `acgc.mapping`        
Distance calculation, scale bar for display on maps

- `acgc.met`        
Miscelaneous functions for PBL properties

- `acgc.modetools`	    
Visualization of eigenmode systems

- `acgc.netcdf`          
High-level functions for reading and writing netCDF files. Legacy code. 
`acgc.netcdf.write_geo_nc` is still useful for concisely creating netCDF files, 
but xarray is better for reading netCDF.

- `acgc.solar`        
Solar zenith angle, azimuth, declination, equation of time

- `acgc.time`       
Functions for manimulating times and dates. Legacy code.
'''

def _package_version(package_name):
    '''Find version string for package name'''
    from importlib.metadata import version, PackageNotFoundError
    try:
        result = version(package_name)
    except PackageNotFoundError:
        result = "unknown version"
    return result

__version__ = _package_version('acgc')

__all__ = [
    'erroranalysis',
    'figstyle',
    'gc',
    'geoschem',
    'hysplit',
    'icartt',
    'igra',
    'mapping',
    'met',
    'modetools',
    'netcdf',
    'stats',
    'solar',
    'time'
]
