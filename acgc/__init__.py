'''Data analysis programs from the ACGC research group

# Submodules
Submodules are imported via `import acgc.<submodule>` or `from acgc import <submodule>`. 

### Key submodules

- `acgc.figstyle`    
Style settings for matplotlib, for publication-ready figures.
[demo](https://github.com/cdholmes/acgc-python/blob/main/demo/demo_figstyle.ipynb)

- `acgc.stats`    
Collection of statistical methods. 
[demo](https://github.com/cdholmes/acgc-python/blob/main/demo/demo_stats.ipynb) 

### Other submodules

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

- `acgc.map_scalebar`	        
Scale bar for display on maps

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

# Demos
The [demo](https://github.com/cdholmes/acgc-python/blob/main/demo) 
folder contains examples of how to accomplish common data analysis and visualization tasks. 
The examples include uses of the `acgc` library as well as other libraries for 
geospatial data analysis.
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
    'map_scalebar',
    'met',
    'modetools',
    'netcdf',
    'stats',
    'solar',
    'time'
]
