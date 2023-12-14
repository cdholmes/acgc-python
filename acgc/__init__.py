"""Data analysis programs from the ACGC research group"""

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
