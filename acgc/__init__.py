"""Data analysis programs from the ACGC research group"""

def package_version(package_name):
    '''Find version string for package name'''
    from importlib.metadata import version, PackageNotFoundError
    try:
        result = version(package_name)
    except PackageNotFoundError:
        result = "unknown version"
    return result

__version__ = package_version('acgc')
