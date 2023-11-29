# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:49:33 2015

@author: cdholmes
"""


from datetime import datetime
import warnings
import netCDF4 as nc
import numpy   as np

def get_nc_var(filename,varname):
    """Read a variable from a netCDF file
    
    Parameters
    ----------
    filename : str
        name/path of netCDF file
    varname : str
        name of variable that will be retrieved
        
    Returns
    -------
    data : N-D array
        value of variable
    """

    # Open file for reading
    ncfile = nc.Dataset(filename,'r')

    # Get the desired variable
    data = ncfile.variables[varname][:]

    # Close the file
    ncfile.close()

    return data

def get_nc_att(filename,varname,attname,glob=False):
    """ Read an attribute from a netCDF file
          
    Parameters
    ----------
    filename : str
        name/path of netCDF file
    varname : str
        name of variable 
    attname : str
        name of attribute that will be retrieved
    glob : bool (default=False)
        Set glob=True to access global file attribues (varname will be ignored) 
        and glob=False for variable attributes
        
    Returns
    -------
    data : float or str
        attribute value
    """

    # Open file for reading
    ncfile = nc.Dataset(filename,'r')

    # Get the desired attribute
    if glob:
        data = ncfile.getncattr(attname)
    else:
        data = ncfile.variables[varname].getncattr(attname)

    # Close the file
    ncfile.close()

    return data
    
def get_nc_varnames(filename):
    """ Read variable names from a netCDF file
    
    Parameters
    ----------
    filename : str
        name/path of netCDF file
    
    Returns
    -------
    list of strings containing variable names within filename   
    """

    # Open file for reading
    ncfile = nc.Dataset(filename,'r')

    # Get the desired variable
    data = list(ncfile.variables.keys())

    # Close the file
    ncfile.close()

    return data

def get_nc_attnames(filename,varname,glob=False):
    """ Read attributes from a netCDF file
    
    Parameters
    ----------
    filename : str
        name/path of netCDF file
    varname : str
        name of variable
    glob : bool (default=False)
        Set glob=True to access global file attribues (varname will be ignored) 
        and glob=False for variable attributes        
    
    Returns
    -------
    list of strings containing attribute names   
    """

    # Open file for reading
    ncfile = nc.Dataset(filename,'r')

    # Get the attribute names
    if glob:
        data = ncfile.ncattrs()
    else:  
        data = ncfile.variables[varname].ncattrs()

    # Close the file
    ncfile.close()

    return data

def put_nc_var(filename,varname,value):
    """ Assign a new value to an existing variable and existing file
    
    Parameters
    ----------
    filename : str
        name/path of netCDF file
    varname : str
        name of variable that will be assigned
    value : N-D array
        data values that will be assigned to variable
        must have same shape as the current variable values
    """

    # Open file for reading
    ncfile = nc.Dataset(filename,'w')

    # Set value
    ncfile.variables[varname][:] = value

    # Close the file
    ncfile.close()

def put_nc_att(filename,varname,attname,value,glob=False):
    """ Assign a new value to an existing attribute
    
    Parameters
    ----------
    filename : str
        name/path of netCDF file
    varname : str
        name of variable
    attname : str
        name of attribute that will be assigned
    value : str, float, list
        data values that will be assigned to the attribute
    """

    # Open file for reading
    ncfile = nc.Dataset(filename,'w')

    # Set attribute
    if glob:
        ncfile.setncattr(attname,value)
    else:
        ncfile.variables[varname].setncattr(attname,value)

    # Close the file
    ncfile.close()

def write_geo_nc(filename, variables,
                xDim=None, yDim=None,
                zDim=None, zUnits=None,
                tDim=None, tUnits=None,
                globalAtt=None,
                classic=True, nc4=True, compress=True,
                clobber=False, verbose=False ):
    '''Create a NetCDF file with geospatial data. Output file is COARDS/CF compliant
    
    This function allows specification of netCDF files more concisely than many alternative
    python packages (e.g. xarray, netCDF4) by making assumptions about the dimensions and 
    inferring the dimensions for each variable from the variable shape. This is well suited 
    for many lat-lon-lev-time and xyzt datasets. 

    Each variable is defined as a dict. 
    Required keys: 
        'name'  (str)       variable name 
        'value' (numeric)   N-D array of variable data 
    Special keys (all optional):
        'dim_names' (list,str)  names of the dimension variables corresponding to dimensions of variable
            If dim_names is not provided, the dimension variables will be inferred from the data shape.
            If all dimensions have unique lengths, the inferred dimensions are unambiguous. 
            If two or more dimensions have equal lengths, then the dim_names key should be used.
        'fill_value'(numeric) value that should replace NaNs
        'unlimited' (bool)  specifies if dimension is unlimited
        'pack'      (bool)  specifies that variable should be compressed with integer packing
        'packtype'  (str)   numeric type for packed data, commonly i1 or i2 (default='i2')
        'calendar'  (str)   string for COARDS/CF calendar convention. Only used for time variable
    All other keys are assigned to variable attributes. CF conventions expect the following:
        'long_name' (str)   long name for variable
        'units'     (str)   units of variable

    e.g. {'name': 'O3',
          'value': data,
          'long_name': 'ozone mole fraction',
          'units': 'mol/mol'}
    

    Parameters
    ----------
    filename : str
        name/path for file that will be created
    variables : list of dict-like
        Each variable is specified as a dict, as described above.
    xDim : array or dict-like (optional)
        x dimension of data. If dict-like, then it should contain same keys as variables.
        If xDim is an array, then it is assumed to be longitude in degrees east and named 'lat'
    yDim : array or dict-like (optional)
        y dimension of data. If dict-like, then it should contain same keys as variables.
        If yDim is an array, then it is assumed to be latitude in degrees north and named 'lon'
    zDim : array or dict-like (optional)
        z dimension of data. If dict-like, then it should contain same keys as variables.
        If zDim is an array, then it is named 'lev'
        zUnits is named used to infer the variable long name:
            m, km   -> zDim is "altitude"
            Pa, hPa -> zDim is "pressure"
            None    -> zDim is "level"
    zUnits : str (optional)
        Units for zDim, ignored if zDim is dict-like. Accepted values are m, km, Pa, level, '', None
    tDim : array or dict-like (optional)
        time dimension of data. If dict-like, then it should contain the same keys as variables.
        If tDim is an array, then tUnits is used and the dimension is set as unlimited and 
        named 'time'. datetime-like variables are supported, as are floats and numeric.
    tUnits : str (optional)
        Units for tDim. Special treatment will be used for "<time units> since <date>"
    globalAtt : dict-like (optional)
        dict of global file attributes
    classic : bool (default=True)
        specify whether file should use netCDF classic data model (includes netCDF4 classic)
    nc4 : bool (default=True)
        specify whether file should be netCDF4. Required for compression.
    compress : bool (default=True)
        specify whether all variables should be compressed (lossless). 
        In addition to lossless compression, setting pack=True for individual variables enables 
        lossy integer packing compression.
    clobber : bool (default=False)
        specify whether a pre-existing file with the same name should be overwritten
    verbose : bool (default=False)
        specify whether extra output should be written while creating the netCDF file
    '''

    # NetCDF file type
    if nc4:
        if classic:
            ncfmt = 'NETCDF4_CLASSIC'
        else:
            ncfmt = 'NETCDF4'
    else:
        ncfmt = 'NETCDF3_64BIT_OFFSET'

    ### Open file for output

    f = nc.Dataset( filename, 'w', format=ncfmt, clobber=clobber )
    f.Conventions = "COARDS/CF"
    f.History = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + \
        ' : Created by write_geo_nc (python)'

    # Write global attributes, if any
    if globalAtt is not None:
        f.setncatts(globalAtt)

    ### Define dimensions

    dimName = []
    dimList = []
    dimSize = []
    varList = []

    if xDim is not None:
        if not isinstance(xDim, dict):
            xDim = {'name': 'lon',
                    'value': xDim,
                    'long_name': 'longitude',
                    'units': 'degrees_east'}    

        ncDim, ncVar = _create_geo_dim( xDim, f,
                                      compress=compress,
                                      classic=classic,
                                      verbose=verbose )

        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(ncVar[:]) )

    if yDim is not None:
        if not isinstance(yDim, dict):
            yDim = {'name': 'lat',
                    'value': yDim,
                    'long_name': 'latitude',
                    'units': 'degrees_north'}

        ncDim, ncVar = _create_geo_dim( yDim, f,
                                      compress=compress,
                                      classic=classic,
                                      verbose=verbose )

        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(ncVar[:]) )

    if zDim is not None:
        if not isinstance(zDim, dict):
            # Infer name from units
            if zUnits in ['m','km']:
                lname = 'altitude'
            elif zUnits in ['Pa','hPa']:
                lname = 'pressure'
            elif zUnits in ['level','',None]:
                lname = 'level'
                zUnits = ''
            else:
                raise ValueError( f'Units of {zUnits:s} for zDim have not been implemented' )

            zDim = {'name': 'lev',
                    'value': zDim,
                    'long_name': lname,
                    'units': zUnits}

        ncDim, ncVar = _create_geo_dim( zDim, f,
                                      compress=compress,
                                      classic=classic,
                                      verbose=verbose )

        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(ncVar[:]) )

    if tDim is not None:
        if not isinstance(tDim, dict):
            if tUnits is None:
                tUnits = ''
            tDim = {'name': 'time',
                    'value': tDim,
                    'long_name': 'time',
                    'units': tUnits,
                    'unlimited': True }

        ncDim, ncVar = _create_geo_dim( tDim, f,
                                      compress=compress,
                                      classic=classic,
                                      time=True,
                                      verbose=verbose )

        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(ncVar[:]) )

    # Dimension sizes that are not unique i.e. shared by two or more dimensions
    duplicate_dim_sizes = {s for s in dimSize if dimSize.count(s) > 1}

    ### Define variables

    for var in variables:

        if type(var) is not dict:
            raise TypeError( 'All variables must be passed as dicts' )

        # Shape of the variable
        vShape = var['value'].shape

        # If dim_names is provided, otherwise use inference
        if 'dim_names' in var.keys():

            # Set dimensions based on provided names
            dID = var['dim_names']

            # Confirm that correct number of dimensions are provided
            if len(dID) != len(vShape):
                raise ValueError( 'Variable {:s} has dimension {:d} and {:d} dim_names'.\
                                 format( var['name'], len(vShape), len(dID) ) )
            # Shape of each named dimension
            dShape = tuple( dimSize[dimName.index(d)] for d in dID )

            # Confirm that the named dimensions match the shape of the variable
            if dShape != vShape:
                raise ValueError( 'Shape of the dimensions [{:s}]=[{:s}] must match '.\
                                 format( ','.join(dID),','.join([str(i) for i in dShape]) )
                                 + 'the shape of variable {:s} which is [{:s}]'.\
                                 format( var['name'],','.join([str(i) for i in vShape]) ) )

        else:

            # If this variable uses any of the duplicate dims, give a warning 
            if np.any(np.isin(vShape, duplicate_dim_sizes)):
                warnings.warn('Dimensions of variable {:s} cannot be uniquely identified.\n'.\
                              format(var['name'])
                              +'The "dim_names" key is strongly recommended.')

            # Match dimensions of variable with defined dimensions based on shape
            dID = []
            for s in vShape:

                # Find the dimensions that match the variable dimension
                try:
                    i = dimSize.index(s)
                except ValueError:
                    # No dimensions match
                    raise ValueError( 'Cannot match dimensions for variable {:s}'.\
                                     format( var['name'] ) )

                # List of the dimension names
                dID.append(dimName[i])

        # Create the variable
        ncVar = _create_geo_var( var, f, dID, compress=compress, classic=classic, verbose=verbose )

        varList.append( ncVar )

    # Close the file
    f.close()

def _create_geo_dim( var, fid, **kwargs ):
    '''Create a netCDF dimension and associated variable in a open file

    Parameters
    ----------
    var : dict-lik
        Must contain keys for 'name', 'value'. Other keys will be assiged to attributes. 
        e.g. 'unlimited' for unlimited dimension.
    fid : netCDF file ID
        reference to an open file

    Returns
    -------
    ncDim, ncVar : 
        IDs for the dimension and variable that were created
    '''

    try:
        assert ('name' in var), \
            'Var structure must contain "name" key: create_geo_dim'
        assert ('value' in var), \
            'Var structure must contain "value" key: create_geo_dim'
    except AssertionError as exc:
        # Close file and exit
        fid.close()
        raise IOError() from exc

    size = len(var['value'])

    # If the unlimited tag is True, then set size=None to create unlimited dimension
    if 'unlimited' in var.keys():
        if var['unlimited']:
            size = None

    ncDim = fid.createDimension( var['name'], size )

    ncVar = _create_geo_var( var, fid, (var['name']), isDim=True, **kwargs )

    return ncDim, ncVar

def _get_nc_type( value, name='', classic=True ):
    '''Get netCDF variable type for a python variable
    
    Parameters
    ----------
    value : any data type
    name : str (optional)
        name of the variable being examined. Only used for error messages
    classic : bool
        If classic=True, then error will be raised if variable type is not 
        allowed in netCDF classic files

    Returns
    -------
    vartype : str

    '''

    # Variable type for numpy arrays or native python
    try:
        vartype = value.dtype.str
    except AttributeError:
        vartype = type(value)

    # Remove brackets
    vartype = vartype.replace('<','').replace('>','')

    # Raise error if the type isn't allowed
    if (classic and (vartype not in ['i1','i2','i4','f4','f8','U1','str','bytes'])):
        raise TypeError( 'Variable {:s} cannot have type {:s} in netCDF classic files'.format(
            name, vartype ) )

    return vartype

def _infer_pack_scale_offset( data, nbits ):
    '''Compute scale and offset for packing data to nbit integers 
    Follow Unidata recommendations:
    https://www.unidata.ucar.edu/software/netcdf/documentation/NUG/_best_practices.html
    
    Parameters
    ----------
    data : N-D array
        data to be compressed
    nbits : int
        number of bits in the packed data type

    Returns
    -------
    scale_factor, add_offset : numeric
        recommended values for scale and offset
    pack_fill_value : numeric
        recommended fill value within the integer range
    '''
    min = np.nanmin( data )
    max = np.nanmax( data )
    scale_factor = ( max - min ) / (2**nbits - 2)
    add_offset   = ( max + min ) / 2

    # Use a fill value within the packed range
    pack_fill_value   = -2**(nbits-1)

    return scale_factor, add_offset, pack_fill_value

def _integer_pack_data( data, scale_factor, add_offset, fill_value, pack_fill_value ):
    '''Pack data using the provided scale, offset, and fill value
    
    Parameters
    ----------
    data : N-D array
        data values to be packed
    scale_factor, add_offset : float
        scale and offset for packing
    fill_value : numeric
        fill value for the unpacked data
    pack_fill_value : numeric
        fill value for the packed data, must be in the allowed range of the packed integer type
    
    Returns
    -------
    pack_value : N-D array
        data values converted to the integer range
    '''

    # Packed values
    pack_value = ( data[:] - add_offset ) / scale_factor

    # Set missing value
    if np.ma.is_masked( data ):
        pack_value[ data.mask ] = pack_fill_value
    if fill_value is not None:
        pack_value[ data==fill_value ] = pack_fill_value
    if np.any( np.isnan(data) ):
        pack_value[ np.isnan( data ) ] = pack_fill_value
    if np.any( np.isinf(data) ):
        pack_value[ np.isinf( data ) ] = pack_fill_value

    return np.round( pack_value )

def _create_geo_var( var, fid, dimIDs, compress=True, classic=True, time=False,
                    fill_value=None, verbose=False, isDim=False ):
    '''Create a netCDF variable, assuming common geospatial conventions
    
    Parameters
    ----------
    var : dict-like
        See create_geo_nc for required and optional dict keys.
    fid : netCDF file ID
        reference to an open file
    dimIDs : list
        list of dimensions ID numbers corresponding to the dimensions of var.value
    compress : bool (default=True)
        compress=True indicates that variable should be deflated with lossless compression
    classic : bool (default=True)
        specify if file is netCDF Classic or netCDF4 Classic
    time : bool (default=False)
        specify if variable has time units that require handling with calendar
    isDim : bool (default=False)
        indicates dimesion variables
        
    Returns
    -------
    ncVar : netCDF variable ID that was created
    '''
    try:
        assert ('name' in var), \
            'Var structure must contain "name" key: create_geo_var'
        assert ('value' in var), \
            'Var structure must contain "value" key: create_geo_var'
    except AssertionError as exc:
        # Close file and exit
        fid.close()
        raise IOError from exc

    # If this is a time variable and units are "<time units> since <date>", then convert
    if (time and (' since ' in var['units']) ):
        if 'calendar' in var.keys():
            calendar = var['calendar']
        else:
            calendar = 'standard'
        var['value'] = nc.date2num( var['value'], units=var['units'], calendar=calendar)    

    # Variable type
    vartype = _get_nc_type( var['value'], var['name'], classic )

    # Fill value, if any
    if 'fill_value' in var.keys():
        fill_value = var['fill_value']
    else:
        fill_value = None

    ### Progress towards packing variables as integers; 
    ### Appears to be working, but only minimal testing so far

    # Check if packing keywords are set
    if isDim:
        # Dimension variables should not be packed
        pack = False
    elif 'pack' in var.keys():
        # Use pack key if provided
        pack = var['pack']
    else:
        # Default to pack = False
        pack = False

    # Pack to integer data, if requested
    if pack:

        # Integer type for packed data
        if 'packtype' in var.keys():
            packtype = var['packtype']
        else:
            packtype = 'i2'
        vartype = packtype

        # Number of bits in the packed data type
        if packtype=='i2':
            n = 16
        elif packtype=='i1':
            n = 8
        else:
            raise ValueError(f'Packing to type {packtype:s} has not been implemented' )

        # Get scale factor, offset and fill value for packing
        scale_factor, add_offset, pack_fill_value = _infer_pack_scale_offset( var['value'][:], n )

        var['scale_factor'] = scale_factor
        var['add_offset']   = add_offset

        # Scale data into the integer range
        pack_value = _integer_pack_data( var['value'],
                                        scale_factor,
                                        add_offset,
                                        fill_value,
                                        pack_fill_value )

        # Rename
        fill_value = pack_fill_value
        var['value'] = pack_value

    else:
        scale_factor = None
        add_offset   = None
    var.pop('pack',None)
    var.pop('packtype',None)
    ###

    #*** Check whether the data type is allowed in classic data type

    # Create the variable
    ncVar = fid.createVariable( var['name'], vartype, dimIDs,
                             zlib=compress, complevel=2,
                             fill_value=fill_value )

    # Write variable values 
    ncVar[:] = var['value'][:]

    # These keys are not attributes, so remove them
    var.pop('name',None)
    var.pop('value',None)
    var.pop('unlimited',None)
    var.pop('dim_names',None)
    var.pop('fill_value',None)

    # Save the remaining attributes
    ncVar.setncatts(var)

    return ncVar
