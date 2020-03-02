# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:49:33 2015

@author: cdholmes
"""

import netCDF4 as nc

def get_nc_var(filename,varname):
    """ Read a variable from a netCDF file"""
    
    # Open file for reading
    ncfile = nc.Dataset(filename,'r')
    
    # Get the desired variable
    data = ncfile.variables[varname][:]
    
    # Close the file
    ncfile.close()
    
    return data
    
def get_nc_att(filename,varname,attname,glob=False):
    """ Read an attribute from a netCDF file"""
    
    # Open file for reading
    ncfile = nc.Dataset(filename,'r')
    
    # Get the desired attribute
    if (glob):
        data = ncfile.getncattr(attname)
    else:
        data = ncfile.variables[varname].getncattr(attname)
    
    # Close the file
    ncfile.close()
    
    return data
    
def get_nc_varnames(filename):
    """ Read variable names from a netCDF file"""
    
    # Open file for reading
    ncfile = nc.Dataset(filename,'r')
    
    # Get the desired variable
    data = list(ncfile.variables.keys())
    
    # Close the file
    ncfile.close()
    
    return data
    
def get_nc_attnames(filename,varname,glob=False):
    """ Read attributes from a netCDF file"""
    
    # Open file for reading
    ncfile = nc.Dataset(filename,'r')
    
    # Get the attribute names
    if (glob):
        data = ncfile.ncattrs()
    else:   
        data = ncfile.variables[varname].ncattrs()
    
    # Close the file
    ncfile.close()
    
    return data

def put_nc_var(filename,varname,value):
    """ Assign a new value to an existing variable"""
    
    # Open file for reading
    ncfile = nc.Dataset(filename,'w')
    
    # Set value
    ncfile.variables[varname][:] = value
    
    # Close the file
    ncfile.close()
     

def put_nc_att(filename,varname,attname,value,glob=False):
    """ Assign a new value to an existing attribute"""
    
    # Open file for reading
    ncfile = nc.Dataset(filename,'w')
    
    # Set attribute
    if (glob):
        ncfile.setncattr(attname,value)
    else:
        ncfile.variables[varname].setncattr(attname,value)
    
    # Close the file
    ncfile.close()

###############################################
### EVERYTHING AFTER HERE IN DEVELOPMENT
 
def create_geo_dim( var, fid, **kwargs ):
    
    try:
        assert ('name' in var), \
            'Var structure must contain "name" key: create_geo_dim'
        assert ('value' in var), \
            'Var structure must contain "value" key: create_geo_dim'
    except AssertionError:
        # Close file and exit
        fid.close()
        raise SystemExit
        
    size = len(var['value'])
    
    # If the unlimited tag is True, then set size=None to create unlimited dimension
    if ('unlimited' in var.keys()):
        if ( var['unlimited'] ):
            size = None

    ncDim = fid.createDimension( var['name'], size )
    
    ncVar = create_geo_var( var, fid, (var['name']), **kwargs )

    return ncDim, ncVar    

def get_nc_type( value, name='', classic=True ):
    
    # Variable type for numpy arrays or native python
    try:
        vartype = value.dtype.str
    except:
        vartype = type(value)
    
    # Remove brackets
    vartype = vartype.replace('<','').replace('>','')

    # Raise error if the type isn't allowed
    if (classic and (vartype in ['u1', 'u2', 'u4', 'u8', 'i8'])):
        raise TypeError( 'Variable {:s} cannot have type {:s} in netCDF classic files'.format(
            name, vartype ) )

    return vartype

def create_geo_var( var, fid, dimIDs, compress=True, classic=True, time=False, verbose=False ):
    
    try:
        assert ('name' in var), \
            'Var structure must contain "name" key: create_geo_var'
        assert ('value' in var), \
            'Var structure must contain "value" key: create_geo_var'
    except AssertionError:
        # Close file and exit
        fid.close()
        raise SystemExit

    # If this is a time variable and units are "<time units> since <date>", then convert
    if (time and (' since ' in var['units']) ):
        if ('calendar' in var.keys()):
            calendar = var['calendar']
        else:
            calendar = 'standard'
        var['value'] = nc.date2num( var['value'], units=var['units'], calendar=calendar)    
    
    # Variable type
    vartype = get_nc_type( var['value'], var['name'], classic )
  
    #*** Check whether the data type is allowed in classic data type
    
    # Create the variable
    ncVar = fid.createVariable( var['name'], vartype, dimIDs, 
                             zlib=compress, complevel=2 )

    # Write variable values 
    ncVar[:] = var['value'][:]

    # These keys are not attributes, so remove them
    var.pop('name',None)
    var.pop('value',None)
    var.pop('unlimited',None)
    var.pop('dim_names',None)
    
    # Save the remaining attributes
    ncVar.setncatts(var)
    
    return ncVar
    
def write_geo_nc(filename, variables,
                xDim=None, yDim=None, 
                zDim=None, zUnits=None, tDim=None, tUnits=None,
                globalAtt=None,
                classic=True, nc4=True, compress=True, 
                clobber=False, verbose=False ):
    """ Create a NetCDF file with geospatial data. Output file is CF-COARDS compliant""" 
    
    from datetime import datetime 
    
    # NetCDF file type
    if (nc4):
        if (classic):
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
    if (globalAtt is not None):
        f.setncatts(globalAtt)
       
    ### Define dimensions
    
    dimName = []
    dimList = []
    dimSize = []
    varList = []
    
    if (xDim is not None):
        if (not isinstance(xDim, dict)):
            xDim = {'name': 'lon',
                    'value': xDim,
                    'long_name': 'longitude',
                    'units': 'degrees_east'}    
        
        ncDim, ncVar = create_geo_dim( xDim, f, compress=compress, classic=classic, verbose=verbose )

        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(varList[-1][:]) )

    if (yDim is not None):
        if (not isinstance(yDim, dict)):
            yDim = {'name': 'lat',
                    'value': yDim,
                    'long_name': 'latitude',
                    'units': 'degrees_north'}

        ncDim, ncVar = create_geo_dim( yDim, f, compress=compress, classic=classic, verbose=verbose )
 
        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(varList[-1][:]) )
        
    if (zDim is not None):
        if (not isinstance(zDim, dict)):
            if (zUnits is not None):
                if (zUnits in ['m','km']): 
                    lname = 'altitude'
                elif (zUnits in ['Pa','hPa']):
                    lname = 'pressure'
                else:
                    lname = 'level'
            else:
                lname = 'level'
                zUnits = ''
                
            zDim = {'name': 'lev',
                    'value': zDim,
                    'long_name': lname,
                    'units': zUnits}

        ncDim, ncVar = create_geo_dim( zDim, f, compress=compress, classic=classic, verbose=verbose )

        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(varList[-1][:]) )
        
    if (tDim is not None):
        if (not isinstance(tDim, dict)):
            if (tUnits is None):
                tUnits = ''
            tDim = {'name': 'time',
                    'value': tDim,
                    'long_name': 'time',
                    'units': tUnits,
                    'unlimited': True }

        ncDim, ncVar = create_geo_dim( tDim, f, compress=compress, classic=classic, time=True, verbose=verbose )

        dimName.append( ncDim.name )
        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(varList[-1][:]) )

   ### Define variables

    for var in variables:

        if (type(var) is not dict):
            raise TypeError( 'All variables must be passed as dicts' )

        # Shape of the variable
        vShape = var['value'].shape

        # If dim_names is provided, otherwise use inference
        if ('dim_names' in var.keys() ):

            # Set dimensions based on provided names
            dID = var['dim_names']

            # Confirm that correct number of dimensions are provided
            if (len(dID) != len(vShape)):
                raise ValueError( 'Variable {:s} has dimenion {:d} and {:d} dim_names'.format( var['name'],
                                                                                               len(vShape),
                                                                                               len(dID) ) )
            # Shape of each named dimension
            dShape = tuple( dimSize[dimName.index(d)] for d in dID )

            # Confirm that the named dimensions match the shape of the variable
            if (dShape != vShape):
                raise ValueError( 'Shape of the dimensions [{:s}]=[{:s}] must match '.format( ','.join(dID),
                                                                                   ','.join([str(i) for i in dShape]) ) + 
                                  'the shape of variable {:s} which is [{:s}]'.format( var['name'],
                                                                                   ','.join([str(i) for i in vShape]) ) )
                        
        else:
            
            # Match dimensions of variable with defined dimensions based on shape
            dID = []
            for s in vShape:

                # Find the dimensions that match the variable dimension
                try:
                    i = dimSize.index(s)
                except ValueError:
                    # No dimensions match
                    raise ValueError( 'Cannot match dimensions for variable {:s}'.format( var['name'] ) )

                # List of the dimension names
                dID.append(dimName[i])

                
        # Create the variable
        ncVar = create_geo_var( var, f, dID, compress=compress, classic=classic, verbose=verbose )

        varList.append( ncVar )

    # Close the file
    f.close()
