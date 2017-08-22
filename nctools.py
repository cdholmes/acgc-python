# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:49:33 2015

@author: cdholmes
"""

import netCDF4 as ncdf

def get_ncdf_var(filename,varname):
    """ Read a variable from a netCDF file"""
    
    # Open file for reading
    ncfile = ncdf.Dataset(filename,'r')
    
    # Get the desired variable
    data = ncfile.variables[varname][:]
    
    # Close the file
    ncfile.close()
    
    return data
    
def get_ncdf_att(filename,varname,attname,glob=False):
    """ Read an attribute from a netCDF file"""
    
    # Open file for reading
    ncfile = ncdf.Dataset(filename,'r')
    
    # Get the desired attribute
    if (glob):
        data = ncfile.getncattr(attname)
    else:
        data = ncfile.variables[varname].getncattr(attname)
    
    # Close the file
    ncfile.close()
    
    return data
    
def get_ncdf_varnames(filename):
    """ Read variable names from a netCDF file"""
    
    # Open file for reading
    ncfile = ncdf.Dataset(filename,'r')
    
    # Get the desired variable
    data = list(ncfile.variables.keys())
    
    # Close the file
    ncfile.close()
    
    return data
    
def get_ncdf_attnames(filename,varname,glob=False):
    """ Read attributes from a netCDF file"""
    
    # Open file for reading
    ncfile = ncdf.Dataset(filename,'r')
    
    # Get the attribute names
    if (glob):
        data = ncfile.ncattrs()
    else:   
        data = ncfile.variables[varname].ncattrs()
    
    # Close the file
    ncfile.close()
    
    return data

def put_ncdf_var(filename,varname,value):
    """ Assign a new value to an existing variable"""
    
    # Open file for reading
    ncfile = ncdf.Dataset(filename,'w')
    
    # Set value
    ncfile.variables[varname][:] = value
    
    # Close the file
    ncfile.close()
     

def put_ncdf_att(filename,varname,attname,value,glob=False):
    """ Assign a new value to an existing attribute"""
    
    # Open file for reading
    ncfile = ncdf.Dataset(filename,'w')
    
    # Set attribute
    if (glob):
        ncfile.setncattr(attname,value)
    else:
        ncfile.variables[varname].setncattr(attname,value)
    
    # Close the file
    ncfile.close()

###############################################
### EVERYTHING AFTER HERE IN DEVELOPMENT
 
def define_geo_dim( var, fid ):
    
    try:
        assert ('name' in var), \
            'Var structure must contain "name" key: define_geo_dim'
        assert ('value' in var), \
            'Var structure must contain "value" key: define_geo_dim'
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
    
    return ncDim    

def get_ncdf_type( value, classic=True ):
    
    # Variable type for numpy arrays or native python
    try:
        vartype = value.dtype.str
        
    except:
        vartype = type(value)
    
    
    vartype = vartype.replace('<','').replace('>','')

def define_geo_var( var, fid, dimIDs, compress=True, verbose=False ):
    
    try:
        assert ('name' in var), \
            'Var structure must contain "name" key: define_geo_var'
        assert ('value' in var), \
            'Var structure must contain "value" key: define_geo_var'
    except AssertionError:
        # Close file and exit
        fid.close()
        raise SystemExit

    # Format strings
#    fmt1 = 'Defined tracer    {:5d} = {:s}'
#    fmt3 = 'Defined attribute {:5d} = {:s} {:s}'
        
    
    #***** Enable other variable types
    #var['value'].dtype.str
    vartype = var['value'].dtype.str
    vartype = vartype.replace('<','').replace('>','')
    
    print( var['name'], var['value'].dtype, vartype, dimIDs )
    
    #*** Check whether the data type is allowed in classic data type
    
    # Create the variable
    ncVar = fid.createVariable( var['name'], 'i8', dimIDs, 
                             zlib=compress, complevel=2 )

    # These keys are not attributes, so remove them
    var.pop('name',None)
    var.pop('value',None)
    var.pop('unlimited',None)
    
    # Save the remaining attributes
    ncVar.setncatts(var)
    
    return ncVar
    
def write_geo_ncdf(filename, var1=None, xDim=None, yDim=None, 
                   zDim=None, zUnits=None, tDim=None, tUnits=None,
                   classic=True, nc4=True, compress=True, verbose=False, globalAtt=None ):
    """ Create a NetCDF file with geospatial data. Output file is CF-COARDS compliant""" 
    
    from datetime import datetime 
    
    ### Setup
    
    # NetCDF file type
    if (nc4):
        if (classic):
            ncfmt = 'NETCDF4_CLASSIC'
        else:
            ncfmt = 'NETCDF4'
    else:
        ncfmt = 'NETCDF3_64BIT_OFFSET'

    ### Create
    
    
    ### Open file for output
    
    f = ncdf.Dataset( filename, 'w', format=ncfmt  )
    f.Conventions = "COARDS/CF"
    f.History = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + \
        ' : Created by write_geo_ncdf (python)'
    
    # Write global attributes, if any
    if (globalAtt is not None):
        f.setncatts(globalAtt)
    

    
    ### Define dimensions
    
    dimList = []
    dimSize = []
    varList = []
    
    if (xDim is not None):
        if (not isinstance(xDim, dict)):
            xDim = {'name': 'lon',
                    'value': xDim,
                    'long_name': 'longitude',
                    'units': 'degrees_east'}    
        
        ncDim = define_geo_dim( xDim, f )
        ncVar = define_geo_var( xDim, f, (xDim['name']), compress, verbose=verbose )

        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(varList[-1][:]) )

    if (yDim is not None):
        if (not isinstance(yDim, dict)):
            yDim = {'name': 'lat',
                    'value': yDim,
                    'long_name': 'latitude',
                    'units': 'degrees_north'}

        ncDim = define_geo_dim( yDim, f )
        ncVar = define_geo_var( yDim, f, (yDim['name']), compress, verbose=verbose )

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

        ncDim = define_geo_dim( zDim, f )
        ncVar = define_geo_var( zDim, f, (zDim['name']), compress, verbose=verbose )

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

        ncDim = define_geo_dim( tDim, f )
        ncVar = define_geo_var( tDim, f, (tDim['name']), compress, verbose=verbose )

        dimList.append( ncDim )
        varList.append( ncVar )
        dimSize.append( len(varList[-1][:]) )

    # Close the file
    f.close()