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
    