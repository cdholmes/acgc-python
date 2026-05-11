import os
import datetime
import warnings
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm

from acgc import stats

# Path to this file
test_path = os.path.dirname(__file__)

# Load test data
co2 = sm.datasets.co2.load().data.co2
co2_fill = co2.resample('MS').mean().ffill()

def compare_results(result, expected, tol=1e-5):
    '''Compare STL results to expected values
    `result` should be a DecomposeResult and `expected` should be a DataFrame
    '''
    assert np.allclose(result.seasonal, expected['seasonal'], rtol=tol, equal_nan=True)
    assert np.allclose(result.trend, expected['trend'], rtol=tol, equal_nan=True)
    assert np.allclose(result.resid, expected['residual'], rtol=tol, equal_nan=True)

def test_stl_01():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2_fill.values, period=12, robust=False)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test01.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)
    
def test_stl_01b():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2_fill.values, np.arange(len(co2_fill)),
                                period=12, robust=False)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test01.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)

def test_stl_01c():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic('co2', np.arange(len(co2_fill)), data=pd.DataFrame(co2_fill),
                                period=12, robust=False)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test01.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)

def test_stl_02():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2_fill.values, period=12, robust=True)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test02.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)

def test_stl_03():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2_fill, period=pd.Timedelta(days=365.25), robust=False)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test03.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)

def test_stl_03b():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2_fill, co2_fill.index, 
                                period=pd.Timedelta(days=365.25), robust=False)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test03.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)

def test_stl_04():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2_fill, period=pd.Timedelta(days=365.25), robust=True)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test04.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)

def test_stl_05():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2.dropna(), period=pd.Timedelta(days=365.25), robust=False)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test05.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)

def test_stl_06():
    '''Test STL decomposition on CO2 data'''
    result = stats.STL_harmonic(co2.dropna(), period=pd.Timedelta(days=365.25), robust=True)
    
    expected = pd.read_csv(test_path+'/input/STLH_co2_test06.csv', 
                           comment='#',parse_dates=True)

    compare_results(result, expected, tol=1e-5)
