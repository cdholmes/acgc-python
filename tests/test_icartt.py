import os
import datetime
import warnings
from pathlib import Path

import pytest
from acgc import icartt
import pandas as pd

test_path = os.path.dirname(__file__)

def test_read_icartt():
    '''Test reading ICARTT file'''
    data1 = icartt.read_icartt( test_path+'/input/icartt_example.ict' )    
    data2 = icartt.read_icartt( test_path+'/input/icartt_write_reference.ict')

def test_write_icartt():
    '''Write a simple ICARTT file, check against expected result'''

    df = pd.DataFrame( [[1,0,30],
                    [2,10,29],
                    [3,20,27],
                    [4,30,25]],
                    columns=['Time_Start','Alt','Temp'])

    metadata = dict(
        INDEPENDENT_VARIABLE_DEFINITION = 
            {'Time_Start':'seconds, time, measurement time in seconds after takeoff'},
        DEPENDENT_VARIABLE_DEFINITION = 
            {'Alt':'m, altitude, altitude above ground level',
             'Temp':'C, temperature, air temperature in Celsius'},
        PI_NAME = 'Jane Doe',
        ORGANIZATION_NAME = 'NASA',
        SOURCE_DESCRIPTION = 'Invented Instrument',
        MISSION_NAME = 'FIREX-AQ',
        SPECIAL_COMMENTS = ['Special comments are optional and can be omitted.',
                        'If used, they should be a list of one or more strings'],
        PI_CONTACT_INFO = 'jdoe@email.com or postal address',
        PLATFORM = 'NASA DC-8',
        LOCATION = 'Boise, ID, USA',
        ASSOCIATED_DATA = 'N/A',
        INSTRUMENT_INFO = 'N/A',
        DATA_INFO = 'N/A',
        UNCERTAINTY = r'10% uncertainty in all values',
        ULOD_FLAG = '-7777',
        ULOD_VALUE = 'N/A',
        LLOD_FLAG = '-8888',
        LLOD_VALUE = 'N/A',
        DM_CONTACT_INFO = 'Alice, data manager, alice@email.com',
        STIPULATIONS_ON_USE = 'FIREX-AQ Data Use Policy',
        PROJECT_INFO = 'FIREX-AQ 2019, https://project.com',
        OTHER_COMMENTS = 'One line of comments',
        REVISION = 'R1',
        REVISION_COMMENTS = ['R0: Initial data',
                            'R1: One string per revision'],
        measurement_start_date = pd.Timestamp('2020-01-30 10:20')
        )
    
    icartt.write_icartt( test_path+'/output/icartt_test_write.ict', df, metadata )

    # Should compare files, but need to exclude line 6, which contains
    # the date when the file was written