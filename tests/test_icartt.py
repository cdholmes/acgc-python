import os
import datetime
import warnings
from pathlib import Path

import pytest
import pandas as pd
from acgc import icartt

# Path to this file
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

    # compute differences between files
    differences = compare_files_excluding_lines(
        test_path+'/output/icartt_test_write.ict',
        test_path+'/input/icartt_write_reference.ict',
    )

    # Check for differences
    for dline in differences:
        lnum, v1, v2 = dline
        if lnum==7:
            # This line is expected to differ as it contains the current date
            continue
        assert v1 == v2, f"Line {lnum} differs: {v1} != {v2}"


def compare_files_excluding_lines(file1_path, file2_path, exclude_lines=[None], exclude_line_numbers=[None]):
    '''Compares two files line by line, excluding specified lines.

    Parameters
    ----------
    file1_path : str
        Path to the first file.
    file2_path : str
        Path to the second file.
    exclude_lines : list
        List of lines to exclude (e.g., ["line1", "line2"]).
    exclude_line_numbers : list
        List of line numbers to exclude (e.g., [1, 2]).

    Returns
    -------
    list
        List of tuples containing differing lines and line numbers.
    '''

    differences = []

    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        line_num = 0

        for line1, line2 in zip(file1, file2):
            line_num += 1

            if line_num in exclude_line_numbers:
                continue
            if line1.strip() in exclude_lines or line2.strip() in exclude_lines:
                continue

            if line1 != line2:
                differences.append((line_num, line1.strip(), line2.strip()))

    return differences