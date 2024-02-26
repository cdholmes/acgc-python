import numpy as np
import pandas as pd
import pytest
from acgc import solar

timeStr = '2010-03-25 00:00'
time = pd.Timestamp(timeStr)
timeUTC = time.tz_localize('UTC')
timeAK = timeUTC.tz_convert('US/Alaska')
lat_brw = 71.323
lon_brw = -156.611

def test_solar_position():
    # result = solar.solar_position(pd.Timestamp('2010-03-25 00UTC-5'))
    # assert result == pytest.approx((1.2910330960129097, 2.9797253248193085, -6.433732038619937, 0.9968690672800375, pd.Timestamp('2010-03-24 19:00:00')))
    
    # Result should be the same regardless of how time is specified
    result    = solar.solar_position(time)
    resultUTC = solar.solar_position(timeUTC)
    resultAK  = solar.solar_position(timeAK)
    resultStr = solar.solar_position(timeStr)

    assert result == pytest.approx((1.684679877456496,
                                    3.8899666546310896,
                                    -6.1320190397270915,
                                    0.9971526179417078,
                                    pd.Timestamp('2010-03-25 00:00:00'))), 'Incorrect solar position'
    assert result == resultUTC, 'Results change when UTC specified explicitly'
    assert result == resultAK, 'Results change when time zone specified explicitly'
    assert result == resultStr, 'Results change when time string provided'

# def test_sunrise_time():
#     assert 
