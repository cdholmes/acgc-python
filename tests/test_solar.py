import datetime
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

def test_time_types():
    '''Run calculations with all expected input types for time variable'''

    for t in [ # Scalar type types, including time zones
              time,
              timeStr,
              timeUTC,
              np.datetime64(time),
              time.to_pydatetime(),
              # Array time types
              [time,time],
              pd.DatetimeIndex([time,time]),
              pd.Series([time,time]),
              # Arrays with time zone
              pd.DatetimeIndex([time,time]).tz_localize('US/Eastern'),
              pd.Series([time,time]).dt.tz_localize('US/Eastern')
              ]:
        solar.insolation_toa(lat_brw,lon_brw,t)
        solar.solar_azimuth_angle(lat_brw,lon_brw,t)
        solar.solar_zenith_angle(lat_brw,lon_brw,t)
        solar.solar_elevation_angle(lat_brw,lon_brw,t)
        solar.sunrise_time(lat_brw,lon_brw,t)
        solar.sunset_time(lat_brw,lon_brw,t)
        solar.solar_noon(lat_brw,lon_brw,t)
        solar.day_length(lat_brw,lon_brw,t)
        solar.solar_declination(t)
        solar.solar_latitude(t)
        solar.solar_longitude(t)
        solar.solar_hour_angle(lon_brw,t)
        solar.equation_of_time(t)
        solar.solar_position(t)
        solar.sun_times(lat_brw,lon_brw,t)
        solar.horizon_zenith_angle(0)
        solar.refraction_angle(0)

def test_solar_position():
    
    # Result should be the same regardless of how time is specified
    result    = solar.solar_position(time)
    resultUTC = solar.solar_position(timeUTC)
    resultAK  = solar.solar_position(timeAK)
    resultStr = solar.solar_position(timeStr)

    # Expected answer
    ans = (1.684679877456496,
            3.8899666546310896,
            -6.1320190397270915,
            0.9971526179417078,
            pd.Timestamp('2010-03-25 00:00:00'))
    # Expected answer with fast calculation
    ans_dec_fast = 1.879
    ans_eot_fast = -6.065

    assert result == pytest.approx(ans), 'Incorrect solar position'
    assert result == resultUTC, 'Results change when UTC specified explicitly'
    assert result == resultAK, 'Results change when time zone specified explicitly'
    assert result == resultStr, 'Results change when time string provided'

    assert solar.solar_declination( time ) == pytest.approx(ans[0]), 'solar_declination error'
    assert solar.equation_of_time( time ) == pytest.approx(ans[2]), 'equation_of_time error'
    assert solar.equation_of_time( time, degrees=True ) == pytest.approx(ans[2]/4), \
        'equation_of_time degrees error'

    # Fast declination and eot calculations
    assert solar.solar_declination( time, fast=True ) == pytest.approx(ans_dec_fast,0.001), \
        'fast solar_declination error'
    assert solar.equation_of_time( time, fast=True ) == pytest.approx(ans_eot_fast,0.001), \
        'fast equation_of_time error'

def test_sun_times():
  
    result = solar.sun_times(lat_brw,lon_brw,time)
    resultAK = solar.sun_times(lat_brw,lon_brw,time,'US/Alaska')

    # Expected answer
    ans = (pd.Timestamp('2010-03-25 16:02:08.205409628'),
            pd.Timestamp('2010-03-26 05:03:00.916875138'),
            pd.Timedelta('0 days 13:00:52.711465510'),
            pd.Timestamp('2010-03-25 22:32:34.561142383'),
            pd.Timestamp('2010-03-25 00:00:00'))
    # Answer in AK time
    ansAK = (pd.Timestamp('2010-03-25 08:02:08.205409628-0800', tz='US/Alaska'), 
             pd.Timestamp('2010-03-25 21:03:00.916875138-0800', tz='US/Alaska'), 
             pd.Timedelta('0 days 13:00:52.711465510'), 
             pd.Timestamp('2010-03-25 14:32:34.561142383-0800', tz='US/Alaska'),
             pd.Timestamp('2010-03-25 00:00:00'))

    assert result == pytest.approx(ans), 'Sun times do not equal expected values'
    assert resultAK == pytest.approx(ansAK), 'Sun times incorrect with time zone'

    assert solar.sunrise_time(lat_brw,lon_brw,time) \
        == pytest.approx(ans[0]), 'Sunrise time error'
    assert solar.sunset_time(lat_brw,lon_brw,time) \
        == pytest.approx(ans[1]), 'Sunset time error'
    assert solar.day_length(lat_brw,lon_brw,time) \
        == pytest.approx(ans[2]), 'Day length error'
    assert solar.solar_noon(lat_brw,lon_brw,time) \
        == pytest.approx(ans[3]), 'Solar noon error'