import datetime
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from acgc import solar

timeStr = '2010-03-25 00:00'
time = pd.Timestamp(timeStr)
timeUTC = time.tz_localize('UTC')
timeAK = timeUTC.tz_convert('US/Alaska')
lat_brw = 71.323
lon_brw = -156.611

xlat = xr.DataArray([lat_brw,lat_brw+1],dims=['lat'],coords={'lat':('lat',[0,1])})
xlon = xr.DataArray([lon_brw,lon_brw+1],dims=['lon'],coords={'lon':('lon',[0,1])})
xtime = xr.DataArray([time,time+pd.Timedelta('1h')],dims=['time'],coords={'time':('time',[0,1])})

def test_param_types():
    '''Run calculations with all expected input types'''

    # Ignore expected warnings about time not including timezone
    warnings.simplefilter('ignore',RuntimeWarning)

    solar.horizon_zenith_angle(0)
    solar.refraction_angle(0)

    try:
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
                pd.Series([time,time]).dt.tz_localize('US/Eastern')]:

            # Functions of time only
            solar.solar_declination(t)
            solar.solar_latitude(t)
            solar.solar_longitude(t)
            solar.equation_of_time(t)
            solar.solar_position(t)

            for lat in [lat_brw,
                        np.array([lat_brw,lat_brw+1])]:
                for lon in [lon_brw,
                            np.array([lon_brw,lon_brw+1])]:

                    # Functions of lat, lon, time
                    solar.insolation_toa(lat,lon,t)
                    solar.solar_azimuth_angle(lat,lon,t)
                    solar.solar_zenith_angle(lat,lon,t)
                    solar.solar_elevation_angle(lat,lon,t)
                    solar.solar_hour_angle(lon,t)

                    # Functions with time zone
                    for tz_out in [None,'UTC','US/Eastern']:
                        solar.sun_times(lat,lon,t,tz_out)
                        solar.sunrise_time(lat,lon,t,tz_out)
                        solar.sunset_time(lat,lon,t,tz_out)
                        solar.solar_noon(lat,lon,t,tz_out)
                        solar.day_length(lat,lon,t,tz_out)

        # DataArrays should also work as long as the other parameters are scalar
        for t in [ # Scalar type types, including time zones
                time,
                timeStr,
                timeUTC,
                np.datetime64(time),
                time.to_pydatetime(),
                xtime]:

            # Functions of time only
            solar.solar_declination(t)
            solar.solar_latitude(t)
            solar.solar_longitude(t)
            solar.equation_of_time(t)
            solar.solar_position(t)

            for lat in [lat_brw,
                        xlat]:
                for lon in [lon_brw,
                            xlon]:

                    # Functions of lat, lon, time
                    solar.insolation_toa(lat,lon,t)
                    solar.solar_azimuth_angle(lat,lon,t)
                    solar.solar_zenith_angle(lat,lon,t)
                    solar.solar_elevation_angle(lat,lon,t)
                    solar.solar_hour_angle(lon,t)

                    # Functions with time zone
                    for tz_out in [None,'UTC','US/Eastern']:
                        solar.sun_times(lat,lon,t,tz_out)
                        solar.sunrise_time(lat,lon,t,tz_out)
                        solar.sunset_time(lat,lon,t,tz_out)
                        solar.solar_noon(lat,lon,t,tz_out)
                        solar.day_length(lat,lon,t,tz_out)

    except:
        msg  =  'Error triggered by \n'
        msg += f'Types: lat={type(lat)} lon={type(lon)} time={type(t)} tz={tz_out}'
        raise RuntimeError(msg)

def test_solar_position():
    '''Check values for solar_position'''

    # Ignore expected warnings about time not including timezone
    warnings.simplefilter('ignore',RuntimeWarning)

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
    '''Check values for sun_times'''

    # Ignore expected warnings about time not including timezone
    warnings.simplefilter('ignore',RuntimeWarning)

    result = solar.sun_times(lat_brw,lon_brw,time)
    resultAK = solar.sun_times(lat_brw,lon_brw,time,'US/Alaska')
    resultAK2 = solar.sun_times(lat_brw,lon_brw,time.tz_localize('UTC'),'US/Alaska')
    resultAK3 = solar.sun_times(lat_brw,lon_brw,time.tz_localize('UTC').tz_convert('US/Eastern'),'US/Alaska')

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

    assert result == pytest.approx(ans), \
        'Sun times do not equal expected values'
    assert resultAK == pytest.approx(ansAK), \
        'Sun times incorrect with time zone'
    assert resultAK2 == pytest.approx(ansAK), \
        'Result incorrect with input timezone'
    assert resultAK3 == pytest.approx(ansAK), \
        'Result incorrect with input timezone'

    assert solar.sunrise_time(lat_brw,lon_brw,time) \
        == pytest.approx(ans[0]), 'Sunrise time error'
    assert solar.sunset_time(lat_brw,lon_brw,time) \
        == pytest.approx(ans[1]), 'Sunset time error'
    assert solar.day_length(lat_brw,lon_brw,time) \
        == pytest.approx(ans[2]), 'Day length error'
    assert solar.solar_noon(lat_brw,lon_brw,time) \
        == pytest.approx(ans[3]), 'Solar noon error'
