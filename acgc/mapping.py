#!/usr/bin/env python3
'''Package of mapping functions

'''

import numpy as np
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo

pi180 = np.pi/180

def great_circle(*args,data=None,radius=None,flattening=None):
    '''Great circle distance between two points on Earth

    Usage: 
    distance = great_circle( start_lon, start_lat, end_lon, end_lat )
    distance = great_circle( start_points, end_points )

    Parameters
    ----------
    *args : array_likes or str
        Longitude and latitude of the start and end points, degrees
        Coordinates can be passed as four 1D arrays (n,) or two 2D arrays (n,2)
        If four arrays, they should be `start_lon`, `start_lat`, `end_lon`, `end_lat` 
        If two arrays, they should be (n,2) shape with longitude as the first column
        If strings, the `data` keyword must be used and args are interpreted as key names
    data : dict_like, optional
        If provided, *args should be strings that are keys to `data`. 
    radius : float, optional
        radius of the sphere in meters. If None, WGS84 will be used.
    flattening : float, optional
        flattening of the ellipsoid. Use 0 for a sphere. If None, WGS84 will be used.

    Returns
    -------
    distance : ndarray
        distance between points, m
    '''

    # Convert tuple -> list
    args = list(args)

    # Check if any args are strings
    if np.any( [isinstance(arg,str) for arg in args] ):
        if data is None:
            raise ValueError('`data` keyword must be used when `*args` contain strings')

        # Get the value from `data`
        for i,item in enumerate(args):
            if isinstance(item,str):
                args[i] = data[item]

    # Number of arguments
    nargs = len(args)

    if nargs == 4:
        # *args contain lon, lat values; broadcast them to same shape
        start_lon, start_lat = np.broadcast_arrays( args[0], args[1] )
        end_lon, end_lat     = np.broadcast_arrays( args[2], args[3] )

        # Stack to (n,2) needed for Geodesic
        start_points = np.stack( [start_lon, start_lat], axis=-1 )
        end_points   = np.stack( [end_lon,   end_lat],   axis=-1 )

    elif nargs == 2:
        # *args contain (lon,lat) arrays
        start_points = args[0]
        end_points = args[1]

    else:
        raise ValueError(f'Function takes either 2 or 4 arguments but {nargs} were passed.')

    # # Distance on a sphere. Cartopy is fast enough that there is no reason to use this
    #
    # # Lat and longitude endpoints, degrees -> radians
    # lat0 = start[:,0] * pi180
    # lon0 = start[:,1] * pi180
    # lat1 = end[:,0] * pi180
    # lon1 = end[:,1] * pi180
    # dlon = lon1 - lon0
    # # Haversine formula for distance on a sphere
    # # dist = 2 * radius * np.arcsin(np.sqrt(
    # #     np.sin( (lat1-lat0)/2 )**2
    # #     + np.cos(lat0) * np.cos(lat1)
    # #       * np.sin( (lon1-lon0)/2 )**2 ) )
    # # Equivalent formula with less roundoff error for antipodal points
    # dist = radius * np.arctan2(
    #     np.sqrt( (np.cos(lat1)*np.sin(dlon))**2
    #             + (np.cos(lat0)*np.sin(lat1)
    #                 - np.sin(lat0)*np.cos(lat1)*np.cos(dlon))**2 ),
    #     np.sin(lat0)*np.sin(lat1) + np.cos(lat0)*np.cos(lat1)*np.cos(dlon)
    # )

    if radius is None:
        # Semi-major radius of Earth, WGS84, m
        radius = 6378137.
    if flattening is None:
        # Flattening of ellipsoid, WGS84
        flattening = 1/298.257223563

    geoid = cgeo.Geodesic(radius,flattening)

    # Calculate the line from all trajectory points to the target
    # The start and end points should be in (lon,lat) order
    vec= np.asarray( geoid.inverse( start_points, end_points ) )

    # Distance, m
    dist = vec[:,0]

    return dist

def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    From StackOverflow
    https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot

    Parameters
    ----------
    ax:              
        CartoPy axes
    location:        
        Position of left-side of bar in axes coordinates.
    length:          
        Geodesic length of the scale bar.
    metres_per_unit: default=1000
        Number of metres in the given unit.
    unit_name: str, default='km'       
        Name of the given unit.
    tol: float, default=0.01             
        Allowed relative error in length of bar
    angle: float           
        Anti-clockwise rotation of the bar.
    color: str, default='black'           
        Color of the bar and text.
    linewidth: float       
        Same argument as for plot.
    text_offset: float, default=0.005     
        Perpendicular offset for text in axes coordinates.
    ha: str or float [0-1], default='center'              
        Horizontal alignment.
    va: str or float [0-1], default='bottom'              
        Vertical alignment.
    plot_kwargs: dict
        Keyword arguments for plot, overridden by **kwargs.
    text_kwargs: dict
        Keyword arguments for text, overridden by **kwargs.
    **kwargs:        
        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)
