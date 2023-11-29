#!/usr/bin/env python3
'''tapply'''

import numpy as np

__all__ = ['tapply']

def tapply( array, group, f ):
    '''Apply user-specified function to array elements within each group

    Reproduces tapply function in R
    
    Parameters
    ----------
    array : ndarray or list of numeric type
        values that will be grouped
    group : list or ndarray
        group ids for the elements of array, must have same size as array
    f : function handle
        function that will be applied to each group, e.g. np.mean

    Returns
    -------
    result : ndarray
        value of function applied to each group. 
        The number of elements in result equals the number of unique elements of the group argument
    groupvalues : list
        group ids corresponding to the elements of result
    '''

    # Ensure that arguments have the same size
    assert array.size == group.size, "Arguments array and group must have the same size"

    # Find the unique GROUP values
    groupvalues = np.unique( group )

    # Make an array with the type as the input array
    result = np.empty( groupvalues.size, type(array[0]) )

    # Loop over the number of unique values
    for i in range(groupvalues.size):

        # Find which elements share the same value
        index = np.where( group == groupvalues[i] )

        # Apply teh given function to the common elements
        result[i] = f(array[index])

    # return
    return result, groupvalues
