# # NAME:
#        TAPPLY
#
# PURPOSE:
#       TAPPLY reproduces the behavior of the R function tapply
#       TAPPLY applies the specified function, to all elements of ARRAY
#       which have the same GROUP value. The function returns an array with
#       as many elements as unique values within GROUP.
# 
# CATEGORY:
#
# CALLING SEQUENCE:
#        RESULT = TAPPLY( ARRAY, GROUP, FUNCTIONSTR [, KEYWORDS] ) 
#
# INPUTS:
#       ARRAY       - array of arbitrary size
#       GROUP       - array of same dimension as ARRAY while classifies
#                     elements of ARRAY
#       FUNCTIONSTR - string naming an IDL function to apply to ARRAY
#       _EXTRA      - any additional keywords are passed to the
#                     specified function
#
# KEYWORD PARAMETERS:
#
# OUTPUTS:
#       RESULT      - array resulting from the function application
#       GROUPVALUES - keyword array containing the unique values of GROUP, in
#                     the same order as RESULT
#
# SUBROUTINES:
#
# REQUIREMENTS:
#
# NOTES:
#
# EXAMPLE:
#        # CALCULATE THE AVERAGE DIURNAL CYCLE
#        import numpy as np
#        ntimes = 1024
#        time   = np.arange(ntimes)
#        # Sinusoidal signal with 24-h period, plus noise
#        signal = np.cos( time * 2 * np.pi / 24. ) + np.random.normal(size=ntimes)
#
#        # Calculate the diurnal mean
#        from tapply import tapply
#        diurnalmean, hour = tapply( signal, np.mod( time, 24 ), np.mean )
#
#        # plot the mean cycle
#        import matplotlib.pyplot as plt
#        plt.plot( hour, diurnalmean )
#        plt.xlabel('hour')
#        plt.ylabel('mean signal')
#
# MODIFICATION HISTORY:
#        cdh, 16 Sep 2014: VERSION 1.00 adapted from IDL
#
#-
# Copyright (C) 2014, Christopher Holmes, UC Irvine
# This software is provided as is without any warranty whatsoever.
# It may be freely used, copied or distributed for non-commercial
# purposes.  This copyright notice must be kept with any copy of
# this software. If this software shall be used commercially or
# sold as part of a larger package, please contact the author.
# Bugs and comments should be directed to cdholmes@post.harvard.edu
# with subject "python routine tapply"
#-----------------------------------------------------------------------

    
def tapply( array, group, f ):

    import numpy as np

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
