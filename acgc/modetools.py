#!/usr/bin/env python3
'''Graphical display of eigenmodes'''

import numpy as np
import matplotlib.pyplot as plt

def show_modes( labels, eigval, eigvec, ax=None ):
    '''Plot figure showing eigenmodes
    
    This program assumes that all modes will be displayed, so it is expected that
    len(labels) = len(eigval) = nmodes and eigvec has shape nmodes x nmodes. 

    Arguments
    ---------
    labels : list, str
        names for the component variables decomposed into modes
    eigval : list or array, float
        eigenvalues
    eigvec : 2D array, float
        eigenvector matrix. Vectors should be the columns of eigvec
    ax : subplot axis (optional)
        location where graph will be displayed. If not provided, then a new axis will be created.        
    '''

    # Number of modes
    nmodes = len(eigval)

    # Ensure number of labels equals number of modes
    if len(labels) != nmodes:
        raise ValueError( 'Provide one label per eigenmode')

    # Create new axis if one is not passed
    if ax is None:
        ax = plt.axes()

    # Sort the modes from largest to smallest eigenvalue
    idx = np.argsort( np.abs(eigval) )[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]

    # Plot the modes
    for i in range(nmodes):
        # Vertical gray line for reference
        ax.plot( [i*2,i*2], [0,nmodes-1], color='gray')
        # Plot mode i
        ax.plot( i*2 + eigvec[::-1,i],  np.arange(nmodes), 'o-' )

    # xaxis labels
    xlabel = [ '{:#.2g}\n({:#.2g})'.format(x,np.abs(1/x)) for x in eigval]

    # Set ticks and labels
    ax.set( yticks=np.arange(nmodes),
            yticklabels=labels[::-1],
            xticks=np.arange(nmodes)*2,
            xticklabels=xlabel,
            ylabel='Reservoir',
            xlabel='Modes: Eigenvalue\n(Time scale)')

    plt.tight_layout()
