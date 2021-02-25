#!/usr/bin/env python3

# Graphically display modes
import numpy as np
import matplotlib.pyplot as plt

def show_modes( labels, eigval, eigvec, ax=None ):
    
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