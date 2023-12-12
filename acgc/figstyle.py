#!/usr/bin/env python3
'''Style settings for matplotlib, for publication-ready figures

Inspired by Proplot'''

import os
import warnings
import matplotlib as mpl
import matplotlib.style as mstyle
import matplotlib.font_manager as mfonts
if 'inline' in mpl.get_backend():
    import matplotlib_inline

# Path to this module
PATH = os.path.dirname(__file__)

def load_style(grid=True,gridaxis='both'):
    '''Load style sheet
    
    Parameters
    ----------
    grid : bool (default=True)
        turn grid lines on (True) or off (False)
    gridaxis : str (default='both')
        specifies which axes should have grid lines: 'x', 'y', 'both'
    '''
    mstyle.use(os.path.join(PATH,'acgc.mplstyle'))

    # Turn grid on or off
    if grid:
        grid_on(axis=gridaxis)

    # Use high quality for inline images; Only use this if the inline backend is active
    # 'png' is default, 'svg' is also good
    if 'inline' in mpl.get_backend():
        matplotlib_inline.backend_inline.set_matplotlib_formats('retina')


def grid_off():
    '''Turn off grid lines'''
    mpl.rcParams['axes.grid'] = False
def grid_on(axis='both'):
    '''Turn on grid lines
    
    Parameter
    ---------
    axis : str (default='both')
        specifies which axes should have grid lines: 'x', 'y', 'both' 
    '''
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.axis'] = axis

def load_fonts():
    '''Load fonts contained in the ./fonts subdirectory'''

    # User fonts
    fonts_pylib = mfonts.findSystemFonts(os.path.join(PATH,'fonts'))

    # Cached fonts
    fonts_cached = mfonts.fontManager.ttflist
    fonts_cached_paths = [ font.fname for font in fonts_cached ]

    # Add fonts that aren't already installed
    rebuild = False
    for font in fonts_pylib:
        if font not in fonts_cached_paths:
            if rebuild == False:
                # Issue warning, first time only
                warnings.warn('Rebuilding font cache. This can take time.')
            rebuild = True    
            mfonts.fontManager.addfont(font)

    # Save font cache
    if rebuild:
        cache = os.path.join(
                        mpl.get_cachedir(),
                        f'fontlist-v{mfonts.FontManager.__version__}.json'
                    )
        mfonts.json_dump(mfonts.fontManager, cache)

###
load_fonts()
load_style()