#!/usr/bin/env python3
'''Style settings for matplotlib, for publication-ready figures

Inspired by Proplot'''

import os
import warnings
import matplotlib as mpl
import matplotlib.style as mstyle
import matplotlib.font_manager as mfonts
import matplotlib_inline

# Path to this module
path = os.path.dirname(__file__)

def load_style(grid=True):
    '''Load style sheet
    
    Parameters
    ----------
    grid : bool (default=True)
        turn grid lines on (True) or off (False)
    '''
    mstyle.use(os.path.join(path,'acgc.mplstyle'))

    # Turn grid on or off
    mpl.rcParams['axes.grid'] = grid

    # Use high quality for inline images
    # 'png' is default, 'svg' is also good
    matplotlib_inline.backend_inline.set_matplotlib_formats('retina')


def grid_off():
    '''Turn off grid lines'''
    mpl.rcParams['axes.grid'] = False
def grid_on():
    '''Turn on grid lines'''
    mpl.rcParams['axes.grid'] = True

def load_fonts():
    '''Load fonts contained in the ./fonts subdirectory'''

    # User fonts
    fonts_pylib = mfonts.findSystemFonts(os.path.join(path,'fonts'))

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