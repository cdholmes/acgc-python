import warnings
import numpy as np
import pytest
import acgc.stats as mystats

# Simple test arrays
x = np.arange(5)
w = np.abs(np.sin(x))

def check_raises_error( ErrorType, func, *args, **kwargs ):
    '''Confirm that the function DOES raise expected error
    Args:
        ErrorType : the error that SHOULD occur
        func : function to be evaluated
        *args and **kwargs passed to func'''
    with pytest.raises(ErrorType):
        func(*args,**kwargs)

def test_wmean():
    '''wmean accuracy'''

    assert mystats.wmean(x,w)==pytest.approx(2.3070399831334814)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        assert mystats.wmean(x**2,robust=True)==3.5

def test_wstd():
    '''wstd accuracy'''

    assert mystats.wstd(x**2,w)==pytest.approx(7.100903221127446)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        assert mystats.wstd(x**2,robust=True)==pytest.approx(4.041451884327381)

def test_ValueErrors():
    '''The following function calls SHOULD generate errors'''

    with pytest.raises(ValueError):
        mystats.wmean(x)

    with pytest.raises(ValueError):
        mystats.wstd(x)

    with pytest.raises(ValueError):
        mystats.wvar(x)
