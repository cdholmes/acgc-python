import numpy as np
import pytest
import acgc.stats as mystats

# Simple test arrays
x = np.arange(5) + 1.
y = x**2
w = np.abs(np.sin(x))

def test_BivariateStatistics():
    '''Check BivariateStatistics'''

    result  = mystats.BivariateStatistics(x,y)
    result2 = mystats.BivariateStatistics('x','y',data={'x':x,'y':y})

    assert result.intercept() == result2.intercept()

    assert result.r2==pytest.approx(0.9625668449197863)

    assert result.slope(method='sma',intercept=True)==pytest.approx(6.115553940568262)
