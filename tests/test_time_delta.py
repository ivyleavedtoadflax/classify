# Give the correct context for the tests relative to the module itself

import nose
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import pandas as pd
from classify import *
import numpy as np

a = pd.to_datetime('2016-01-02')
b = pd.to_datetime('2016-01-01')

def test_time_delta_output():

    nose.tools.assert_equal(
            classify.time_delta(a,b),
            86400
            )

def test_time_delta_type():

    nose.tools.assert_true(
            np.isscalar(classify.time_delta(a,b))
            )
