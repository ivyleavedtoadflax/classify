import nose.tools as nt
import os, sys

# There is probably a cleaner way of doing this

sys.path.insert(0, os.path.abspath('../..'))
import pandas as pd
from classify import *
import numpy as np

# Test whether it works on a single instance

a = pd.to_datetime('2016-01-02')
b = pd.to_datetime('2016-01-01')

def test_time_delta_output():

    nt.assert_equal(
            classify.time_delta(a,b)[1],
            86400
            )

def test_time_delta_type():
    nt.assert_true(
            isinstance(classify.time_delta(a,b)[1], np.int64)
            )

# Test whether it works on a list

a = pd.Series(pd.date_range('20160102', periods=4))
b = pd.Series(pd.date_range('20160101', periods=4))

def test_time_delta_list():

    nt.assert_equal(
            classify.time_delta(a,b).tolist(),
            [86400,86400,86400,86400]
            )
