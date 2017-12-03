# coding: utf-8
import pandas as pd
import numpy as np
from classifyintents import (normalise, date_features, string_len, 
        string_capsratio, string_nexcl, clean_date)

class TestFeatureGenerators(object):
    
    # Would be run before each class method

    #def setup(self)

    # To run after each class method

    #def teardown(self):
    
    # To run when the class is instantiated.

    @classmethod
    def setup_class(self):

        print('Testing functions which clean and create new features')

        self.test_case = pd.Series([
            '-', ' ', 'none, ', 'What is my purpose?', 'You pass butter!'
        ])

        # Define the output dataframe expected from date_features

    # No teardown required
    #@classmethod
    #def teardown_class(cls):
    #    print ("teardown_class() after any methods in this class")

    # Test whether it works on a single instance

    def test_normalise_works_as_expected(self):

        # This is based on normalising range(10)

        expected_normalise = [
            -1.5666989, -1.21854359, -0.87038828, -0.52223297, -0.17407766,
            0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989
        ]

        assert all(np.isclose(expected_normalise, normalise(range(10))))


    def test_string_len__works_as_expected(self):

        expected_str_len = pd.Series([
            -0.368421, -0.421053, -0.210526, 0.578947, 0.421053
            ])

        actual_str_len = string_len(self.test_case)

        print(actual_str_len)

        assert all(np.isclose(expected_str_len, actual_str_len))


    def test_string_capsratio_works_as_expected(self):

        expected_str_capsratio = pd.Series([
                0.000000, 0.000000, 0.000000, 0.052632, 0.062500
            ])

        actual_str_capsratio = pd.Series([string_capsratio(i) for i in self.test_case])

        assert all(np.isclose(expected_str_capsratio, actual_str_capsratio))


    def test_string_nexcl_works_as_expected(self):

        expected_str_nexcl = pd.Series([0, 0, 0, 0, 0.0625])

        actual_str_nexcl = pd.Series([string_nexcl(i) for i in self.test_case])

        assert all(np.isclose(expected_str_nexcl, actual_str_nexcl))


    def test_clean_date_works_as_expected(self):

        date = '2016-01-01 00:00:00'

        assert isinstance(clean_date(date), pd.Timestamp)


    def test_date_features_works_as_expected(self):

        test_dates = pd.Series([
            '2015-01-01 00:00:00',
            '2016-01-01 00:00:00',
            '2017-01-01 00:00:00',
            '2018-01-01 00:00:00'
            ])

        d = {
            'day': [1] * 4,
            'month': [1] * 4,
            'week': [1, 53, 52, 1],
            'weekday': [3, 4, 6, 0],
            'year': [2015, 2016, 2017, 2018]
            }

        expected_date_features = pd.DataFrame(data=d)
        actual_date_features = date_features(test_dates)

        assert expected_date_features.equals(actual_date_features)
