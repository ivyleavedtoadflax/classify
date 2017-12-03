import os, sys
import nose.tools as nt
import pandas as pd
from classifyintents import *

class TestCleanRaw:

    # Would be run before each class method

    #def setup(self):

    # To run after each class method

    #def teardown(self):
    # To run when the class is instantiated.

    @classmethod
    def setup_class(self):

        print('Loading test_data/raw_test_data.csv into survey class')
        print('This test dataset tests basic functionality')

        # Predicting example: no target added

        self.pred = classifyintents.survey()
        self.pred.load('test_data/raw_test_data.csv')
        self.pred.clean_raw()
        self.pred.columns = self.pred.data.columns.tolist().sort()

        # Training example: target already added

        self.train = classifyintents.survey()
        self.train.load('test_data/raw_test_data_classified.csv')
        self.train.clean_raw()
        self.train.columns = self.train.data.columns.tolist().sort()

        # Load a pre-defined list of expected features
        # Note fillna otherwise this test will fail!

        self.clean_raw_expected_columns = pd.read_csv(
            'test_data/data_clean_raw_expected_columns.csv',
            skip_blank_lines=False
            ).fillna('')

        # Convert to list and sort

        self.clean_raw_expected_columns = self.clean_raw_expected_columns['columns'].tolist().sort()
   
    # No teardown required
    #@classmethod
    #def teardown_class(cls):
    #    print ("teardown_class() after any methods in this class")

    # Test whether it works on a single instance

    def test_columns_are_all_present_after_clean_raw(self):

        nt.assert_equal(
                self.train.columns, 
                self.clean_raw_expected_columns
                )

        nt.assert_equal(
                self.pred.columns, 
                self.clean_raw_expected_columns
                )

    # Now check the content of the target column
    # For training it should countain a mix of ok and finding-general
    # For predicting it should be empty

    def test_target_is_not_null_after_clean_raw(self):
        
        nt.assert_true(
                sum(self.train.data.target.isnull()) != len(self.train.data.target)
                )
    
        nt.assert_true(
                sum(self.pred.data.target.isnull()) != len(self.pred.data.target)
                )

    # For training value counts should be...

    def test_target_value_counts_after_clean_raw_for_predicting(self):

        nt.assert_true(
                sum(self.train.data.target == 'ok') == 39
                )

        nt.assert_true(
                sum(self.train.data.target == 'finding-general') == 29
                )
