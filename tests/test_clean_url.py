import nose.tools as nt
import os, sys
import pandas as pd
from classifyintents import *
import numpy as np

class TestCleanUrls:
    
    # Would be run before each class method

    #def setup(self):

    # To run after each class method

    #def teardown(self):
    
    # To run when the class is instantiated.

    @classmethod
    def setup_class(self):
                
        print('Load test_data/raw_test_data_2.csv into survey class')
        print('This test dataset tests basic functionality with real cases')

        self.b = classifyintents.survey()
        self.b.load('test_data/raw_test_data_2.csv')
        self.b.clean_raw()
        self.b.clean_urls()

        print('Load test_data/raw_test_data_3.csv into survey class')
        print('This test dataset tests the clean_url logic')
        
        self.c = classifyintents.survey()
        self.c.load('test_data/raw_test_data_3.csv')
        self.c.clean_raw()
        self.c.clean_urls()

        # Note fillna otherwise this test will fail!

        self.clean_url_expected_pages = pd.read_csv('test_data/data_page_expected.csv',skip_blank_lines=False).fillna('')
        self.clean_url_expected_pages = self.clean_url_expected_pages['page'].tolist()

        self.data_expected = pd.read_csv('test_data/clean_url_expected_section_org.csv',skip_blank_lines=False).fillna('nan')
        self.data_expected_section = self.data_expected['section'].tolist()
        self.data_expected_org = self.data_expected['org'].tolist()

    #@classmethod
    #def teardown_class(cls):
    #    print ("teardown_class() after any methods in this class")

    # Test on n=196 example

    def test_unique_pages_is_a_dataframe(self):
    
        nt.assert_true(
                isinstance(self.b.unique_pages, pd.DataFrame)
                )

    def test_full_url_length_is_correct(self):
    
        nt.assert_equal(
                len(self.b.data['full_url']),
                196
                )

    def test_unique_pages_length_is_correct(self):

        nt.assert_equal(
                len(self.b.unique_pages['page']),
                124
                )

    # Test the outcomes based on Sean's Rules

    def test_clean_url_returns_expected_page(self):

        nt.assert_equal(
                self.c.unique_pages['page'].tolist(),
                self.clean_url_expected_pages 
                )

    def test_clean_url_returns_expected_section(self):
     
        nt.assert_equal(
                self.c.data['section'].tolist(),
                self.data_expected_section 
                )
    
    def test_clean_url_returns_expected_org(self):
     
        nt.assert_equal(
                self.c.data['org'].tolist(),
                self.data_expected_org 
                )
