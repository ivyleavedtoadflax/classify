
import nose.tools as nt
import os, sys

# There is probably a cleaner way of doing this

sys.path.insert(0, os.path.abspath('../..'))
import pandas as pd
from classify import *
import numpy as np

class TestCleanUrls:
    
    # Would be run before each class method

    #def setup(self):

    # To run after each class method

    #def teardown(self):
    
    # To run when the class is instantiated.

    @classmethod
    def setup_class(self):
                
        print('Loading test_data/raw_test_data.csv into survey class')
        print('This test dataset tests basic functionality')
        
        self.a = classify.survey()
        self.a.load('test_data/raw_test_data.csv')
        self.a.clean_raw()
        self.a.clean_urls()

        print('Loading test_data/clean_url_expected_toy.csv')
        print('This test dataset provides the expected values returned from raw_test_data.csv')

        self.clean_url_expected_toy = pd.read_csv('test_data/clean_url_expected_toy.csv')
        self.clean_url_expected_toy = self.clean_url_expected_toy['page'].tolist() 
        
        print('Load test_data/raw_test_data_2.csv into survey class')
        print('This test dataset tests basic functionality with real cases')

        self.b = classify.survey()
        self.b.load('test_data/raw_test_data_2.csv')
        self.b.clean_raw()
        self.b.clean_urls()

        print('Load test_data/raw_test_data_3.csv into survey class')
        print('This test dataset tests the clean_url logic')
        
        self.c = classify.survey()
        self.c.load('test_data/raw_test_data_3.csv')
        self.c.clean_raw()
        self.c.clean_urls()

        print('Loading test_data/clean_url_expected_toy.csv')
        print('This test dataset provides the expected values returned from raw_test_data.csv')

        # Note fillna otehrwise this test will fail!

        self.clean_url_expected_pages = pd.read_csv('test_data/data_page_expected.csv',skip_blank_lines=False).fillna('')
        self.clean_url_expected_pages = self.clean_url_expected_pages['page'].tolist()

        self.data_expected = pd.read_csv('test_data/clean_url_expected_section_org.csv',skip_blank_lines=False).fillna('nan')
        self.data_expected_section = self.data_expected['section'].tolist()
        self.data_expected_org = self.data_expected['org'].tolist()

    #@classmethod
    #def teardown_class(cls):
    #    print ("teardown_class() after any methods in this class")

    # Test whether it works on a single instance

    def test_unique_pages_df(self):

        nt.assert_true(
                isinstance(self.a.unique_pages, pd.DataFrame)
                )

    def test_a_data_len(self):

        nt.assert_equal(
                len(self.a.data['full_url']),
                14
                )

    def test_unique_pages_len(self):

        nt.assert_equal(
                len(self.a.unique_pages),
                6
                )

    # Test on toy example

    def test_unique_pages_outcome(self):
    
        nt.assert_equal(
                self.a.unique_pages['page'].tolist(),
                self.clean_url_expected_toy 
                )

    # Test on n=198 example

    def test_unique_pages_df_2(self):
    
        nt.assert_true(
                isinstance(self.b.unique_pages, pd.DataFrame)
                )

    def test_b_data_len_2(self):
    
        nt.assert_equal(
                len(self.b.data['full_url']),
                198
                )

    def test_unique_pages_len_2(self):

        nt.assert_equal(
                len(self.b.unique_pages['page']),
                124
                )

    # Test the outcomes based on Sean's Rules

    def test_clean_url_rules_pages(self):

        print(self.c.unique_pages['page'].tolist())
        print(self.clean_url_expected_pages)

        nt.assert_equal(
                self.c.unique_pages['page'].tolist(),
                self.clean_url_expected_pages 
                )

    def test_clean_url_rules_section(self):
     
        print(self.c.data['section'].tolist())
        print(self.data_expected_section)

        nt.assert_equal(
                self.c.data['section'].tolist(),
                self.data_expected_section 
                )
    
    def test_clean_url_rules_org(self):
     
        print(self.c.data['org'].tolist())
        print(self.data_expected_org)
        
        nt.assert_equal(
                self.c.data['org'].tolist(),
                self.data_expected_org 
                )
