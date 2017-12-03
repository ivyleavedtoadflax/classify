
import nose.tools as nt
import os, sys
import pandas as pd
from classifyintents import *
import numpy as np

class TestTrainerMethod:
    
    # Would be run before each class method

    #def setup(self):

    # To run after each class method

    #def teardown(self):
    
    # To run when the class is instantiated.

    @classmethod
    def setup_class(self):
                
        print('Running TestTrainer class')
        
        self.a = classifyintents.survey()
        self.a.load('test_data/raw_test_data_classified.csv')
        self.a.clean_raw()
        self.a.clean_urls()
        print(self.a.data.code1.value_counts())

    #@classmethod
    #def teardown_class(cls):
    #    print ("teardown_class() after any methods in this class")

    # Test whether it works on a single instance
    def test_trainer_method_preserves_code1_variable(self):

        nt.assert_true(
                'code1' in self.a.cleaned.columns.tolist()
                )        

    def test_trainer_method_converts_one_code_correctly(self):

        self.a.trainer(['ok'])
        
        nt.assert_equal(
                set(self.a.cleaned.code1.tolist()), 
                set([1,0])
                )

    def test_trainer_method_converts_two_codes_correctly(self):
        
        self.a.trainer(['ok','finding-general'])

        nt.assert_equal(
                set(self.a.cleaned.code1.tolist()), 
                set([1,0])
                )
