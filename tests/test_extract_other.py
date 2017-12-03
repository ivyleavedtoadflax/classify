
import nose.tools as nt
import pandas as pd
from classifyintents import *
import numpy as np

class TestSmartSurveyFunctions:
    
    # Would be run before each class method

    #def setup(self):

    # To run after each class method

    #def teardown(self):
    
    # To run when the class is instantiated.

    @classmethod
    def setup_class(self):
                
        print('Running TestSmartSurveyFunctions class')
        
        self.original = pd.Series(['Yes', 'No', 'Not sure / Not yet', 'This is an other'])
        self.extracted = pd.Series(['none', 'none', 'none', 'This is an other'])
        self.rewritten = pd.Series(['Yes', 'No', 'Not sure / Not yet', 'other'])

    def test_extract_other_works_as_expected(self):

        extract_other_a = classifyintents.extract_other(self.original)

        nt.assert_true(
                self.extracted.equals(extract_other_a)
                )        

    def test_rewrite_other_works_as_expected(self):

        rewrite_other_a  = classifyintents.rewrite_other(self.original)       
        
        nt.assert_true(
                self.rewritten.equals(rewrite_other_a)
                )
