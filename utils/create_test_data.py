
"""
Command line tool to obfuscate survey files from smart survey
"""

# coding: utf-8
import pandas as pd
import random, loremipsum, sys
from os.path import splitext, basename

# Handle commandline arguments:

input_file = sys.argv[1]

if len(sys.argv) == 3:
    classified = sys.argv[2]
else:
    classified = 0


def main():

    a = pd.read_csv(input_file)
    nrow = len(a)

    # Remove IP addresses and check that they are gone. Exit if they are not!

    a['IP Address'] = ''

    if not a['IP Address'].equals(pd.Series(['' for x in range(nrow)])):
        sys.exit()

    # Create random UserIDs

    a.UserID = pd.Series(random.sample(range(10000000,11000000),nrow))
    a.UserNo = pd.Series(random.sample(range(1,1000),nrow))
    
    # Randomise the url

    a['Page Path'] = a['Page Path'][random.sample(range(nrow),nrow)]

    # Obfuscate the comments

    comment_cols = range(11,18)

    def rewrite_comments(x):

        if x != '-':
            x = loremipsum.get_sentence()
        return(x)

    def rc_iterate(x):
        
        y = [rewrite_comments(i) for i in x]
        return(y)

    # Now iterate through the cols and rewrite comments

    for i in comment_cols:

        a.ix[:,i] = rc_iterate(a.ix[:,i])
    
    # Add a dummy variable for compatibility with previous smartsurvey version

    a['dummy'] = ''
    
    # If classified argument passed, then add code1 column

    if classified:
        
        codes = ['govuk-specific','contact-government','complaint-government',
                 'service-problem','address-problem','report-issue','finding-general',
                 'compliment','internal','change-details','check-status','ok',
                 'not-ok','none','finding-problem','things-problem']

        a['code1'] = [random.choice(codes) for i in range(nrow)]
    
    # Save out to a file

    output_file = splitext(basename(input_file))[0] + '_clean.csv'

    a.to_csv(output_file, index=False)

if __name__ == '__main__':
        main()
