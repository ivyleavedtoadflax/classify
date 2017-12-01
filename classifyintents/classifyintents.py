# coding: utf-8

import numpy as np
import pandas as pd
import re, requests
from sklearn.preprocessing import LabelEncoder
import os.path
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk import ngrams

class survey:
    """Class for handling intents surveys from google sheets"""

    def __init__(self):
        print('***** Instantiated survey class *****')
        pass
  
    def load(self, x):

        print('***** Running load method *****')
        print('*** Loading data ', x)

        try:
            
            self.raw = pd.read_csv(x)

            # Strip whitespace from columns to save problems later!
            # Remove no break whitespace first.
            
            self.raw.columns = [i.replace("\xa0", " ") for i in self.raw.columns]
            self.raw.columns = self.raw.columns.str.strip()
            
            # If raw csv from survey monkey: strip the second row of headers:
            
            self.raw = drop_sub(self.raw)

            # Strange behaviour leading to top 10 rows being filled with NaN.
            # Drop these by dropping rows with now RespondentID

            self.raw.dropna(subset=['RespondentID'],inplace=True)

            self.raw['RespondentID'] = self.raw['RespondentID'].astype('int')
            
        except FileNotFoundError:
            print('*** Target file ', x,' does not exist')
            raise
            
        except: 
            print('Unexpected error loading raw data from file ' + x)
            print('Original error message:', sys.exc_info()[0])
            raise

        # Define mappings and columns used in iterators

        # Clean date columns
        
    def clean_raw(self, date_format=None):

        print('***** Running clean_raw method *****')
        print('*** The cleaned data are stored in survey.data')

        self.data = self.raw.copy()

        # Use mapping to rename and subset columns

        self.data.rename(columns = self.raw_mapping, inplace=True)

        # Subset columns mentioned in mapping dict

        cols = list(self.raw_mapping.values())
        
        # Strip down only to the columns listed in raw.mapping - append code1 here
        # as it should always now be present in the data.    
        
        cols.extend(['code1'])

        # Check here: if code1 is not in the raw data, i.e. we are predicting, not
        # training, then add the column to the dataframe.

        if 'code1' not in self.data.columns.tolist():
            self.data['code1'] = str()            
        
        self.data = self.data[cols]
        
# Arrange date features

        self.data['start_date'] = clean_date(self.data['start_date'], date_format)
        self.data['end_date'] = clean_date(self.data['end_date'], date_format)
        
        # Create time delta and normalise

        self.data['time_delta'] = time_delta(self.data['end_date'], self.data['start_date'])
         
        self.data['time_delta'] = normalise(self.data['time_delta'])

        self.data = pd.concat([
             pd.DataFrame(columns=['org','section']),date_features(self.data['start_date']), self.data],
            axis = 1
        )
        
        # Classify all empty relevant comments as 'none'. This has been moved out of the class!
        # Need to have a think about whether this should be in the class or not!

        #no_comments = (self.data['comment_further_comments'] == 'none') & (self.data['comment_where_for_help'] == 'none') & (self.data['comment_other_where_for_help'] == 'none') & (self.data['comment_why_you_came'] == 'none')

        #self.data.loc[no_comments,'code1'] = 'none'

        # Features on column names

        try:
            for col in self.data:
                
                # Is the column entirely NaN?
                # Currently this is only implemented for comment columns
                # May make sense to do this for all column types...
                
                all_null = (self.data[col].isnull().sum() == len(self.data[col]))
                
                # Start by cleaning the categorical variables

                if col in self.categories:
                    self.data[col] = clean_category(self.data[col])

                # Now clean the comment variables

                elif 'comment' in col and not all_null:
                    self.data[col + '_capsratio'] = [string_capsratio(x) for x in self.data[col]]
                    self.data[col + '_nexcl'] = [string_nexcl(x) for x in self.data[col]]
                    self.data[col + '_len'] = string_len(self.data[col])
                    self.data[col] = clean_comment(self.data[col])

                elif 'comment' in col and all_null:
                    self.data[col + '_capsratio'] = 0
                    self.data[col + '_nexcl'] = 0
                    self.data[col + '_len'] = 0
                    self.data[col] = 'none'

                # Finally clean the outcome codes        

                #elif col in self.codes:
                #    self.data[col] = clean_code(self.data[col], self.code_levels)
        except: 
            print('Error cleaning ' + col + ' column')
            print('Original error message:', sys.exc_info()[0])
            raise

    def clean_urls(self):

        print('***** Running clean_urls() method *****')

        # First apply URL filtering rules, and output these cleaned URLs to 
        # a DataFrame called unique_pages.

        # Quick fix here - convert the org and section columns back to strings, they previously
        # were converted to categorical. Need to fix this higher upstream.

        self.data.org = self.data.org.astype('str')
        self.data.section = self.data.section.astype('str')

        query = '\/?browse'

        # Add a blank page column

        self.data['page'] = str()

        if 'full_url' in list(self.data.columns):

            for index, row in self.data.iterrows():
    
                # Deal with cases of no address
                
                if ((row['full_url'] == '/') | (row['full_url'] == np.nan) | (str(row['full_url']) == 'nan')):
        
                    continue
    
                # If FCO government/world/country page:
                # Strip back to /government/world and
                # set org to FCO
    
                elif re.search('/government/world', str(row['full_url'])):

                    self.data.loc[index,['org','page']] = ['Foreign & Commonwealth Office','/government/world']
        
                # If full_url starts with /guidance or /government:
                # and there is no org (i.e. not the above)
                # Set page to equal full_url                

                elif re.search('\/guidance|\/government', str(row['full_url'])):
                    if row['org'] == 'nan':
                        self.data.loc[index,'page'] = row['full_url']  
    
                # If page starts with browse:
                # set page to equal /browse/xxx/ 
    
                elif re.search('\/browse', str(row['full_url'])):
                    self.data.loc[index, 'page'] = reg_match(query, row['full_url'], 1)
              
                # If the section is also empty:
                # Set section to be /browse/--this-bit--/

                    if row['section'] == 'nan':
                        self.data.loc[index, 'section'] = reg_match(query, row['full_url'], 2)
            
                # Otherwise:
                # Strip back to the top level

                else:
                    self.data.loc[index, 'page'] = '/' + reg_match('.*', row['full_url'], 0)

        else:
            print('Full_url column not contained in survey.data object.')
            print('Are you working on a raw data frame? You should be!')
            
        
        # Take only urls where there is no org or section.
        
        self.unique_pages = self.data.loc[(self.data['org'] == 'nan') & (self.data['section'] == 'nan'),'page']
        
        # Convert to a DataFrame to make easier to handle

        self.unique_pages = pd.DataFrame(self.unique_pages, columns = ['page'])
        
        # Drop duplicate pages!

        self.unique_pages = self.unique_pages.drop_duplicates()

        print('*** There are ' + str(len(self.unique_pages['page'])) + ' unique URLs to query. These are stored in survey.unique_pages.')


    def api_lookup(self):

        # Run the api lookup, then subset the return (we're not really interested in most of what we get back)
        # then merge this back into self.data, using 'page' as the merge key.

        print('***** Running api_lookup() method *****')
        print('*** This may take some time depending on the number of URLS to look up')

        # This is all a bit messy from the origin function.
        # Would be good to clean this up at some point.
        
        column_names = ['organisation0',
                        'organisation1',
                        'organisation2',
                        'organisation3',
                        'organisation4',
                        'section0',
                        'section1',
                        'section2',
                        'section3']
        
        # Only run the lookup on cases where we have not already set an org and section
                   
       # self.org_sect = [get_org(i) for i in self.data.loc[((self.data.section == 'nan') &(self.data.org == 'nan')),['page']]]
        self.org_sect = [get_org(i) for i in self.unique_pages['page']]
        self.org_sect = pd.DataFrame(self.org_sect, columns=column_names)
        self.org_sect = self.org_sect.set_index(self.unique_pages.index)
 
        # Convert any NaNs to none, so they are not dropped when self.trainer/predictor is run
        
        self.org_sect['organisation0'].replace(np.nan, 'none', regex=True, inplace=True)
        self.org_sect['section0'].replace(np.nan, 'none', regex=True, inplace=True)   

        # Retain the full lookup, but only add a subset of it to the clean dataframe

        org_sect = self.org_sect[['organisation0','section0']]
        org_sect.columns = ['org','section']
 
        self.unique_pages = pd.concat([self.unique_pages, org_sect], axis = 1)
        
        print('*** Lookup complete, merging results back into survey.data')

        self.data = pd.merge(right = self.data.drop(['org','section'], axis=1), left = self.unique_pages, on='page', how='outer')

        self.data.drop_duplicates(subset=['respondent_ID'],inplace=True)
     
    # Define code to encode to true (defualt to ok)

    def trainer(self, classes = None):

        print('***** Running trainer method *****')

        if classes == None:
            classes = ['ok']
            
        try:
            self.cleaned = self.data.copy()
            self.cleaned = self.data[self.selection + self.codes]
            self.cleaned = self.cleaned.dropna(how = 'any')
            
            # There is an argument for doing this in the .clean() method.
            # It might useful to be able to call the data before this is
            # applied however. Note that after running load(), clean(),
            # trainer() there are now three similar copies of the data being
            # stored within the class object. At the present small scale this
            # is not a problem, but in time it may be necessary to readress 
            # this.
            
            # LabelEncoder converts labels into numeric codes for all of the factors.

            le = LabelEncoder()

            for col in self.categories:
                self.cleaned.loc[:,col] = le.fit_transform(self.cleaned.loc[:,col])
            
            le.fit(self.cleaned['code1'])
            self.cleaned['code1'] = le.transform(self.cleaned['code1'])

            # At present this deals only with the binary case. Would be
            # good to generalise this in future to allow it to be customised.
            # This codes the outcomes as 0 or 1, but ideall would do 0, 1, 2, etc.

            self.bin_true = le.transform(classes)

            self.cleaned['code1'] = [1 if x in self.bin_true else 0 for x in self.cleaned['code1']] 
            #self.cleaned.loc[self.cleaned['code1'] not in self.bin_true,'code1'] = 0
            #self.cleaned.loc[self.cleaned['code1'] in self.bin_true,'code1'] = 1

            self.cleaned.drop('respondent_ID', axis=1, inplace=True)            

        except Exception as e:
            print('There was an error while running trainer method')
            print('Original error message:')
            print(repr(e))

    def predictor(self):

        print('***** Running predictor method *****')

        try:

            self.cleaned = self.data.copy()
            self.cleaned = self.data[self.selection]
            self.cleaned = self.cleaned.dropna(how = 'any')

# Debug            print(self.cleaned.isnull().sum())

            le = LabelEncoder()

            for col in self.categories:
                self.cleaned.loc[:,col] = le.fit_transform(self.cleaned.loc[:,col])

            self.cleaned.drop('respondent_ID', axis=1, inplace=True)            

        except Exception as e:
            print('There was an error while subsetting survey data')
            print('Original error message:')
            print(repr(e))

    raw_mapping = {
            "clientID" : "client_id",
            "UserID" : "respondent_ID",
            "UserNo" : "user_no_drop",
            "Tracking.Link" : "collector_id",
            "Started" : "start_date",
            "Ended" : "end_date",
            "IP.Address" : "ip_address",
            "Email" : "email_address",
            "Name" : "first_name",
            "Unique.ID" : "unique_id_drop",
            "Page.Path" : "full_url",
            "Q1..Are.you.using.GOV.UK.for.professional.or.personal.reasons.":"cat_work_or_personal",
            "Q2..What.kind.of.work.do.you.do.":"comment_what_work",
            "Q3..Describe.why.you.came.to.GOV.UK.todayPlease.do.not.include.personal.or.financial.information..eg.your.National.Insurance.number.or.credit.card.details.":"comment_why_you_came",
            "Q4..Have.you.found.what.you.were.looking.for." :"cat_found_looking_for",
            "Q5..Overall..how.did.you.feel.about.your.visit.to.GOV.UK.today." : "cat_satisfaction",
            "Q5.1..." : "cat_satisfaction",
            "Q6..Have.you.been.anywhere.else.for.help.with.this.already.":"cat_anywhere_else_help",
            "Q7..Where.did.you.go.for.help.":"comment_where_for_help",
            "Q8..If.you.wish.to.comment.further..please.do.so.here.Please.do.not.include.personal.or.financial.information..eg.your.National.Insurance.number.or.credit.card.details.":"comment_further_comments"
    }

    categories = [
        # May be necessary to include date columns at some juncture  
        #'weekday', 'day', 'week', 'month', 'year', 
        'org', 'section', 'cat_work_or_personal', 
        'cat_satisfaction', 'cat_found_looking_for', 
        'cat_anywhere_else_help'
    ]
  
    comments = [
        'comment_what_work', 'comment_why_you_came', 'comment_other_found_what',
        'comment_other_else_help', 'comment_other_where_for_help', 'comment_where_for_help', 'comment_further_comments'
    ]
    
    codes = [
        'code1'
    ]
    
    # Could do some fuzzy matching here to improve matching to category names
    # Some training examples are likely to be lost in clean_codes due to
    # inconsistent naming of classes by volunteers.
    
    code_levels = [
    'ok', 'finding-general', 'service-problem', 'contact-government', 
    'check-status', 'change-details', 'govuk-specific', 'compliment',
    'complaint-government','notify', 'internal', 'pay', 'report-issue',
    'address-problem', 'verify'
    ]

    selection = ['respondent_ID', 'weekday', 'day', 'week', 'month', 'year', 'time_delta'] + categories + [(x + '_len') for x in comments] + [(x + '_nexcl') for x in comments] + [(x + '_capsratio') for x in comments]

def drop_sub(x):
    if x.iloc[0,].str.match('Open-Ended Response').sum():
        x.drop(0, inplace=True)
    return(x)

def string_len(x):
    try:

        x = x.str.strip()
        x = x.str.lower()

        x = x.replace(r'\,\s?\,?$|none\,', 'none', regex=True)
        
        # Convert NaN to 'a'. Then when counted this will
        # be a 1. Whilst not 0, any entry with 1 is virtually
        # meaningless, so 1 is a proxy for 0.
        
        x = pd.Series([len(y) for y in x.fillna('a')])
        # Now normalise the scores
        
        x = (x - x.mean()) / (x.max() - x.min())
               
    except Exception as e:
        print('There was an error converting strings to string length column!')
        print('Original error message:')
        print(repr(e))
    return(x)

def string_capsratio(x):
    try:
        if not pd.isnull(x):
            x = sum([i.isupper() for i in x])/len(x)
        else:
            x = 0

    except Exception as e:
        print('There was an error creating capitals ratio on column: ' + x)
        print('Original error message:')
        print(repr(e))
    return(x)

def string_nexcl(x):
    try:
        if not pd.isnull(x):
            x = sum([i == '!' for i in x]) / len(x)
        else:
            x = 0

    except Exception as e:
        print('There was an error creating n of exclamations on column: ' + x)
        print('Original error message:')
        print(repr(e))
    return(x)
    
def clean_date(x, format=None):
    try:
        x = pd.to_datetime(x, format=format)
               
    except Exception as e:
        print('There was an error cleaning the StartDate column!')
        print('Original error message:')
        print(repr(e))
    return(x)

def date_features(x):
    try:
        x = pd.to_datetime(x)
        
        X = pd.DataFrame({
                'weekday' : x.dt.weekday,
                'day' : x.dt.day,
                'week' : x.dt.week,
                'month' : x.dt.month,
                'year' : x.dt.year,
             })
        
    except Exception as e:
        print('There was an error creating date feature: ' + x)
        print('Original error message:')
        print(repr(e))
    return(X)

def clean_category(x):
    try:
        
        # May be needed if columns are integer
        x = x.apply(str)
        x = x.str.lower()
        x = x.replace(r'null|\#Value\!', 'none', regex=True)
        x = x.fillna('none')
        x = pd.Series(x)
        x = x.astype('category')
        
    except Exception as e:
        print('There was an error cleaning the', x ,'column.')
        print('Original error message:')
        print(repr(e))
    return(x)

def clean_comment(x):
    try:
        
        x = x.str.strip()
        x = x.str.lower()
        
        # Weirdness with some columns being filled with just a comma.
        # Is this due to improper handling of the csv file somewhere?        
        
        x = x.replace(r'\,\s?\,?$|none\,', 'none', regex=True)
        x = x.fillna('none')
        
    except Exception as e:
        print('There was an error cleaning the', x ,'column.')
        print('Original error message:')
        print(repr(e))
    return(x)
      
def clean_code(x, levels):
    try:

       # If the whole column is not null
       # i.e. we want to train rather than just predict

        if not pd.isnull(x).sum() == len(x):        
            x = x.str.strip()
            x = x.str.lower()
            x = x.replace('\_', '\-', regex=True)
        
            # Rules for fixing random errors.
            # Commented out for now 

            #x = x.replace(r'^k$', 'ok', regex=True)
            #x = x.replace(r'^finding_info$', 'finding_general', regex=True)
            #x = x.replace(r'^none$', np.nan, regex=True)
        
            x[~x.isin(levels)] = np.nan
            x = pd.Series(x)
            x = x.astype('category')
        
    except Exception as e:
        print('There was an error cleaning the', x ,'column.')
        print('Original error message:')
        print(repr(e))
    return(x)

#stops = set(stopwords.words("english"))     # Creating a set of Stopwords
#p_stemmer = PorterStemmer() 
#
#def concat_ngrams(x):
#    #if len(x) > 1 & isinstance(x, list):
#    if isinstance(x, tuple):
#        x = '_'.join(x)
#    return(x)
#
#def cleaner(row):
#    
#    # Function to clean the text data and prep for further analysis
#    text = row.lower()                      # Converts to lower case
#    text = re.sub("[^a-zA-Z]"," ",text)          # Removes punctuation
#    text = text.split()                          # Splits the data into individual words 
#    text = [w for w in text if not w in stops]   # Removes stopwords
#    text = [p_stemmer.stem(i) for i in text]     # Stemming (reducing words to their root)
#    text3 = list(ngrams(text, 2))
#    text2 = list(ngrams(text, 3))
#    text = text + text2 + text3
#    text = list([concat_ngrams(i) for i in text])
#    return(text)  

## Functions dealing with the API lookup

def lookup(r,page,index):        
    try:
        if page == 'mainstream_browse_pages':
            x = r['results'][0][page][index]            
        elif page == 'organisations':
            x = r['results'][0][page][index]['title']
        else:
            print('page argument must be one of "organisations" or "mainstream_browse_pages"')
            sys.exit(1)
    except (IndexError, KeyError) as e:
        x = 'null'
    return(x)

def get_org(x):
    
    # argument x should be pd.Series of full length urls
    # Loop through each entry in the series

    url = "https://www.gov.uk/api/search.json?filter_link[]=%s&fields=organisations&fields=mainstream_browse_pages" % x
    
    #print('Looking up ' + url)
    
    try:
       
        #url = "https://www.gov.uk/api/search.json?filter_link[]=%s&fields=y" % (x, y)

        # read JSON result into r
        r = requests.get(url).json()

        # chose the fields you want to scrape. This scrapes the first 5 instances of organisation, error checking as it goes
        # this exception syntax might not work in Python 3

        organisation0 = lookup(r,'organisations', 0)
        organisation1 = lookup(r,'organisations', 1)
        organisation2 = lookup(r,'organisations', 2)
        organisation3 = lookup(r,'organisations', 3)
        organisation4 = lookup(r,'organisations', 4)
        section0 = lookup(r,'mainstream_browse_pages', 0)
        section1 = lookup(r,'mainstream_browse_pages', 1)
        section2 = lookup(r,'mainstream_browse_pages', 2)
        section3 = lookup(r,'mainstream_browse_pages', 3)

        row = [organisation0,
                organisation1,
                organisation2,
                organisation3,
                organisation4,
                section0,
                section1,
                section2,
                section3]
        
        return(row)

    except Exception as e:
        print('Error looking up ' + url)
        print('Returning "none"')
        row = ['none'] * 9
        return(row)

## Functions dealing with developing a time difference feature

def normalise(x):
    
    x = (x - np.mean(x)) / np.std(x)
    return(x)

def time_delta(x,y):
    
    # Expects datetime objects

    delta = x - y
    # Required for old versions!
    #delta = np.timedelta64(delta, 's')
    #delta = delta.astype('int')
    delta = delta.astype('timedelta64[s]')
    delta = delta.astype('int')

    # normalise statment moved to method to keep this function simple

    return(delta)

def reg_match(r, x, i):

    r = r + '/'
    
    # r = uncompiled regex query
    # x = string to search
    # i = index of captive group (0 = all)
    
    p = re.compile(r)
    s = p.search(x)
    
    if s:
        t = re.split('\/', x, maxsplit=3)
        if i == 0:
            found = t[1]
        if i == 1:
            found = '/' + t[1] + '/' + t[2]
        elif i == 2:
            found = t[2]
    else: 
        found = x
    return(found)

