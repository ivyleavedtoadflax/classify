# coding: utf-8
"""
Classifyintents: a collection of classes and functions for
wrangling data from the govuk intent survey.
"""

import re
import sys
import time
import logging
import logging.config
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class survey(object):
    """Class for handling intents surveys from google sheets"""

    def __init__(self):
        """
        Instantiate the class

        Expects a logging object to have been created in the
        script executing the class.
        """

        self.logger = logging.getLogger("classifyintents")
        self.logger.info('Instantiated survey class')

        self.raw = pd.DataFrame()
        self.data = pd.DataFrame()
        self.unique_pages = pd.DataFrame()
        self.org_sect = pd.DataFrame()
        self.cleaned = pd.DataFrame()

        # TODO: LabelEncoder() is used both to convert categorical variables to 
        # integer, and for converting the targets to integer classes. Only the
        # latter is recorded to the class, the other encoders are ephemeral. These
        # encoders need to be made available for calculating features in new data
        # in future versions.

        self.target_encoder = LabelEncoder()

    def load(self, path):
        """
        Load the data from csv file

        The data file will need to have been through some initial cleaning.
        This is currently handled by the R script reformat.R
        (https://raw.githubusercontent.com/ukgovdatascience/classifyintentspipe/1de84e053e485c53fd2d82c9512b0dc084487d83/reformat.R)

        An initial check is run to ensure that expect column names are present.

        :param path: <str> Path to the data file.
        """

        self.logger.info('Running survey.load method')
        self.logger.info('Loading data %s', path)

        try:

            self.raw = pd.read_csv(path)

        except FileNotFoundError:
            self.logger.exception('Input file %s does not exist', path)
            raise

        except:
            self.logger.error('Unexpected error loading raw data from file %s', path)
            raise

        # Run check on the incoming columns against expected
        try:

            assert set(self.raw.columns) == set(self.raw_mapping.values())

        except AssertionError:
            self.logger.error('Incorrect columns in %s:', path)
            self.logger.error('Expected:\n%s\nReceived:\n%s',
                              set(self.raw.columns), set(self.raw_mapping.values()))
            raise

        self.logger.info('Shape of %s: %s', path, self.raw.shape)
        self.raw.dropna(subset=['respondent_id'], inplace=True)

        self.logger.info('Shape of %s after dropping missing respondent_ids %s',
                         path, self.raw.shape)

        # Convert respondent_id to int.

        self.raw['respondent_id'] = self.raw['respondent_id'].astype('int')
        self.logger.debug('%s', self.raw.dtypes)

        # Define mappings and columns used in iterators

        # Clean date columns

    def clean_raw(self, date_format=None):

        self.logger.info('Running clean_raw method')
        self.logger.info('The cleaned data are stored in survey.data')

        self.data = self.raw.copy()

        # NOTE: in 0.6.1> it is assumed that this process is being done
        # prior to the data being loaded by the survey class. It would
        # be good to do the process here, but for now, assume that it is
        # being done prior to loading.

        # Use mapping to rename and subset columns

        #self.data.rename(columns = self.raw_mapping, inplace=True)

        # NOTE: in 0.6.1> In the assertion in the .load method we assume that the right
        # columns are passed, so no columns need to be dropped here.

        # Subset columns mentioned in mapping dict

        cols = list(self.raw_mapping.values())

        # Strip down only to the columns listed in raw.mapping - append target here
        # as it should always now be present in the data.

        cols.extend(['target'])

        # NOTE: in 0.6.1> this functionality is not used, because in the .load method,
        # the incoming columns are checked against raw_columns, which does not contain
        # target.

        # Check here: if target is not in the raw data, i.e. we are predicting, not
        # training, then add the column to the dataframe.

        if 'target' not in self.data.columns.tolist():
            self.data['target'] = str()

        # NOTE: This step of dropping additional columns, is not required here.

        self.data = self.data[cols]

        # Arrange date features

        self.data['start_date'] = clean_date(self.data['start_date'], date_format)
        self.data['end_date'] = clean_date(self.data['end_date'], date_format)

        self.logger.info('Added date features: start_date and end_date')
        self.logger.debug("Head of data['start_date']: %s", self.data['start_date'].head())
        self.logger.debug("Head of data['end_date']: %s", self.data['start_date'].head())

        # Create time delta and normalise

        self.data['time_delta'] = time_delta(self.data['end_date'], self.data['start_date'])
        self.data['time_delta'] = normalise(self.data['time_delta'])

        self.logger.info('Added date feature: time_delta')
        self.logger.debug("Head of data['time_delta']: %s", self.data['time_delta'].head())

        # Combine new date features with existing features.
        # Prepare org and section features for population from API lookup.

        self.data = pd.concat([
            pd.DataFrame(columns=['org', 'section']),
            date_features(self.data['start_date']),
            self.data], axis=1)

        # Create features on column names

        try:
            for col in self.data:

                # Is the column entirely NaN?
                # NOTE: Currently this is only implemented for comment columns
                # It may make sense to do this for all column types, though it is
                # less likely that these columns will be empty.

                all_null = (self.data[col].isnull().sum() == len(self.data[col]))
                if all_null:

                    self.logger.info('%s column is all empty', col)
                    self.logger.debug('head of %s column: \n%s', col, self.data[col].head())

                # Start by cleaning the categorical variables

                if col in self.categories:
                    self.data[col] = clean_category(self.data[col])

                # Now clean the comment variables

                elif 'comment' in col and not all_null:
                    self.data[col + '_capsratio'] = [string_capsratio(x) for x in self.data[col]]
                    self.data[col + '_nexcl'] = [string_nexcl(x) for x in self.data[col]]
                    self.data[col + '_len'] = string_len(self.data[col])
                    self.data[col] = clean_comment(self.data[col])

                    self.logger.info('Added string features to %s', col)
                    self.logger.debug('head of %s column: \n%s',
                                      col + '_capsratio', self.data[col + '_capsratio'].head())

                # If the column is all null, just return zeros.

                elif 'comment' in col and all_null:
                    self.data[col + '_capsratio'] = 0
                    self.data[col + '_nexcl'] = 0
                    self.data[col + '_len'] = 0
                    self.data[col] = 'none'

        except:
            self.logger.error('Error cleaning %s column', col)
            raise

    def clean_urls(self):
        """
        Extract additional features from the gov.uk content API
        """

        self.logger.info('Running clean_urls() method')

        # First apply URL filtering rules, and output these cleaned URLs to
        # a DataFrame called unique_pages.

        # NOTE: Quick fix here - convert the org and section columns back to
        # strings, they previously were converted to categorical. Need to
        # fix this higher upstream.

        self.data.org = self.data.org.astype('str')
        self.data.section = self.data.section.astype('str')

        # NOTE: The logic for this section was provided as expert knowledge
        # from a performance analyst familiar with the process. It may need
        # updating in the future as the content API develops.

        # Set regex query to be used in the reg_match function later.

        query = r'\/?browse'

        # Add a blank page column

        self.data['page'] = str()

        try:

            if 'full_url' in list(self.data.columns):

                for index, row in self.data.iterrows():

                    # Deal with cases of no address

                    if ((row['full_url'] == '/') | (row['full_url'] == np.nan)
                            | (str(row['full_url']) == 'nan')):

                        continue

                    # If FCO government/world/country page:
                    # Strip back to /government/world and
                    # set org to FCO

                    elif re.search('/government/world', str(row['full_url'])):

                        self.data.loc[index, ['org', 'page']] = ['Foreign & Commonwealth Office',
                                                                 '/government/world']

                    # If full_url starts with /guidance or /government:
                    # and there is no org (i.e. not the above)
                    # Set page to equal full_url

                    elif re.search(r'\/guidance|\/government', str(row['full_url'])):
                        if row['org'] == 'nan':
                            self.data.loc[index, 'page'] = row['full_url']

                    # If page starts with browse:
                    # set page to equal /browse/xxx/

                    elif re.search(r'\/browse', str(row['full_url'])):
                        self.data.loc[index, 'page'] = reg_match(query, row['full_url'], 1)

                    # If the section is also empty:
                    # Set section to be /browse/--this-bit--/

                        if row['section'] == 'nan':
                            self.data.loc[index, 'section'] = reg_match(query, row['full_url'], 2)

                    # Otherwise:
                    # Strip back to the top level

                    else:
                        self.data.loc[index, 'page'] = '/' + reg_match('.*', row['full_url'], 0)

        except KeyError:
            self.logger.error("'full_url' column not contained in survey.data object. "
                              "Ensure you are working with the .data DataFrame.")
            raise

        # Take only urls where there is no org or section.

        self.unique_pages = self.data.loc[(self.data['org'] == 'nan') &
                                          (self.data['section'] == 'nan'), 'page']

        # Convert to a DataFrame to make easier to handle

        self.unique_pages = pd.DataFrame(self.unique_pages, columns=['page'])

        # Drop duplicate pages!

        self.unique_pages = self.unique_pages.drop_duplicates()

        self.logger.info('There are %s unique URLs to query. '
                         'These are stored in survey.unique_pages.',
                         str(len(self.unique_pages['page'])))


    def api_lookup(self, wait=0.1):
        """
        Perform a lookup using the GOV.UK content API
        """
        # NOTE: Future versions could use github.com/ukgovdatascience/govukurllookup

        # Run the api lookup, then subset the return (we're not really interested
        # in most of what we get back) then merge this back into self.data, using
        # 'page' as the merge key.

        self.logger.info('Running api_lookup() method')
        self.logger.info('Looking up %s urls', self.unique_pages.shape[0])

        # Only run the lookup on cases where we have not already set an org and section

        org_sect = []
        for i, page in enumerate(self.unique_pages['page']):

            total = self.unique_pages.shape[0]

            if i % 50 == 0:

                self.logger.info('Looking up page %s/%s', i, total)

            response = get_org(page)
            org_sect.append(response)

            # NOTE: as of 2017-12-02 I have experienced difficulties running the API lookup
            # from a laptop not connected to the VPN. Introduce a pause here to reduce the
            # rate of requests being sent to the API. There is no logic to the size of this
            # sleep.

            time.sleep(wait)

        self.logger.debug('First five entries of org_sect list:\n%s', org_sect[0:5])

        # This is all a bit messy from the origin function.
        # Would be good to clean this up at some point.

        column_names = ['organisation0', 'organisation1', 'organisation2',
                        'organisation3', 'organisation4', 'section0', 'section1',
                        'section2', 'section3']

        self.org_sect = pd.DataFrame(org_sect, columns=column_names)
        self.org_sect = self.org_sect.set_index(self.unique_pages.index)

        self.logger.info('Finished API lookup')
        self.logger.info('org_sect shape: %s: self.org_sect.shape')

        # Convert any NaNs to none, so they are not dropped when
        # self.trainer/predictor is run

        self.org_sect['organisation0'].replace(np.nan, 'none', regex=True, inplace=True)
        self.org_sect['section0'].replace(np.nan, 'none', regex=True, inplace=True)

        # Retain the full lookup, but only add a subset of it to the clean dataframe

        org_sect = self.org_sect[['organisation0', 'section0']]
        org_sect.columns = ['org', 'section']

        # Merge the unique_pages dataframe with the org_sect lookup dataframe

        self.unique_pages = pd.concat([self.unique_pages, org_sect], axis=1)

        self.logger.info('Lookup complete, merging results back into survey.data')
        self.logger.debug('unique_pages.head:\n%s', self.unique_pages.head())

        self.data = pd.merge(right=self.data.drop(['org', 'section'], axis=1),
                             left=self.unique_pages, on='page', how='outer',
                             indicator=True)

        self.logger.info('Merged data shape is:\n%s', self.data.shape)
        self.logger.debug('Merged data head is:\n%s', self.data.head())
        self.logger.debug('Merged data columns is:\n%s', self.data.columns)
        self.logger.debug('data merge success:\n%s', self.data['_merge'].value_counts())
        self.logger.debug('Top five right_only:\n%s',
                          self.data[self.data['_merge'] == 'right_only'][0:5])

        self.logger.info('Shape before dropping duplicates:\n%s', self.data.shape)
        self.data.drop_duplicates(subset=['respondent_id'], inplace=True)
        self.logger.info('Shape after dropping duplicates:\n%s', self.data.shape)

    # Define target to encode to true (defualt to ok)

    def trainer(self, classes=None):
        """
        Prepare the data for training
        """

        self.logger.info('Running trainer method')

        if classes is None:
            classes = ['ok']

        try:
            self.cleaned = self.data.copy()
            self.cleaned = self.data[self.selection + self.codes]
            self.cleaned = self.cleaned.dropna(how='any')

            # TODO: There is an argument for doing this in the .clean() method.
            # It might useful to be able to call the data before this is
            # applied however. Note that after running load(), clean(),
            # trainer() there are now three similar copies of the data being
            # stored within the class object. At the present small scale this
            # is not a problem, but in time it may be necessary to readress
            # this.

            # LabelEncoder converts labels into numeric codes for all of the factors.

            for col in self.categories:

                encoder = LabelEncoder()
                encoder.fit(self.cleaned[col]) 

                self.logger.debug('%s converted to the following integers:\n%s',
                                  col, dict(zip(encoder.transform(self.cleaned[col]),
                                                self.cleaned[col])))

                self.cleaned.loc[:, col] = encoder.transform(self.cleaned.loc[:, col])

            # Convert targets to integer classes

            self.target_encoder.fit(self.cleaned['target'])
            self.cleaned['target'] = self.target_encoder.transform(self.cleaned['target'])

            # TODO: At present this deals only with the binary case. Would be
            # good to generalise this in to allow it to be customised.
            # This codes the outcomes as 0 or 1, but ideall would do 0, 1, 2, etc.

            bin_true = self.target_encoder.transform(classes)

            targets_numeric = [1 if x in bin_true else 0 for x in self.cleaned['target']]

            self.logger.debug('Targets converted to the following integers:\n%s',
                              dict(zip(targets_numeric, self.cleaned['target'])))

            self.cleaned['targets'] = targets_numeric

            self.cleaned.drop('respondent_id', axis=1, inplace=True)

        except Exception:
            self.logger.error('Error while running trainer method')
            raise

    def predictor(self):
        """
        Prepare data for prediction using a pre-trained model
        """

        self.logger.info('Running predictor method')

        try:

            self.cleaned = self.data.copy()
            self.cleaned = self.data[self.selection]

            self.logger.info('Dropping any remaining NAs')
            self.logger.info('cleaned shape before dropping:\n%s',
                             self.cleaned.shape)

            self.cleaned = self.cleaned.dropna(how='any')

            self.logger.info('cleaned shape after dropping:\n%s',
                             self.cleaned.shape)

            for col in self.categories:

                encoder = LabelEncoder()
                encoder.fit(self.cleaned[col])

                self.logger.debug('%s converted to the following integers:\n%s',
                                  col, dict(zip(encoder.transform(self.cleaned[col]),
                                                self.cleaned[col])))

                self.cleaned.loc[:, col] = encoder.transform(self.cleaned.loc[:, col])

            self.logger.info('Dropping respondent_id from cleaned')

            self.cleaned.drop('respondent_id', axis=1, inplace=True)

            self.logger.info('Final shape of cleaned: %s', self.cleaned.shape)

        except Exception:
            self.logger.error('There was an error while subsetting survey data')
            raise

    raw_mapping = {
        'UserID':'respondent_id',
        'Started':'start_date',
        'Ended': 'end_date',
        'Page Path':'full_url',
        'Unique ID':'collector_id',
        'Q1. Are you using GOV.UK for professional or personal reasons?':'cat_work_or_personal',
        'Q2. What kind of work do you do?':'comment_what_work',
        'Q3. Describe why you came to GOV.UK todayPlease do not include personal or financial information, eg your National Insurance number or credit card details.':'comment_why_you_came',
        'Q4. Have you found what you were looking for?':'cat_found_looking_for',
        'Q5.1.':'cat_satisfaction',
        'Q6. Have you been anywhere else for help with this already?':'cat_anywhere_else_help',
        'Q7. Where did you go for help?':'comment_where_for_help',
        'Q8. If you wish to comment further, please do so here.Please do not include personal or financial information, eg your National Insurance number or credit card details.':'comment_further_comments',
        'Unnamed: 13':'comment_other_found_what',
        'Unnamed: 17':'comment_other_else_help',
        'dummy':'comment_other_where_for_help'
        }

    categories = [
        'org', 'section', 'cat_work_or_personal',
        'cat_satisfaction', 'cat_found_looking_for',
        'cat_anywhere_else_help'
    ]

    comments = [
        'comment_what_work', 'comment_why_you_came', 'comment_other_found_what',
        'comment_other_else_help', 'comment_other_where_for_help',
        'comment_where_for_help', 'comment_further_comments'
    ]

    codes = [
        'target'
    ]

    code_levels = [
        'ok', 'finding-general', 'service-problem', 'contact-government',
        'check-status', 'change-details', 'govuk-specific', 'compliment',
        'complaint-government', 'notify', 'internal', 'pay', 'report-issue',
        'address-problem', 'verify'
    ]

    selection = (['respondent_id', 'weekday', 'day', 'week', 'month', 'year', 'time_delta'] +
                 categories + [(x + '_len') for x in comments] +
                 [(x + '_nexcl') for x in comments] + [(x + '_capsratio') for x in comments])


def string_len(feature):
    """
     Calculate feature length of comment features

    :param feature: <pd.Series> Comment feature.
    """
    try:

        feature = feature.str.strip()
        feature = feature.str.lower()

        feature = feature.replace(r'\,\s?\,?$|none\,', 'none', regex=True)

        # Convert NaN to 'a'. Then when counted this will
        # be a 1. Whilst not 0, any entry with 1 is virtually
        # meaningless, so 1 is a profeaturey for 0.

        feature = pd.Series([len(y) for y in feature.fillna('a')])
        # Now normalise the scores

        ratio = (feature - feature.mean()) / (feature.max() - feature.min())

    except Exception:
        print('There was an error converting feature to string length column')
        raise
    return ratio

def string_capsratio(feature):
    """
    Calculate ratio of capitals to all characters

    :param feature: <pd.Series> Comment feature.
    """
    try:
        if not pd.isnull(feature):
            feature = sum([i.isupper() for i in feature]) / len(feature)
        else:
            feature = 0

    except Exception:
        print('There was an error creating capitals ratio on column: ' + feature)
        raise
    return feature

def string_nexcl(feature):
    """
    Calculate ratio of exclamations to all characters

    :param feature: <pd.Series> Comment feature.
    """
    try:
        if not pd.isnull(feature):
            feature = sum([i == '!' for i in feature]) / len(feature)
        else:
            feature = 0

    except Exception:
        print('There was an error creating n of exclamations on column')
        raise
    return feature

def clean_date(feature, format=None):
    """
    Convert feature to a datetime object

    :param feature: <pd.Series> Date feature.
    """
    try:
        feature = pd.to_datetime(feature, format=format)

    except Exception:
        print('There was an error cleaning the StartDate column!')
        raise
    return feature

def date_features(feature):
    """
    Create time features from date time feature

    :param feature: <pd.Series> Date feature.
    """
    try:
        feature = pd.to_datetime(feature)

        date_features = pd.DataFrame({
            'weekday' : feature.dt.weekday,
            'day' : feature.dt.day,
            'week' : feature.dt.week,
            'month' : feature.dt.month,
            'year' : feature.dt.year,
        })

    except Exception:
        print('There was an error creating date feature')
        raise
    return date_features

def clean_category(feature):
    """
    Clean categorical features

    :param feature: <pd.Series> Categorical feature.
    """
    try:

        feature = feature.apply(str)
        feature = feature.str.lower()
        feature = feature.replace(r'null|\#Value\!', 'none', regex=True)
        feature = feature.fillna('none')
        feature = pd.Series(feature)
        feature = feature.astype('category')
        
    except Exception:
        print('There was an error cleaning the column.')
        raise
    return feature

def clean_comment(feature):
    """
    Clean comment features

    :param feature: <pd.Series> Comment feature.
    """
    try:

        feature = feature.str.strip()
        feature = feature.str.lower()

        # Weirdness with some columns being filled with just a comma.
        # Is this due to improper handling of the csv file somewhere?

        feature = feature.replace(r'\,\s?\,?$|none\,', 'none', regex=True)
        feature = feature.fillna('none')

    except Exception:
        print('There was an error cleaning the column.')
        raise
    return feature

def clean_code(feature, levels):
    """
    Clean target feature

    :param feature: <pd.Series> Target feature.
    """
    try:

       # If the whole column is not null
       # i.e. we want to train rather than just predict

        if not pd.isnull(feature).sum() == len(feature):
            feature = feature.str.strip()
            feature = feature.str.lower()
            feature = feature.replace(r'\_', r'\-', regex=True)

            feature[~feature.isin(levels)] = np.nan
            feature = pd.Series(feature)
            feature = feature.astype('category')

    except Exception:
        print('There was an error cleaning column.')
        raise
    return feature

    def drop_sub(df):
        """
        Drop sub-heading created by survey platform
        """
        if df.iloc[0,].str.match('Open-Ended Response').sum():
            df.drop(0, inplace=True)
        return df
## Functions dealing with the API lookup

def lookup(r, page, index):
    """
    Helper function for parsing results from api lookup
    """

    try:
        if page == 'mainstream_browse_pages':
            x = r['results'][0][page][index]            
        elif page == 'organisations':
            x = r['results'][0][page][index]['title']
        else:
            print('page argument must be one of "organisations" or "mainstream_browse_pages"')
            sys.exit(1)
    except (IndexError, KeyError):
        x = 'null'
    return x

def get_org(page):
    """
    Perform lookup against the GOV.UK Content API
    """

    # argument x should be pd.Series of full length urls
    # Loop through each entry in the series

    url = ("https://www.gov.uk/api/search.json?filter_link[]"
           "=%s&fields=organisations&fields=mainstream_browse_pages" % page)

    #self.logger.info('Looking up ' + url)

    try:

        #url = "https://www.gov.uk/api/search.json?filter_link[]=%s&fields=y" % (x, y)

        # read JSON result into r
        r = requests.get(url).json()

        # chose the fields you want to scrape. This scrapes the first 5
        # instances of organisation, error checking as it goes
        # this exception syntax might not work in Python 3

        organisation0 = lookup(r, 'organisations', 0)
        organisation1 = lookup(r, 'organisations', 1)
        organisation2 = lookup(r, 'organisations', 2)
        organisation3 = lookup(r, 'organisations', 3)
        organisation4 = lookup(r, 'organisations', 4)
        section0 = lookup(r, 'mainstream_browse_pages', 0)
        section1 = lookup(r, 'mainstream_browse_pages', 1)
        section2 = lookup(r, 'mainstream_browse_pages', 2)
        section3 = lookup(r, 'mainstream_browse_pages', 3)

        row = [organisation0,
               organisation1,
               organisation2,
               organisation3,
               organisation4,
               section0,
               section1,
               section2,
               section3]

        return row

    except Exception:
        print('Error looking up ' + url)
        print('Returning "none"')
        row = ['none'] * 9
        return row

## Functions dealing with developing a time difference feature

def normalise(x):

    x = (x - np.mean(x)) / np.std(x)
    return x

def time_delta(x,y):

    # Expects datetime objects

    delta = x - y
    # Required for old versions!
    #delta = np.timedelta64(delta, 's')
    #delta = delta.astype('int')
    delta = delta.astype('timedelta64[s]')
    delta = delta.astype('int')

    # normalise statment moved to method to keep this function simple

    return delta

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
    return found
