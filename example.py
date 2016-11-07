# coding: utf-8
from classify import *

print('instantiating class')

a = survey()

print('Loading full_data')

a.load('../data/full_data.csv')

a.raw = a.raw[1:200]

#a.raw = a.raw.loc[[str(i).strip().startswith('/government/world') for i in a.raw['Custom Data']],:]

print('Applying survey.clean_raw()...')

a.clean_raw()

print('Cleaning urls...')

a.clean_urls()

#a.data = a.data[a.data.org == 'Foreign & Commonwealth Office']

a.api_lookup()

a.data[['full_url','page','org','section']].to_csv('test-cases.csv')
print(a.data[['full_url','page','org','section']].head(100))


