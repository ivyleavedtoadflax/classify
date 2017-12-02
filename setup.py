from setuptools import setup, find_packages

setup(name='classifyintents',
      version='0.6.1',
      description='Data wrangling for classification of the GOV.UK intents survey',
      url='http://github.com/ukgovdatascience/classifyintents',
      author='Matthew Upson',
      packages=find_packages(exclude=['tests']),
      author_email='matthew.upson@digital.cabinet-office.gov.uk',
      license='MIT',
      zip_safe=False,
      install_requires=['pandas', 'numpy', 'scikit-learn']
     )
