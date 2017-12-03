# coding: utf-8
import logging
import logging.config
import classifyintents

logging.config.fileConfig('test_logging.conf')
logger = logging.getLogger('test_trainer')

class TestTrainerMethod(object):

    @classmethod
    def setup_class(self):

        print('Running TestTrainer class')

        self.a = classifyintents.survey()
        self.a.load('test_data/raw_test_data_classified.csv')
        self.a.clean_raw()
        self.a.clean_urls()

    def test_trainer_method_preserves_target_variable(self):

        self.a.trainer(['ok'])
        assert 'target' in self.a.cleaned.columns.tolist()


    def test_trainer_method_converts_one_code_correctly(self):

        self.a.trainer(['ok'])
        logger.info(self.a.cleaned.head())

        assert set(self.a.cleaned.target.tolist()) == set([1, 0])


    def test_trainer_method_converts_two_codes_correctly(self):

        self.a.trainer(['ok', 'finding-general'])

        assert set(self.a.cleaned.target.tolist()) == set([1, 0])
