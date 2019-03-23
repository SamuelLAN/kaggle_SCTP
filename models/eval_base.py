#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
PATH_RESULT = os.path.join(PATH_CUR, 'results')
sys.path.append(PATH_PRJ)

# from models.lgb import LGB
from pre_processing.process import Data


class Eval:
    MODEL_NAME = 'lgb'
    DATA_CACHE_NAME = 'origin'
    PROCESSOR_LIST = []

    def __init__(self):
        # Load train data
        self.data = Data(self.PROCESSOR_LIST, cache_name=self.DATA_CACHE_NAME, new_cache_name='')
        self.train_x, self.train_y = self.data.train_data()
        self.val_x, self.val_y = self.data.val_data()
        self.test_x, self.test_y = self.data.test_data()
        self.real_test_x, self.real_test_ids = self.data.real_test_data()

        self.model_name = '%s_%s' % (self.MODEL_NAME, self.DATA_CACHE_NAME)
        self.init_model()

    def init_model(self):
        # self.model = LGB(self.model_name)
        self.model = None

    def train(self):
        self.model.train(self.train_x, self.train_y, self.val_x, self.val_y)
        self.model.test_auc(self.test_x, self.test_y)
        self.model.save()

    def predict(self):
        print('\nStart predicting test data ...')
        prob = self.model.predict(self.real_test_x)
        print('Finish predicting')
        # save result to csv
        self.gen_csv(self.model_name, self.real_test_ids, prob)

    @staticmethod
    def gen_csv(name, ids, prob):
        print('\nStart generating %s.csv ...' % name)

        # integrate content
        content = 'ID_code,target\n'
        for i in range(len(ids)):
            content += '%s,%f\n' % (ids[i], prob[i])

        # write content to csv file
        with open(os.path.join(PATH_RESULT, name + '.csv'), 'wb') as f:
            f.write(content)

        print('Finish generating csv')
