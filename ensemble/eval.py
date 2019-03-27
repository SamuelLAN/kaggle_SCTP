#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import time

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)

from pre_processing.process import EnsembleReadData
from models.lgb import LGB
from models.eval_base import Eval


class EvalEnsemble:
    MODEL_NAME = 'ensemble_lgb'
    TIME = time.strftime('%Y_%m_%d_%H_%M_%S')

    def __init__(self):
        data = EnsembleReadData(0.9, [
            # 'origin_7_min_max_scaling_aug_4.0_standardization',
        ])
        self.__train_x, self.__train_y = data.train_data
        self.__val_x, self.__val_y = data.val_data
        self.__test_x, self.__test_ids = data.test_data

        self.model_name = '%s_%s' % (self.MODEL_NAME, self.TIME)
        self.__init_model()

    def __init_model(self):
        # self.model = LGB(self.model_name)
        self.model = LGB('lgb')

    def train(self):
        self.model.train(self.__train_x, self.__train_y, self.__val_x, self.__val_y)
        self.model.save()

    def predict(self):
        print('\nStart predicting test data ...')
        prob = self.model.predict(self.__test_x)
        print('Finish predicting')
        # save result to csv
        Eval.gen_csv(self.model_name, self.__test_ids, prob)


o_model = EvalEnsemble()
o_model.train()
o_model.predict()

print('\ndone')
