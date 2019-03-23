#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)

from models.eval_base import Eval
from pre_processing.processors import Processors
from models.lgb import LGB
from models.xgb import XGB
from models.cb import CB


class EvalML(Eval):
    MODEL_NAME = 'cb'
    DATA_CACHE_NAME = 'add_lda_standardization_min_max_scaling'
    PROCESSOR_LIST = []

    def init_model(self):
        self.model = CB(self.model_name)
        # self.model = LGB(self.model_name)
        # self.model = XGB(self.model_name)


o_model = EvalML()
o_model.train()
o_model.predict()
