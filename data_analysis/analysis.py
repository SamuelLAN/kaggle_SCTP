#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import pandas as pd

PATH_CUR = os.path.split(__file__)[0]
PATH_PRJ = os.path.split(PATH_CUR)[0]
PATH_TRAIN_DATA = os.path.join(PATH_PRJ, 'dataset', 'train.csv')


class Analysis:
    '''
    Data Analysis
    '''

    def __init__(self):
        self.__data = pd.read_csv(PATH_TRAIN_DATA)

    pass
