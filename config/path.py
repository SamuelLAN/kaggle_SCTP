#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os

# the root directory for all data
__PATH_DATA_DIR = '/Users/samuellin/Desktop/kaggle_data/'

# the directory path for original data
PATH_TRAIN_DATA = os.path.join(__PATH_DATA_DIR, 'origin_data', 'train.csv')
PATH_TEST_DATA = os.path.join(__PATH_DATA_DIR, 'origin_data', 'test.csv')

# directory for saving the test predictions
PATH_RESULT_DIR = os.path.join(__PATH_DATA_DIR, 'results')

# directory for saving the validation prediction and test prediction,
#   and it will be used by the model fusion
__PATH_FUSION_DIR = os.path.join(__PATH_DATA_DIR, 'fusion')

# data for the model fusion
PATH_FUSION_TRAIN = os.path.join(__PATH_FUSION_DIR, 'train.csv')
PATH_FUSION_TEST = os.path.join(__PATH_FUSION_DIR, 'test.csv')

# the log file path, record all the models results and params
PATH_LOG = os.path.join(__PATH_DATA_DIR, 'model.log')

# directory for saving models
__PATH_MODEL_DIR = os.path.join(__PATH_DATA_DIR, 'models')
PATH_MODEL_LGB = os.path.join(__PATH_MODEL_DIR, 'LightGBM')
PATH_MODEL_XGB = os.path.join(__PATH_MODEL_DIR, 'XgBoost')
PATH_MODEL_CB = os.path.join(__PATH_MODEL_DIR, 'CatBoost')


def mkdir_time(upper_path, _time):
    """ create directory with time (for save model) """
    dir_path = os.path.join(upper_path, _time)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path
