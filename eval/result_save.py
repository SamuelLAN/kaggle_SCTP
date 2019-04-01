#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from config import path
from config.param import DATA_PARAMS


def __get_result_path(result_name, is_best_fold=False):
    return os.path.join(path.PATH_RESULT_DIR, result_name + ('_best' if is_best_fold else '') + '.csv')


def log_results(model_name, result_name, val_auc_list, val_pred_all, y, model_params=''):
    """
    show the validation result to console
     and save the results and the relevant params to the log file
    """
    data = (model_name,
            result_name,
            val_auc_list,
            np.mean(val_auc_list),
            roc_auc_score(np.cast['float32'](y), np.cast['float32'](val_pred_all)),
            __get_result_path(result_name),
            __get_result_path(result_name, True),
            model_params,
            DATA_PARAMS,
            time.strftime('%Y.%m.%d %H:%M:%S'))

    output = 'Model_name: %s\n' \
             'Result_name:%s\n' \
             'Each fold auc results: %s\n' \
             'Mean of each fold auc: %f\n' \
             'Auc of val predict of all data: %f\n' \
             'Result file path: %s\n' \
             'Result of best fold path: %s\n' \
             'Model params: %s\n' \
             'Data params: %s\n' \
             'Time: %s\n\n' % data

    # show to the console
    print(output)
    # save to the log file
    with open(path.PATH_LOG, 'ab') as f:
        f.write(output)


def gen_csv(result_name, ids, prob, is_best_fold=False):
    """ save the test prediction result to csv """
    print('\nStart generating %s.csv ...' % result_name)

    # integrate content
    content = 'ID_code,target\n'
    for i in range(len(ids)):
        content += '%s,%f\n' % (ids[i], prob[i])

    # write content to csv file
    with open(__get_result_path(result_name, is_best_fold), 'wb') as f:
        f.write(content)

    print('Finish generating csv')


def save_for_ensemble(result_name, val_pred, test_pred):
    """ save the validation and testing predictions so that it can be used by model fusion """
    print('Start saving results for model fusion ...')

    train_content = reduce(lambda x, y: x + ',' + str(y), val_pred, result_name)
    with open(path.PATH_FUSION_TRAIN, 'ab') as f:
        f.write(train_content + '\n')

    test_content = reduce(lambda x, y: x + ',' + str(y), test_pred, result_name)
    with open(path.PATH_FUSION_TEST, 'ab') as f:
        f.write(test_content + '\n')

    print('Finish saving')
