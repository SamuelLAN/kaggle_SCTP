#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import config.path as path

__FUSION_USE_ORIGIN_FEAT = True


def train_data():
    """
    Get original train data
    :return: ID_code, X, y
    """
    print('Start Loading original train data ...')
    data = pd.read_csv(path.PATH_TRAIN_DATA)
    print('Finish loading original train data')
    return data.values[:, 0], np.cast['float32'](data.values[:, 2:]), np.cast['int32'](data.values[:, 1])


def test_data():
    """
    Get original test data
    :return: ID_code, X
    """
    print('Start Loading original train data ...')
    data = pd.read_csv(path.PATH_TEST_DATA)
    print('Finish loading original train data')
    return data.values[:, 0], np.cast['float32'](data.values[:, 1:])


def fusion_train_data(choose_list=[]):
    """
    Get training data for model fusion
    :param choose_list: [result_name, ...] (option)
    :return: X, y
    """
    # load validation predictions of all model results
    data = pd.read_csv(path.PATH_FUSION_TRAIN).values
    data = __choose_data(data, choose_list)
    X = np.transpose(data[:, 1:], axes=[1, 0])

    # load labels
    data = pd.read_csv(path.PATH_TRAIN_DATA).values
    y = np.expand_dims(data[:, 1], axis=-1)

    # add features to X
    if __FUSION_USE_ORIGIN_FEAT:
        X = np.hstack([X, data[:, 2:]])

    return np.cast['float32'](X), np.cast['int32'](y)


def fusion_test_data(choose_list=[]):
    """
    Get testing data for model fusion
    :param choose_list: [result_name, ...] (option)
    :return: X, ids
    """
    # load test predictions of all model results
    data = pd.read_csv(path.PATH_FUSION_TEST).values
    data = __choose_data(data, choose_list)
    X = np.transpose(data[:, 1:], axes=[1, 0])

    # load ID_code
    data = pd.read_csv(path.PATH_TEST_DATA).values
    ids = data[:, 0]

    # add features to X
    if __FUSION_USE_ORIGIN_FEAT:
        X = np.hstack([X, data[:, 1:]])

    return np.cast['float32'](X), ids


def __choose_data(data, choose_list):
    """ choose data for loading fusion data """
    if not choose_list:
        return data

    name_list = data[:, 0]
    choose_data = []
    for i, name in enumerate(name_list):
        if name not in choose_list:
            continue
        choose_data.append(data[i, :])
    return np.asarray(choose_data)
