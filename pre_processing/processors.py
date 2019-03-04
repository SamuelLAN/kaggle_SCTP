#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)

from lib.ml import Norm


class Processors:
    def __init__(self):
        pass

    @staticmethod
    def standardization(train_data, val_data, test_data):
        train_data, means, stds = Norm.standardization(train_data, 0)
        # val_data may be null
        if val_data.any():
            val_data, _, _ = Norm.standardization(val_data, 0, means, stds)
        test_data, _, _ = Norm.standardization(test_data, 0, means, stds)
        del means, stds
        return train_data, val_data, test_data

    @staticmethod
    def min_max_scaling(train_data, val_data, test_data):
        train_data, minimums, maximums = Norm.min_max_scaling(train_data, 0)
        # val_data may be null
        if val_data.any():
            val_data, _, _ = Norm.min_max_scaling(val_data, 0, minimums, maximums)
        test_data, _, _ = Norm.min_max_scaling(test_data, 0, minimums, maximums)
        del minimums, maximums
        return train_data, val_data, test_data
