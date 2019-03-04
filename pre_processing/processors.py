#!/usr/bin/Python
# -*- coding: utf-8 -*-
from lib.ml import Norm

class Processors:
    def __init__(self):
        pass

    @staticmethod
    def standardization(train_data, val_data, test_data):
        train_data, means, stds = Norm.standardization(train_data, 1)
        # val_data may be null
        if val_data.any():
            val_data, _, _ = Norm.standardization(val_data, 1, means, stds)
        test_data, _, _ = Norm.standardization(test_data, 1, means, stds)
        return train_data, val_data, test_data

    @staticmethod
    def min_max_scaling(train_data, val_data, test_data):
        train_data, minimums, maximums = Norm.min_max_scaling(train_data, 1)
        # val_data may be null
        if val_data.any():
            val_data, _, _ = Norm.min_max_scaling(val_data, 1, minimums, maximums)
        test_data, _, _ = Norm.min_max_scaling(test_data, 1, minimums, maximums)
        return train_data, val_data, test_data
