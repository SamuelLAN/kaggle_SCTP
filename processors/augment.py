#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
from config import param
from lib.ml import Sampling


def duplicate(X, y):
    ''' Duplicate the minority several times '''
    print('Start duplication ... ')

    ar_augment_x = Sampling.duplicate(X, y, 1, param.ratio_duplicate)
    ar_augment_y = np.ones([len(ar_augment_x), ])

    print('Finish duplication')

    # stack train data and shuffle it
    return __stack_and_shuffle([X, ar_augment_x], [y, ar_augment_y])


def shuffle_same_dim(X, y):
    print('Start shuffling same dimension and augmentation ... ')

    ar_augment_minor_x = Sampling.shuffle_same_dim(X, y, 1, param.ratio_aug_minority)
    ar_augment_minor_y = np.ones([len(ar_augment_minor_x), ])

    stack_list_x = [X, ar_augment_minor_x]
    stack_list_y = [y, ar_augment_minor_y]

    if param.ratio_aug_majority > 0:
        ar_augment_major_x = Sampling.shuffle_same_dim(X, y, 0, param.ratio_aug_majority)
        ar_augment_major_y = np.ones([len(ar_augment_major_x), ])

        stack_list_x.append(ar_augment_major_x)
        stack_list_y.append(ar_augment_major_y)

    print('Finish shuffling and augmentation')

    # stack train data and shuffle it
    return __stack_and_shuffle(stack_list_x, stack_list_y)


def duplicate_shuffle_same_dim(X, y):
    print('Start duplication ... ')

    ar_duplication_x = Sampling.duplicate(X, y, 1, param.ratio_duplicate)
    ar_duplication_y = np.ones([len(ar_duplication_x), ])

    print('Finish duplication\nStart shuffling same dimension and augmentation ... ')

    ar_augment_minor_x = Sampling.shuffle_same_dim(X, y, 1, param.ratio_aug_minority)
    ar_augment_minor_y = np.ones([len(ar_augment_minor_x), ])

    stack_list_x = [X, ar_duplication_x, ar_augment_minor_x]
    stack_list_y = [y, ar_duplication_y, ar_augment_minor_y]

    if param.ratio_aug_majority > 0:
        ar_augment_major_x = Sampling.shuffle_same_dim(X, y, 0, param.ratio_aug_majority)
        ar_augment_major_y = np.ones([len(ar_augment_major_x), ])

        stack_list_x.append(ar_augment_major_x)
        stack_list_y.append(ar_augment_major_y)

    print('Finish shuffling and augmentation')

    # stack train data and shuffle it
    return __stack_and_shuffle(stack_list_x, stack_list_y)


def smote(X, y):
    print('Start smote ... ')

    # over-sample by smote
    synthetic_train_x = Sampling.smote(X, y, 1, 5, param.ratio_smote)
    synthetic_train_y = np.ones([len(synthetic_train_x), ])

    print('Finish smote')

    # stack train data and shuffle it
    return __stack_and_shuffle([X, synthetic_train_x], [y, synthetic_train_y])


def under_sample(X, y):
    ''' under sample the majority class '''
    return Sampling.under_sample(X, y, 0, param.ratio_under_sample_major)


def __stack_and_shuffle(stack_list_x, stack_list_y):
    ''' stack train data and shuffle it '''
    # add the over-sampling data to train data
    train_x = np.vstack(stack_list_x)
    train_y = np.hstack(stack_list_y)

    # shuffle train data
    return Sampling.shuffle(train_x, train_y)
