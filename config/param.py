#!/usr/bin/Python
# -*- coding: utf-8 -*-

# param for func duplicate
ratio_duplicate = 3.5

# param for func shuffle_same_dim
ratio_aug_minority = 2.0
ratio_aug_majority = 0.0

# param for func smote
ratio_smote = 0.2

# param for func under_sample
ratio_under_sample_major = 0.75

K_FOLD = 5

RANDOM_STATE = 1

# for log, record the params
DATA_PARAMS = {
    'ratio_duplicate': ratio_duplicate,
    'ratio_aug_minority': ratio_aug_minority,
    'ratio_aug_majority': ratio_aug_majority,
    'k_fold': K_FOLD,
    'random_state': RANDOM_STATE,
    'AUG_PROCESSORS': 'augment.duplicate_shuffle_same_dim, augment.under_sample',
    'NORM_PROCESSORS': 'norm.min_max_scaling, norm.standardization',
}
