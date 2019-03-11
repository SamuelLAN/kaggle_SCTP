#!/usr/bin/Python
# -*- coding: utf-8 -*-
import lightgbm as lgb


class LGB:
    '''
    LightGBM
    Experiement:
        1).
        param:  max_depth=-1, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=2, objective='binary'
        auc:    origin 0.899
                origin_min_max_scaling: 0.901
        2).
        param:  max_depth=-1, n_estimators=999999, learning_rate=0.01,
                colsample_bytree=0.3, num_leaves=5, objective='binary'
        auc:    origin_min_max_scaling: 0.899192
        3).
        param:  max_depth=-1, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=3, objective='binary'
        auc:    origin_min_max_scaling: 0.899802
        4).
        param:  max_depth=10, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=2, objective='binary'
        auc:    origin_min_max_scaling: 0.900032
        5).
        param:  max_depth=20, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=2, objective='binary'
        auc:    origin_min_max_scaling: 0.900032
        6).
        param:  max_depth=20, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=2, objective='binary'
        auc:    origin_min_max_scaling: 0.900032
        7).
        param:  max_depth=-1, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=2, objective='binary', boosting_type='goss'
        auc:    origin_min_max_scaling: 0.901438
    '''

    def __init__(self, **kwargs):
        self.__model = lgb.LGBMClassifier(max_depth=-1,
                                          n_estimators=999999,
                                          learning_rate=0.02,
                                          colsample_bytree=0.3,
                                          num_leaves=2,
                                          boosting_type='goss',  # 'gbdt',
                                          metric='auc',
                                          objective='binary',
                                          n_jobs=-1)

    def train(self, train_x, train_y, val_x, val_y):
        self.__model.fit(train_x, train_y,
                         eval_set=[(val_x, val_y)],
                         verbose=1,
                         early_stopping_rounds=1000)

    def predict(self, test_x, test_y=None):
        return self.__model.predict(test_x)
