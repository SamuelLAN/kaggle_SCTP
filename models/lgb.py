#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_MODEL = os.path.join(PATH_CUR, 'saved_models')


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
        8).
        param:  max_depth=-1, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=5, objective='binary', boosting_type='goss'
        auc:    origin_min_max_scaling: 0.900925
        9).
        param:  max_depth=-1, n_estimators=999999, learning_rate=0.02,
                colsample_bytree=0.3, num_leaves=2, objective='binary', boosting_type='dart'
        auc:    origin_min_max_scaling: 0.899511
        10).
        param:  max_depth=-1, n_estimators=999999, learning_rate=0.02, reg_lambda=0.1,
                colsample_bytree=0.3, num_leaves=2, objective='binary', boosting_type='goss'
        auc:    origin_min_max_scaling: 0.901238
    '''

    def __init__(self, model_name=None):
        self.__model_path = os.path.join(PATH_MODEL, model_name + '.model' if model_name else 'lgb.model')
        if model_name and os.path.isfile(self.__model_path):
            self.__has_train = True
            self.__model = lgb.Booster(model_file=self.__model_path)
        else:
            self.__has_train = False
            self.__model = lgb.LGBMClassifier(max_depth=-1,
                                              n_estimators=999999,
                                              learning_rate=0.02,
                                              colsample_bytree=0.3,
                                              num_leaves=2,
                                              boosting_type='goss',  # 'gbdt',
                                              metric='auc',
                                              objective='binary',
                                              # reg_lambda=0.1,
                                              n_jobs=-1)

    def train(self, train_x, train_y, val_x, val_y):
        if self.__has_train:
            return
        self.__model.fit(train_x, train_y,
                         eval_set=[(val_x, val_y)],
                         verbose=True,
                         early_stopping_rounds=1000)

    def predict(self, X):
        if self.__has_train:
            return self.__model.predict(X)
        return self.__model.predict_proba(X)[:, 1]

    def test_auc(self, test_x, test_y):
        test_output = self.predict(test_x)
        auc = roc_auc_score(test_y, test_output)
        print('test auc: %f' % auc)
        return auc

    def save(self):
        if self.__has_train:
            return
        self.__model.booster_.save_model(self.__model_path)
