#!/usr/bin/Python
# -*- coding: utf-8 -*-
import xgboost as xgb


class XGB:
    '''
    Xgboost
    Experiement:
        1).
        param:  max_depth=2, learning_rate=0.01,
        auc:    origin_min_max_scaling: 0.899622
    '''

    params = {
        'max_depth': 2,
        'n_estimators': 999999,
        'colsample_bytree': 0.3,
        'learning_rate': 0.02,
        'objective': 'binary:logistic',
        'n_jobs': -1,
    }

    def __init__(self, model_dir='', model_name=None):
        self.__model = xgb.XGBClassifier(**self.params)

    def train(self, train_x, train_y, val_x, val_y):
        self.__model.fit(train_x, train_y,
                         eval_set=[(val_x, val_y)],
                         verbose=True,
                         eval_metric='auc',
                         early_stopping_rounds=1000)

    def predict(self, test_x, test_y=None):
        return self.__model.predict_proba(test_x)[:, 1]

    def save(self):
        return
