#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

path = 'D:/CS/ML/DataFountain/unicom/'
train = pd.read_csv('%s%s' %(path, 'train_all.csv'))
test = pd.read_csv('%s%s' %(path, 'republish_test.csv'))

from numpy.core.umath_tests import inner1d
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

train['2_total_fee'] = train['2_total_fee'].replace('\\N', 0)
train['3_total_fee'] = train['3_total_fee'].replace('\\N', 0)
train['age'] = train['age'].replace('\\N', 0)
train['gender'] = train['gender'].replace('\\N', 0)
test['2_total_fee'] = test['2_total_fee'].replace('\\N', 0)
test['3_total_fee'] = test['3_total_fee'].replace('\\N', 0)
test['age'] = test['age'].replace('\\N', 0)
test['gender'] = test['gender'].replace('\\N', 0)

train['gender'] = train['gender'].apply(lambda x : int(x))
test['gender'] = test['gender'].apply(lambda x : int(x))
train['age'] = train['age'].apply(lambda x : int(x))
test['age'] = test['age'].apply(lambda x : int(x))
train['2_total_fee'] = train['2_total_fee'].apply(lambda x : float(x))
test['2_total_fee'] = test['2_total_fee'].apply(lambda x : float(x))
train['3_total_fee'] = train['3_total_fee'].apply(lambda x : float(x))
test['3_total_fee'] = test['3_total_fee'].apply(lambda x : float(x))

ntrain = train.shape[0]
ntest = test.shape[0]

train_copy = train.copy()
train_copy.drop('current_service', axis = 1, inplace=True)

train_test = pd.concat((train_copy, test)).reset_index(drop=True)

train_test.drop('user_id', axis = 1, inplace=True)
#train_test.drop('net_service', axis = 1, inplace=True)
#train_test.drop('many_over_bill', axis = 1, inplace=True)

label = train.pop('current_service')
le = LabelEncoder()
train_y = le.fit_transform(label)

train_test['service_type'] = train_test['service_type'].astype(str)
train_test['is_mix_service'] = train_test['is_mix_service'].astype(str)
train_test['many_over_bill'] = train_test['many_over_bill'].astype(str)
train_test['contract_type'] = train_test['contract_type'].astype(str)
train_test['is_promise_low_consume'] = train_test['is_promise_low_consume'].astype(str)
train_test['net_service'] = train_test['net_service'].astype(str)
train_test['complaint_level'] = train_test['complaint_level'].astype(str)
train_test['gender'] = train_test['gender'].astype(str)

train_test = pd.get_dummies(train_test)
train_X = train_test[:ntrain]
test_X = train_test[ntrain:]


class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=2,
                                   return_train_score=True)
        grid_search.fit(X, y)
        print(grid_search.best_params_, grid_search.best_score_)
        grid_search.cv_results_['mean_test_score'] = grid_search.cv_results_['mean_test_score']#np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'std_test_score', 'mean_test_score']])

param_test1 = {'n_estimators':[100], 'n_jobs': [-1], 'max_depth':[36], 'min_samples_split':[2],'min_samples_leaf':[1], 'max_features':[14]}
#grid(RandomForestClassifier()).grid_get(train_X,train_y,param_test1)

param_rf = {'num_leaves':[180], 'max_depth':[8], 'learning_rate':[0.1],'seed':[1500],'colsample_bytree':[0.6],'subsample':[0.7]}
#grid(lgb.LGBMClassifier(bjective='multiclass', boosting_type='gbdt')).grid_get(train_X,train_y,param_rf)

'''
def model_cv(model, x, y):
    score = cross_val_score(model, x, y,cv=5)#,scoring='f1'
    return score

models = [RandomForestClassifier()]
names = ['rf']

for name, model in zip(names, models):
    Score = model_cv(model, train_X,train_y)
    print("{}: {:.6f}, {:.4f}".format(name,Score.mean(),Score.std()))
'''
'''
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=36, min_samples_split=2, min_samples_leaf=1, max_features=14, oob_score=True)
rf_model.fit(train_X, train_y)
pred = rf_model.predict(test_X)
pred = le.inverse_transform(pred)
test['predict'] = pred
test[['user_id', 'predict']].to_csv('./result/rf.csv', index=False)
'''


clf = lgb.LGBMClassifier(bjective='multiclass',boosting_type='gbdt',num_leaves=35,max_depth=8,learning_rate=0.05,
                         seed=2018,colsample_bytree=0.8,subsample=0.9,n_estimators=2000)
clf1 = lgb.LGBMClassifier(bjective='multiclass',boosting_type='gbdt',num_leaves=180,max_depth=8,learning_rate=0.1,
                         seed=1500,colsample_bytree=0.6,subsample=0.7,n_estimators=2000)
clf1.fit(train_X, train_y)
pred = clf1.predict(test_X)
pred = le.inverse_transform(pred)
test['predict'] = pred
test[['user_id', 'predict']].to_csv('./result/lgb1.csv', index=False)


'''
clf = XGBClassifier(max_depth=12, learning_rate=0.05,
                            n_estimators=752, silent=True,
                            objective="multi:softmax",
                            nthread=4, gamma=0,
                            max_delta_step=0, subsample=1, colsample_bytree=0.9, colsample_bylevel=0.9,
                            reg_alpha=1, reg_lambda=1, scale_pos_weight=1,
                            base_score=0.5, seed=2018, missing=None,num_class=15)
clf.fit(train_X, train_y)
pred = clf.predict(test_X)
pred = le.inverse_transform(pred)
test['predict'] = pred
test[['user_id', 'predict']].to_csv('./result/xgbst.csv', index=False)
'''