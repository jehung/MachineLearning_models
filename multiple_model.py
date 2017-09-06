import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
import itertools
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed
pd.set_option('display.max_columns', None)
import get_all_data
import utility




#d = {'train': None, 'cv set': None, 'test': None}


class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=10, n_jobs=-1, verbose=5, scoring=None, refit=True):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
            return pd.Series({**params, **d})

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters)
                for k in self.keys
                for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        print(df[columns])
        return df[columns]


    def train_size_summary(self, sort_by='mean_score'):
        def row_report(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
            return pd.Series({**params, **d})

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters)
                for k in self.keys
                for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        print(df[columns])
        return df[columns]


models = {
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
    'NeuralNetwork': MLPClassifier(),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
    'SupportVectorMachine': SVC(class_weight='balanced', probability=True),
    'KNearestNeighbor': KNeighborsClassifier(n_neighbors=5)
}

params1 = {
    'DecisionTree': {'max_depth': [13]},
    'NeuralNetwork': {'hidden_layer_sizes': [(160, 112, 112, 112, 112)]},
    'GradientBoosting': {'max_depth': [3]},
    'SupportVectorMachine': {'C': [1]},
    'KNearestNeighbor': {'n_neighbors': [7]}
}


params2 = {
    'DecisionTree': {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]},
    'NeuralNetwork': {'hidden_layer_sizes': [(160), (160, 112, 112), (160, 112, 112, 112, 112), (160, 112, 112, 112, 112, 112, 112)]},
    'GradientBoosting': {'max_depth': [1, 2, 3]},
    'SupportVectorMachine': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'KNearestNeighbor': {'n_neighbors': [3,7,11]}}



def train_size(train=None, target=None, size=0):
    helper1 = EstimatorSelectionHelper(models, params1)
    X_train, X_val, y_train, y_val = train_test_split(train, target, train_size=size)
    helper1.fit(X_train, y_train, scoring='f1')
    d['model'] = helper1.key
    d['train'] = f1_score(y_train, clf.predict(X_train), average='weighted')
    d['cv set'] = f1_score(y_val, clf.predict(X_val), average='weighted')
    d['test'] = f1_score(test_target, clf.predict(test_features), average='weighted')
    return d


def complexity():
    helper1 = EstimatorSelectionHelper(models, params2)
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    training_features, test_features, \
    training_target, test_target, = train_test_split(train, target, test_size=0.33, random_state=778)
    X_train, X_val, y_train, y_val = train_test_split(training_features, training_target)
    helper1.fit(X_train, y_train, scoring='f1', n_jobs=1)
    helper1.score_summary(sort_by='min_score')


#analysis1 = train_size()
if  __name__== '__main__':
    d = {'model': None, 'train': None, 'cv set': None, 'test': None}
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    training_features, test_features, \
    training_target, test_target, = train_test_split(train, target, test_size=0.33, random_state=778)
    df = Parallel(n_jobs=6)(delayed(train_size)(train=training_features, target=training_target, size=size) for size in np.arange(0.3, 1, 0.1))
    utility.merge_dict(df)
    print(df)
#analysis2 = complexity()