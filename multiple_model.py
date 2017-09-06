import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
import itertools
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed
pd.set_option('display.max_columns', None)
import get_all_data
import utility



models = {
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced', max_depth=13),
    #'NeuralNetwork': MLPClassifier(),
    'GradientBoosting': GradientBoostingClassifier(max_depth=1, n_estimators=50),
    #'SupportVectorMachine': SVC(class_weight='balanced', probability=True),
    #'KNearestNeighbor': KNeighborsClassifier(n_neighbors=5)
}

params1 = {
    'DecisionTree': {'max_depth': [1]},
    #'NeuralNetwork': {'hidden_layer_sizes': [(160, 112, 112, 112, 112)]},
    'GradientBoosting': {'max_depth': [1]},
    #'SupportVectorMachine': {'C': [1]},
    #'KNearestNeighbor': {'n_neighbors': [7]}
}


params2 = {
    'DecisionTree': {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]},
    'NeuralNetwork': {'hidden_layer_sizes': [(160), (160, 112, 112), (160, 112, 112, 112, 112), (160, 112, 112, 112, 112, 112, 112)]},
    'GradientBoosting': {'max_depth': [1, 2, 3]},
    'SupportVectorMachine': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'KNearestNeighbor': {'n_neighbors': [3,7,11]}}



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



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], scoring=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, \
                                                            train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt



def complexity():
    helper1 = EstimatorSelectionHelper(models, params2)
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    training_features, test_features, \
    training_target, test_target, = train_test_split(train, target, test_size=0.33, random_state=778)
    X_train, X_val, y_train, y_val = train_test_split(training_features, training_target)
    helper1.fit(X_train, y_train, scoring='f1', n_jobs=1)
    helper1.score_summary(sort_by='min_score')




all_data = get_all_data.get_all_data()
train, target = get_all_data.process_data(all_data)
for model in models:
    title = 'Learning Curves: ' + model
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    print(title)
    plot_learning_curve(models[model], title, train, target, ylim=(0.1, 1.01), cv=cv, n_jobs=1)
    plt.show()


#analysis2 = complexity()