import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from dateutil.relativedelta import relativedelta
from sklearn.cross_validation import train_test_split
import sklearn.tree as tree
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed
pd.set_option('display.max_columns', None)
import get_all_data
import utility



def train_size_boost(X, y):
    d = {'train': [], 'cv set': [], 'test': []}
    training_features, test_features, \
    training_target, test_target, = train_test_split(X, y, test_size=0.2, random_state=12)
    for size in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print('size', size)
        X_train, X_val, y_train, y_val = train_test_split(training_features, training_target, train_size=size)
        smote = SMOTE(ratio=1)
        X_train_res, y_train_res = smote.fit_sample(X_train, y_train)

        clf = GradientBoostingClassifier(n_estimators=50)
        clf.fit(X_train_res, y_train_res)

        from sklearn.metrics import f1_score

        d['train'].append(f1_score(y_train_res, clf.predict(X_train_res), average='weighted'))
        d['cv set'].append(f1_score(y_val, clf.predict(X_val), average='weighted'))
        d['test'].append(f1_score(test_target, clf.predict(test_features), average='weighted'))

    return d



def complexity_boost(X, y):
    #X_train, y_train, X_test, y_test = train_test_split(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=778)
    smote = SMOTE(ratio=1)
    X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
    print('Start Decision Tree Search')
    boost = GradientBoostingClassifier(n_estimators=100)
    pipe = Pipeline([('smote', smote), ('boost', boost)])
    param_grid = {'boost__max_depth': [1, 2, 3]}
    #sss = StratifiedShuffleSplit(n_splits=500, test_size=0.2)  ## no need for this given 50000 random sample
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=6, cv=10, scoring='neg_log_loss',verbose=5)
    grid_search.fit(X_train_res, y_train_res)
    clf = grid_search.best_estimator_
    print('clf', clf)
    print('best_score', grid_search.best_score_)
    y_pred = clf.predict(X_test)
    check_pred = clf.predict(X_train)
    target_names = ['Not delinq', 'Delinq']
    print(classification_report(y_test, y_pred, target_names=target_names))
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=target_names,
                      title='Confusion matrix, without normalization')
    plt.show()
    return clf, clf.predict(X_train_res), y_pred




if  __name__== '__main__':
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    df = train_size_boost(train, target)
    #clf, score, mat = complexity_boost(train, target)
    print(df)