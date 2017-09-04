import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_predict, train_test_split
from joblib import Parallel, delayed
import multiprocessing
pd.set_option('display.max_columns', None)
import get_all_data
import utility
from itertools import repeat


d = {'train': [], 'cv set': [], 'test': []}


def train_size_svc(X, y):
    training_features, test_features, \
    training_target, test_target, = train_test_split(X, y, test_size=0.33, random_state=778)
    return training_features, test_features, training_target, test_target, d




def fit_svc(train=None, target=None, size=0):
    #for size in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print('size', size)
    print('here')
    training_features, test_features, training_target, test_target, d = train_size_svc(train, target)
    X_train, X_val, y_train, y_val = train_test_split(training_features, training_target, train_size=size)
    smote = SMOTE(ratio=1)
    X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
    print('start')
    clf = SVC(verbose=True)
    clf.fit(X_train_res, y_train_res)
    print('process')
    d['train'].append(f1_score(y_train_res, clf.predict(X_train_res), average='weighted'))
    d['cv set'].append(f1_score(y_val, clf.predict(X_val), average='weighted'))
    d['test'].append(f1_score(test_target, clf.predict(test_features), average='weighted'))
    print('end')

    return d


def gridSearch_nn(X, y):
    # X_train, y_train, X_test, y_test = train_test_split(X,y)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, learning_rate='invscaling',
                        verbose=True)
    parameters = {
        'hidden_layer_sizes': [(160), (160, 112, 112), (160, 112, 112, 112, 112), (160, 112, 112, 112, 112, 112, 112)]}
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    gs = GridSearchCV(estimator=mlp, param_grid=parameters, n_jobs=6, cv=sss, scoring='neg_log_loss', verbose=3)
    gs.fit(X, y)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)
    mat = clf.predict_proba(X)
    print(mat)

    return clf, gs.best_score_, mat






if  __name__== '__main__':
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    #df = Parallel(n_jobs=6)(delayed(fit_svc)(train=train, target=target, size=size) for size in [0.4, 0.9])
    #pool = multiprocessing.Pool(processes=6)
    #df = pool.starmap(fit_svc, zip(repeat(train, target), range(0.4,0.6,0.1)))
    #print(df)
    clf, score, mat = gridSearch_nn(train, target)
