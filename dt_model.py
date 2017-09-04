import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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


d = {'train': None, 'cv set': None, 'test': None}


def train_size_dt(train, target, size=0):
    print('size', size)
    print('here')
    training_features, test_features, \
    training_target, test_target, = train_test_split(train, target, test_size=0.33, random_state=778)
    X_train, X_val, y_train, y_val = train_test_split(training_features, training_target, train_size=size)
    print('start')
    start_time = time.time()
    dt = DecisionTreeClassifier(class_weight='balanced')
    dt.fit(X_train, y_train)
    print('Decision Tree took', time.time() - start_time, 'to run')
    d['train'] = f1_score(y_train, dt.predict(X_train), average='weighted')
    d['cv set'] = f1_score(y_val, dt.predict(X_val), average='weighted')
    d['test'] = f1_score(test_target, dt.predict(test_features), average='weighted')
    print('end')

    return d


def complexity_dt(X, y):
    #X_train, y_train, X_test, y_test = train_test_split(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=778)
    #smote = SMOTE(ratio=1)
    #X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
    print('Start Search')
    dt = DecisionTreeClassifier(class_weight='balanced')
    pipe = Pipeline([('dt', dt)])
    param_grid = {'dt__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=6, cv=5, scoring='neg_log_loss', verbose=5)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    print('clf', clf)
    print('best_score', grid_search.best_score_)
    y_pred = clf.predict(X_test)
    check_pred = clf.predict(X_train)
    target_names = ['Not delinq', 'Delinq']
    print(classification_report(y_test, y_pred, target_names=target_names))
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    utility.plot_confusion_matrix(conf_mat, classes=target_names,
                      title='Confusion matrix, without normalization')
    plt.show()
    return clf, clf.predict(X_train), y_pred




if  __name__== '__main__':
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    df = Parallel(n_jobs=6)(delayed(train_size_dt)(train=train, target=target, size=size) for size in np.arange(0.1, 1, 0.1))
    df = utility.merge_dict(df)
    print(df)
    clf, score, mat = complexity_dt(train, target)
