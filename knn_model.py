import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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



def fit_knn(train=None, target=None, size=0):
    #for size in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print('size', size)
    print('here')
    training_features, test_features, \
    training_target, test_target, = train_test_split(train, target, test_size=0.33, random_state=778)
    X_train, X_val, y_train, y_val = train_test_split(training_features, training_target, train_size=size)
    #smote = SMOTE(ratio=1)
    #X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
    print('start')
    start_time = time.time()
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    print('KNN took', time.time() - start_time, 'to run')
    d['train'] = f1_score(y_train, clf.predict(X_train), average='weighted')
    d['cv set'] = f1_score(y_val, clf.predict(X_val), average='weighted')
    d['test'] = f1_score(test_target, clf.predict(test_features), average='weighted')
    print('end')

    return d



def complexity_knn(X, y):
    #X_train, y_train, X_test, y_test = train_test_split(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=778)
    #smote = SMOTE(ratio=1)
    #X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
    print('Start Search')
    knn= KNeighborsClassifier()
    pipe = Pipeline([('knn', knn)])
    param_grid = {
        'knn__n_neighbors': [3,7,11]}
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=6, cv=3, scoring='neg_log_loss', verbose=5)
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
    plot_confusion_matrix(conf_mat, classes=target_names,
                      title='Confusion matrix, without normalization')
    plt.show()
    return clf, clf.predict(X_train_res), y_pred




if  __name__== '__main__':
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    df = Parallel(n_jobs=6)(delayed(fit_knn)(train=train, target=target, size=size) for size in np.arange(0.1, 1, 0.1))
    utility.merge_dict(df)
    #pool = multiprocessing.Pool(processes=6)
    #df = pool.starmap(fit_knn, zip(repeat(train, target), range(0.4,0.6,0.1)))
    print(df)
    #clf, score, mat = complexity_knn(train, target)
