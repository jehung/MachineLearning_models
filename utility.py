def complexity_dt(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=778)
    smote = SMOTE(ratio=1)
    X_train_res, y_train_res = smote.fit_sample(X_train, y_train)
    print('Start Decision Tree Search')
    clf = tree.DecisionTreeClassifier(criterion='gini', class_weight='balanced')
    pipe = Pipeline([('smote', smote), ('dt', clf)])
    param_grid = {'dt__max_depth': [2, 3, 4, 5, 6, 7, 8]}
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

