import pandas as pd
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


def get_all_data():
    dir = 'D:\\Backups\\StemData\\'
    files = ['sample_orig_2016.txt', 'sample_orig_2015.txt', 'sample_orig_2014.txt', 'sample_orig_2013.txt',
             'sample_orig_2012.txt', 'sample_orig_2011.txt',
             'sample_orig_2010.txt', 'sample_orig_2009.txt', 'sample_orig_2008.txt', 'sample_orig_2007.txt']

    files1 = ['sample_svcg_2016.txt', 'sample_svcg_2015.txt', 'sample_svcg_2014.txt', 'sample_svcg_2013.txt',
              'sample_svcg_2012.txt', 'sample_svcg_2011.txt',
              'sample_svcg_2010.txt', 'sample_svcg_2009.txt', 'sample_svcg_2008.txt', 'sample_svcg_2008.txt']

    merged_data = pd.DataFrame()
    for i in range(1,2):
        print(files[i])
        raw = pd.read_csv(dir + files[i], sep='|', header=None, low_memory=False)
        raw.columns = ['credit_score', 'first_pmt_date', 'first_time', 'mat_date', 'msa', 'mi_perc', 'units',
                       'occ_status', 'ocltv', 'odti', 'oupb', 'oltv', 'oint_rate', 'channel', 'ppm', 'fixed_rate',
                       'state', 'prop_type', 'zip', 'loan_num', 'loan_purpose', 'oterm', 'num_borrowers', 'seller_name',
                       'servicer_name', 'exceed_conform']

        raw1 = pd.read_csv(dir + files1[i], sep='|', header=None, low_memory=False)
        raw1.columns = ['loan_num', 'yearmon', 'curr_upb', 'curr_delinq', 'loan_age', 'remain_months', 'repurchased',
                        'modified', 'zero_bal', 'zero_date', 'curr_rate', 'curr_def_upb', 'ddlpi', 'mi_rec',
                        'net_proceeds',
                        'non_mi_rec', 'exp', 'legal_costs', 'maint_exp', 'tax_insur', 'misc_exp', 'loss', 'mod_exp']

        data = pd.merge(raw, raw1, on='loan_num', how='inner')

        merged_data = merged_data.append(data)

    data.drop(['seller_name', 'servicer_name', 'first_pmt_date', 'mat_date', 'msa'], axis=1, inplace=True)

    # all data must have the following: credit_score, ocltv, odti, oltv, oint_rate, curr_upb
    # remove any datapoints with missing values from the above features
    data.dropna(subset=['credit_score', 'odti', 'oltv', 'oint_rate', 'curr_upb'], how='any', inplace=True)
    data.credit_score = pd.to_numeric(data['credit_score'], errors='coerce')
    data.yearmon = pd.to_datetime(data['yearmon'], format='%Y%m')
    data.fillna(value=0, inplace=True, axis=1)

    return merged_data


def process_data_rev(data):
    #data.sort_values(['loan_num'], ascending=True).groupby(['yearmon'], as_index=False)  ##consider move this out
    #data.set_index(['loan_num', 'yearmon'], inplace=True) ## consider move this out
    y = data['curr_delinq']
    #data['prev_delinq'] = data.curr_delinq.shift(1) ## needs attention here
    #data['prev_delinq'] = data.groupby(level=0)['curr_delinq'].shift(1)
    #print(sum(data.prev_delinq.isnull()))
    data.fillna(value=0, inplace=True, axis=1)
    data.drop(['curr_delinq'], axis=1, inplace=True)
    print(y.shape)
    ## how many classes are y?
    ## remove y from X
    X = pd.get_dummies(data, columns=['first_time', 'occ_status', 'channel', 'ppm', 'fixed_rate',
                                  'state', 'prop_type', 'loan_purpose', 'exceed_conform', 'repurchased'])
    #y = label_binarize(y, classes=[0, 1, 2, 3]) ## do we really have to do this?
    X[['credit_score','mi_perc','units','ocltv', 'odti', 'oupb', 'oltv', 'oint_rate','zip',
       'curr_upb','loan_age','remain_months', 'curr_rate','curr_def_upb', 'ddlpi','mi_rec',
       'non_mi_rec', 'exp', 'legal_costs','maint_exp','tax_insur', 'misc_exp', 'loss','mod_exp']] = \
    scale(X[['credit_score','mi_perc','units','ocltv', 'odti', 'oupb', 'oltv', 'oint_rate','zip',
       'curr_upb','loan_age','remain_months', 'curr_rate','curr_def_upb', 'ddlpi','mi_rec',
       'non_mi_rec', 'exp', 'legal_costs','maint_exp','tax_insur', 'misc_exp', 'loss','mod_exp']], with_mean=False)
    return X, y



#all_data = get_all_data()
#train, target = process_data(alldata)
#print(all_data.shape)
#all_data.to_csv('data_12_16.csv')