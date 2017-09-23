import pandas as pd
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from dateutil.relativedelta import relativedelta
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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed
pd.set_option('display.max_columns', None)


def get_all_data_bloodDonation():
    file = 'train_cleaned.csv'
    data = pd.read_csv(file)

    return data



def process_data_bloodDonation(data):
    numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation']

    y = data['Made Donation in March 2007']
    X = data.loc[:, numerical_features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
