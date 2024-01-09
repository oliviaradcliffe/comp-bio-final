"""
Created on Wed Jul 12 17:05:02 2017

@author: ZHANGXUAN

Edited by: Lizzy Klosa & Olivia Radcliffe
Date: November 27, 2023
"""

import os
import time
from datetime import datetime
import random
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

KEY_LEN = 20
np.random.seed(42)

def key_gen():
    '''
    generate random files
    '''
    keylist = [random.choice('abcdefghijklmnopqrstuvwxyz123456789') for i in range(KEY_LEN)]
    return ("".join(keylist))


def best_feature(df):
    '''
    optimal features
    '''
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import ExtraTreesClassifier
    y = np.array(df.label)
    X=np.array(df.iloc[:,1:].values)

    # Run etxra tree classifier to remove unimportant features
    forest = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs = -1)
    rfecv = RFECV(forest, step=1, cv=10, n_jobs = -1)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)
    names = list(df.columns)
    result_list = sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), names))
    best_feature_list = []
    best_feature_list_with_rank = []
    # Create a list where each element is a list of the rank and feature name
    for index, name in result_list:
        if len(best_feature_list) < rfecv.n_features_:
            best_feature_list.append(name)
            best_feature_list_with_rank.append([index, name])
    
    best_column_names = []
    # Create a list of feature names
    for rank_and_name in best_feature_list_with_rank:
        rank, name = rank_and_name[0], rank_and_name[1]
        if rank == 1:
            best_column_names.append(name)
    best_columns = df[best_column_names]

    return rfecv.n_features_, best_feature_list, best_columns

def feature_select(df):
    '''
    feature importance
    '''
    from sklearn.ensemble import ExtraTreesClassifier
    y = np.array(df.label)
    X=np.array(df.iloc[:,1:].values)
    
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=100,
                                  random_state=0, n_jobs = -1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    best_features = ['label']
    #save_feature = open("save_feature.txt",'w')
    for f in range(X.shape[1]):
        best_features.append(df.columns[indices[f]+1])

    n_best = len(best_features)
    df = df.loc[:,best_features[0:n_best]] # 0 = label
    
    return df, importances, best_features

def cross_val_roc(classifier, clf, X, y, curve = True):
    '''
    k-fold cross-validation
    '''
    from sklearn.model_selection import KFold
    from scipy import interp
    import matplotlib.pyplot as plt
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 30)
    cv = 10
    k_fold = KFold(n_splits = cv, shuffle = True)
    k_scores = []
    recall_score = []
    precision_score = []
    accuracy_score = []
    AUC_score = []
    tpr_list = []
    fpr_list = []

    for i, (train, test) in enumerate(k_fold.split(X, y)):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        score = clf.score(X[test], y[test])
        k_scores.append(score)
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        recall_score.append(metrics.recall_score(y[test],clf.predict(X[test])))
        precision_score.append(metrics.precision_score(y[test],clf.predict(X[test])))
        accuracy_score.append(metrics.accuracy_score(y[test],clf.predict(X[test])))
        AUC_score.append(roc_auc_score(y[test],clf.predict(X[test])))
        tpr_list.append(tpr.tolist())
        fpr_list.append(fpr.tolist())

    mean_tpr /= cv
    mean_tpr[-1] = 1.0
    mean_tpr = list(map(lambda x:str(x), mean_tpr))
    mean_tpr_str = "\t".join(mean_tpr)
    print("%s recall: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(recall_score).mean(), np.array(recall_score).std()))
    print("%s precision: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(precision_score).mean(), np.array(precision_score).std()))
    print("%s accuracy: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(accuracy_score).mean(), np.array(accuracy_score).std()))
    print("%s cross validation accuracy: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(k_scores).mean(), np.array(k_scores).std()))
    
    print("%s AUC: %0.2f (+/- %0.2f)" % (classifier, \
          np.array(AUC_score).mean(), np.array(AUC_score).std()))
    
    # find the best auc for plotting
    best_auc_index = np.argmax(AUC_score)

    plt.figure()
    lw = 2
    plt.plot(
        fpr_list[best_auc_index],
        tpr_list[best_auc_index],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % AUC_score[best_auc_index],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic - %s" % (classifier))
    plt.legend(loc="lower right")
    # plt.show()

        
def random_forest_classifier(classifier, train_X, train_y, curve):
    '''
    Random Forest
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    """ Grid search commented out for run time efficiency
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    """

    rf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, random_state=1, n_jobs = -1)
    rf.fit(train_X,train_y)
    cross_val_roc(classifier, rf, train_X, train_y, curve)
    return rf
    

def naive_bayes_classifier(classifier, train_X,train_y, curve):
    '''
    Naive bayes
    '''
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha = 5)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf


def knn_classifier(classifier, train_X, train_y, curve):
    '''
    K-nearest neighbors
    '''
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf,train_X, train_y, curve)
    return clf

def logistic_regression_classifier(classifier, train_X, train_y, curve):
    '''
    Logistic Regression
    '''
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=10, n_jobs = -1)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf


def svm_cross_validation(classifier, train_X, train_y, curve):
    '''
    SVM with grid search
    '''
    train_X = minmaxscaler(train_X)
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    clf = SVC(kernel = 'rbf', probability = True)
    C_range = np.logspace(-2, 10 ,13)
    gamma_range = np.logspace(-9, 3, num = 13)
    param_grid = dict(gamma = gamma_range, C = C_range)
    grid_search = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)
    grid_search.fit(train_X,train_y)
    best_parameters = grid_search.best_params_
    for para, val in best_parameters.items():
        print(para , val)
    clf = SVC(kernel='rbf', C= best_parameters['C'], gamma = best_parameters['gamma'], probability = True)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf


def xgboost_classifier(classifier, train_X, train_y, curve):
    '''
    xgboost
    '''
    import xgboost as xgb

    # needed for xgb.XGBClassifier
    train_y = np.where(train_y == -1, 0, 1)

    # Grid search function commented out for run time efficiency
    #clf = xgboost_best_params(classifier, train_X, train_y, curve)
    clf = xgb.XGBClassifier(gamma=0.03, learning_rate=0.5, max_depth=6, n_estimators=500, subsample=0.75, objective="binary:logistic", random_state=42)
    clf.fit(train_X,train_y)
    cross_val_roc(classifier, clf, train_X, train_y, curve)
    return clf

def xgboost_best_params(classifier, train_X, train_y, curve):
    """ Function to perform grid search on XGBoost for best hyperparameters
    """
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb

    param_grid = {
        'n_estimators': [50, 100, 500, 600],
        'subsample': [0.5, 0.75, 1],
        'learning_rate': [0.01, 0.5],
        'max_depth': [5, 6, 10],
        'gamma' : [0, 0.03, 0.06]
    }

    # Create a XGBClassifier
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    # Use GridSearchCV to perform the grid search
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, n_jobs=10, cv=5, verbose=1)
    grid_search.fit(train_X, train_y)

    # Print the best parameters and best score found during the grid search
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Accuracy: {:.2f}".format(grid_search.best_score_))

    return grid_search.best_estimator_

def data_normalize(X):
    '''
    data scale
    '''
    from sklearn import preprocessing
    normalized_X = preprocessing.normalize(X)
    standardized_X = preprocessing.scale(normalized_X)
    return standardized_X

def minmaxscaler(X_train):
    '''
    data scale
    '''
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train=min_max_scaler.fit_transform(X_train)
    return X_train
    

def random_negative_set(file_name, data_file):
    '''
    load positive data
    '''
    # global file_name
    df = pd.read_csv(data_file, sep = '\t', index_col = 0, header = 0)

    flag0 = 1 # use dataframe function fillna
    if flag0 != 0:
        df = df.fillna(method = 'pad')
    
    with open("negative.lncRNA.glist.xls", "r") as f:
        negative_list = f.readlines()
        negative_list = list(map(lambda x:x.strip(), negative_list))
        print("negative_list:", len(set(negative_list)))
        f.close()

    random_negative_list = list(random.sample(negative_list, 150))
    
    file_random_negative = "{}.random_negative.pickle".format(file_name)
    file_random_negative_path = os.path.join("background", file_random_negative)
    print("background gene set: {}".format(file_random_negative))
    with open(file_random_negative_path, "wb") as f:
        pickle.dump(random_negative_list, f)
        f.close()  
    

def true_labels(file_name, data_file):
    '''
    load positive data
    '''
    df = pd.read_csv(data_file, sep = '\t', index_col = 0, header = 0)

    flag0 = 1 # use dataframe function fillna
    if flag0 != 0:
        df = df.fillna(method = 'pad')
    with open("positive.lncRNA.glist.xls", "r") as f:
        positive_list = f.readlines()
        positive_list = list(map(lambda x:x.strip(), positive_list))
        print("true positive set:", len(set(positive_list)))
        f.close()
    
    if True:
        file_random_negative = "{}.random_negative.pickle".format(file_name)
        file_random_negative_path = os.path.join("background", file_random_negative)
        with open(file_random_negative_path, "rb") as f:
            negative_list = pickle.load(f)
            print("negetive set:", len(set(negative_list)))
            f.close()

    positive_df = df.loc[positive_list,:]
    negative_df = df.loc[negative_list,:]
    positive_df['label'] = 1
    negative_df['label'] = -1
               
    frames = [positive_df, negative_df]
      
    df1 = pd.concat(frames)

    cols = list(df1.columns)
    new_cols = cols[:-1]
    new_cols.insert(0,'label')
    df1 = df1.loc[:,new_cols]
    
    # feature selection
    df2, _, _ = feature_select(df1)
    y = np.array(df2.label)
    X=np.array(df2.iloc[:,1:].values)
    
    return df1, df2, X, y

def standardize_data(X):
    # standardize instance
    scaler = StandardScaler()

    # standardizing train data
    standardized_data = scaler.fit_transform(X)

    return standardized_data

    
def run_classifiers(test_classifiers, x_data):
    """ Function to run and test the given test classifiers
    """
    for classifier in test_classifiers:
        print("\n>>> *** %s ***" % (classifier))
        clf = classifiers[classifier](classifier, x_data, y, curve = True)
    end_time = datetime.now()
    print("end time:", end_time)
    print('cost: {}'.format(end_time - start_time))


if __name__ == '__main__':
    file_name = "yxv4i3btdbczetp4hyom1700683880.253802"
    data_file = 'bigtable.txt'

    test_classifiers = ['XGB', 'RF']
    
    classifiers = {'NB': naive_bayes_classifier,
               'KNN': knn_classifier,
               'LR': logistic_regression_classifier,
               'RF': random_forest_classifier,
               'SVMCV': svm_cross_validation,
               'XGB': xgboost_classifier
        }

    start_time = datetime.now()
    print("start time:", start_time)

    df1, _,  X, y = true_labels(file_name, data_file)

    print("\nRun with Best selected Parameters")
    run_classifiers(test_classifiers, X)

    # ----- run with only most important features --------
    _, _, best_columns  = best_feature(df1)

    print("\nRun with only highest ranked features")
    run_classifiers(test_classifiers, best_columns.values)
    
    
    # ----- run with standarized data -------
    standardized_data = standardize_data(X)

    print("\nRun with standarized data")
    run_classifiers(test_classifiers, standardized_data)


    # ----- run without the last 2 features -----
    selected_features = df1.drop(columns=["label", "TP53", "genebody_K562H3k4me3"])
    selected_features = df1.values

    print("\nRun with visually non-discrimitive features removed")
    run_classifiers(test_classifiers, selected_features)


    # ----- Run with PCA components = 90% variance -----
    # retain 90% of the variance 
    pca = PCA(n_components=0.9)

    # Fit the PCA model to the standardized training data
    ninetyPCA = pca.fit_transform(standardized_data)

    print("\nRun with PCA components = 90% variance")
    run_classifiers(test_classifiers, ninetyPCA)

    
