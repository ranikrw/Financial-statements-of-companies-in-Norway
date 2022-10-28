import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

def make_ratio(numerator,denumerator):
    # If the denominator of a financial ratio is equal to zero, we set the value of the variable to zero, if the numerator is equal to zero as well. 
    # If the numerator is positive or negative, we set the ratio to infinity or negative infinity, respectively.
    # Infinity or negative infinity is later set to the 1st and 99th, respectively,
    # percentiles of the distribution of the ratio during the winsorizing.
    ratio = [None]*len(numerator)
    for i in range(len(numerator)):
        if denumerator[i]!=0:
            ratio[i] = numerator[i] / denumerator[i]
        else: #denumerator[i]==0:
            if numerator[i]>0:
                ratio[i] = np.inf
            elif numerator[i]<0:
                ratio[i] = -np.inf
            else: # numerator[i]==0:
                ratio[i] = 0
    return pd.Series(ratio)


def prepare_training_and_test_data(data_train,data_test,input_variables,response_variable):
    # Dividing between X (independent variables) and y (dependent variable)
    
    if np.sum(np.sum(pd.isnull(data_train[input_variables])))!=0:
        print('ERROR: missing values in training data')
    if np.sum(np.sum(pd.isnull(data_train[response_variable])))!=0:
        print('ERROR: missing values in training data')

    if np.sum(np.sum(pd.isnull(data_test[input_variables])))!=0:
        print('ERROR: missing values in training data')
    if np.sum(np.sum(pd.isnull(data_test[response_variable])))!=0:
        print('ERROR: missing values in training data')

    # Making training data (into ndarray)
    X_train = data_train[input_variables]
    y_train = data_train[response_variable]
    X_train = X_train.astype(float)
    y_train = y_train.astype(int)
        
    # Making test data (into ndarray)
    X_test = data_test[input_variables]
    y_test = data_test[response_variable]
    X_test = X_train.astype(float)
    y_test = y_train.astype(int)

    return X_train, y_train, X_test, y_test


