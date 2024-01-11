import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

def make_ratio(numerator,denumerator):
    # If the denominator of a financial ratio is equal to zero, we set the value of the variable to zero, if the numerator is equal to zero as well. 
    # If the numerator is positive or negative, we set the ratio to infinity or negative infinity, respectively.
    # Infinity or negative infinity is later set to the 1st and 99th, respectively,
    # percentiles of the distribution of the ratio during the winsorizing.
    return numerator.div(denumerator).fillna(0)


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


