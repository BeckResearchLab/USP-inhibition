#!/usr/bin/env python

"""
Construct a neural network model, support vector and decision trees regression models from the data
"""

import pickle

import lasagne
import numpy as np
import sklearn
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, BayesianRidge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"


def run_models(x_train, y_train, x_test, y_test, n_features):
    """
    Driving all machine learning models as parallel processes.
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    model_choice = int(input("Type your choice of model to be run:" + "\n" +
                             "1 for Linear Regression" + "\n" +
                             "2 for Neural Network" + "\n" +
                             "3 for Support Vector Machine" + "\n" +
                             "4 for Decision Tree" + "\n" +
                             "5 for Ridge Regression" + "\n" +
                             "6 for Bayesian Ridge Regression" + "\n" +
                             "7 for Lasso:" + "\n" +
                             "8 for Random Forest Regressor:" + "\n"
                             ))
    if model_choice == 1:
        build_linear(x_train, y_train, x_test, y_test, n_features)
    elif model_choice == 2:
        build_nn(x_train, y_train, x_test, y_test, n_features)
    elif model_choice == 3:
        build_svm(x_train, y_train, x_test, y_test, n_features)
    elif model_choice == 4:
        build_tree(x_train, y_train, x_test, y_test, n_features)
    elif model_choice == 5:
        build_ridge(x_train, y_train, x_test, y_test, n_features)
    elif model_choice == 6:
        build_bayesian_rr(x_train, y_train, x_test, y_test, n_features)
    elif model_choice == 7:
        build_lasso(x_train, y_train, x_test, y_test, n_features)
    elif model_choice == 8:
        build_forest(x_train, y_train, x_test, y_test, n_features)
    else:
        print("Please choose from list of available models only")

    return


def build_linear(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a decision trees regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = LinearRegression(n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)

    with open('../trained_networks/lr_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_nn(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a regression neural network model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    net = NeuralNet(layers=[('input', InputLayer),
                            ('hidden0', DenseLayer),
                            ('hidden1', DenseLayer),
                            ('output', DenseLayer)],
                    input_shape=(None, x_train.shape[1]),  # Number of i/p nodes = number of columns in x
                    hidden0_num_units=15,
                    hidden0_nonlinearity=lasagne.nonlinearities.softmax,
                    hidden1_num_units=17,
                    hidden1_nonlinearity=lasagne.nonlinearities.softmax,
                    output_num_units=1,  # Number of o/p nodes = number of columns in y
                    output_nonlinearity=lasagne.nonlinearities.softmax,
                    max_epochs=100,
                    update_learning_rate=0.01,
                    regression=True,
                    verbose=0)

    # Finding the optimal set of params for each variable in the training of the neural network
    param_dist = {'hidden0_num_units':sp_randint(3, 30), 'hidden1_num_units':sp_randint(3, 30)}
    clf = RandomizedSearchCV(estimator=net, param_distributions=param_dist,
                             n_iter=15, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)

    with open('../trained_networks/nn_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(net, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_svm(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a support vector regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """

    clf = LinearSVR(random_state=1, dual=False, epsilon=0,
                    loss='squared_epsilon_insensitive')
    # Random state has int value for non-random sampling
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)

    with open('../trained_networks/svm_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_tree(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a decision trees regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    model = DecisionTreeRegressor()
    param_dist = {'max_depth': sp_randint(1, 15),
                  'min_samples_split': sp_randint(2, 15)}
    clf = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                             n_iter=15, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(clf.best_params_, clf.best_score_)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)

    with open('../trained_networks/dt_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_ridge(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a ridge regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = Ridge()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)
    # Optimal ridge regression alpha value from CV
    ridge_alpha = clf.alpha_

    with open('../trained_networks/rr_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_bayesian_rr(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a Bayesian ridge regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = BayesianRidge()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)
    # Optimal ridge regression alpha value from CV
    ridge_alpha = clf.alpha_

    with open('../trained_networks/brr_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_lasso(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a Lasso linear model with cross validation from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """

    model = Lasso(random_state=1)
    # Random state has int value for non-random sampling
    param_dist = {'alpha': np.arange( 0.0001, 1, 0.001 ).tolist()}
    clf = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                             n_iter=15, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(clf.best_params_, clf.best_score_)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)

    with open('../trained_networks/lasso_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_forest(x_train, y_train, x_test, y_test, n_features):
    """
    Constructing a random forest regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    model = RandomForestRegressor()
    param_dist = {'max_depth': sp_randint(1, 15),
                  'min_samples_split': sp_randint(2, 15)}
    clf = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                             n_iter=15, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Mean absolute error regression loss
    mean_abs = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    # Mean squared error regression loss
    mean_sq = sklearn.metrics.mean_squared_error(y_test, y_pred)
    # Median absolute error regression loss
    median_abs = sklearn.metrics.median_absolute_error(y_test, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_test, y_pred)

    with open('../trained_networks/rfr_%d_data.pkl' % n_features, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)
    print(r2)
    return
