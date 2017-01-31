#!/usr/bin/env python

"""
Construct a neural network model, support vector and decision trees regression models from the data
"""

import pickle
from multiprocessing import Process

import numpy as np
from lasagne import nonlinearities
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet
import sklearn

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

NODES = 10
NN_PICKLE = 'nn_data.pkl'
SVM_PICKLE = 'svm_data.pkl'
DT_PICKLE = 'dt_data.pkl'
RR_PICKLE = 'rr_data.pkl'
BRR_PICKLE = 'brr_data.pkl'
LASSO_PICKLE = 'lasso_data.pkl'


def run_models(x_train, y_train, x_test, y_test):
    """
    Function to drive all models in parallel.

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    p1 = Process(target=build_nn, args=(x_train, y_train, x_test, y_test))
    p1.start()
    p2 = Process(target=build_svm, args=(x_train, y_train, x_test, y_test))
    p2.start()
    p3 = Process(target=build_tree, args=(x_train, y_train, x_test, y_test))
    p3.start()
    p4 = Process(target=build_ridge, args=(x_train, y_train, x_test, y_test))
    p4.start()
    p5 = Process(target=build_bayesian_rr, args=(x_train, y_train, x_test, y_test))
    p5.start()
    p6 = Process(target=build_lasso, args=(x_train, y_train, x_test, y_test))
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()

    return


def build_nn(x_train, y_train, x_test, y_test):
    """
    Construct a regression neural network model from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """

    # Create classification model
    net = NeuralNet(layers=[('input', InputLayer),
                            ('hidden0', DenseLayer),
                            ('hidden1', DenseLayer),
                            ('output', DenseLayer)],
                    input_shape=(None, x_train.shape[1]),
                    hidden0_num_units=NODES,
                    hidden0_nonlinearity=nonlinearities.softmax,
                    hidden1_num_units=NODES,
                    hidden1_nonlinearity=nonlinearities.softmax,
                    output_num_units=len(np.unique(y_train)),
                    output_nonlinearity=nonlinearities.softmax,
                    update_learning_rate=0.1,
                    verbose=1,
                    max_epochs=100)

    param_grid = {'hidden0_num_units': [1, 4, 17, 25],
                  'hidden0_nonlinearity': 
                  [nonlinearities.sigmoid, nonlinearities.softmax],
                  'hidden1_num_units': [1, 4, 17, 25],
                  'hidden1_nonlinearity': 
                  [nonlinearities.sigmoid, nonlinearities.softmax],
                  'update_learning_rate': [0.01, 0.1, 0.5]}
    clf = sklearn.grid_search.GridSearchCV(net, param_grid, verbose=0, n_jobs=3, cv=3)
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
    # Accuracy prediction score
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    with open(NN_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(net, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_svm(x_train, y_train, x_test, y_test):
    """
    Construct a support vector regression model from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """

    clf = sklearn.svm.SVR()
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
    # Accuracy prediction score
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    with open(SVM_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_tree(x_train, y_train, x_test, y_test):
    """
    Construct a decision trees regression model from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = sklearn.tree.DecisionTreeRegressor()
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
    # Accuracy prediction score
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

    with open(DT_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_ridge(x_train, y_train, x_test, y_test):
    """
    Construct a ridge regression model from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = sklearn.linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
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
    # Accuracy prediction score
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    # Optimal ridge regression alpha value from CV
    ridge_alpha = clf.alpha_

    with open(RR_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ridge_alpha, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_bayesian_rr(x_train, y_train, x_test, y_test):
    """
    Construct a Bayesian ridge regression model from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = sklearn.linear_model.BayesianRidge()
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
    # Accuracy prediction score
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    # Optimal ridge regression alpha value from CV
    ridge_alpha = clf.alpha_

    with open(BRR_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ridge_alpha, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_lasso(x_train, y_train, x_test, y_test):
    """
    Construct a Lasso linear model with cross validation from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """

    clf = sklearn.linear_model.LassoCV()
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
    # Accuracy prediction score
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    # Optimal ridge regression alpha value from CV
    lasso_alpha = clf.alpha_

    with open(LASSO_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(lasso_alpha, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return
