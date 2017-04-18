#!/usr/bin/env python

"""
Construct a neural network model, support vector and decision trees regression models from the data
"""

import numpy as np
import pickle
import sklearn
import lasagne
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet
from sklearn.linear_model import RidgeCV, BayesianRidge, LassoCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

LR_PICKLE = '../trained_networks/lr_data.pkl'
NN_PICKLE = '../trained_networks/nn_data.pkl'
SVM_PICKLE = '../trained_networks/svm_data.pkl'
DT_PICKLE = '../trained_networks/dt_data.pkl'
RR_PICKLE = '../trained_networks/rr_data.pkl'
BRR_PICKLE = '../trained_networks/brr_data.pkl'
LASSO_PICKLE = '../trained_networks/lasso_data.pkl'


def run_models(x_train, y_train, x_test, y_test):
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
                         "7 for Lasso:" + "\n"
                         ))
    if model_choice == 1:
        build_linear(x_train, y_train, x_test, y_test)
    elif model_choice == 2:
        build_nn(x_train, y_train, x_test, y_test)
    elif model_choice == 3:
        build_svm(x_train, y_train, x_test, y_test)
    elif model_choice == 4:
        build_tree(x_train, y_train, x_test, y_test)
    elif model_choice == 5:
        build_ridge(x_train, y_train, x_test, y_test)
    elif model_choice == 6:
        build_bayesian_rr(x_train, y_train, x_test, y_test)
    elif model_choice == 7:
        build_lasso(x_train, y_train, x_test, y_test)
    else:
        print("Please choose from list of available models only")

    return


def build_linear(x_train, y_train, x_test, y_test):
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

    with open(LR_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_nn(x_train, y_train, x_test, y_test):
    """
    Constructing a regression neural network model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    # scaled_tanh = lasagne.nonlinearities.ScaledTanH(scale_in=2. / 3, scale_out=1.7159)
    reg = NeuralNet(layers=[('input', InputLayer),
                            ('hidden0', DenseLayer),
                            ('hidden1', DenseLayer),
                            ('output', DenseLayer)],
                    input_shape=(None, x_train.shape[1]),  # Number of i/p nodes = number of columns in x
                    hidden0_nonlinearity=lasagne.nonlinearities.softmax,
                    hidden1_nonlinearity=lasagne.nonlinearities.softmax,
                    output_num_units=1,  # Number of o/p nodes = number of columns in y
                    output_nonlinearity=lasagne.nonlinearities.linear,
                    max_epochs=1000,
                    regression=True,
                    verbose=1)

    param_grid = [(2, 5), (2, 5)]

    def objective(params):
        hidden0_num_units, hidden1_num_units = params
        reg.set_params(hidden0_num_units=hidden0_num_units, hidden1_num_units=hidden1_num_units)

        return -np.mean(cross_val_score(reg, x_train, y_train, cv=5, n_jobs=-1, scoring="neg_mean_absolute_error"))

    res_gp = gp_minimize(objective, param_grid, n_calls=100, random_state=0)

    print(res_gp.x[0])
    print(res_gp.x[1])
    with open(NN_PICKLE, 'wb') as results:
        pickle.dump(res_gp, results, pickle.HIGHEST_PROTOCOL)

    """# Finding the optimal set of params for each variable in the training of the neural network
    clf = sklearn.model_selection.GridSearchCV(net, param_grid, verbose=0, n_jobs=18, cv=5) # find the score used here
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

    with open(NN_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(net, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)"""

    return


def build_svm(x_train, y_train, x_test, y_test):
    """
    Constructing a support vector regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """

    clf = SVR(kernel='linear', verbose=True)  # LinearSVR
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

    with open(SVM_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_tree(x_train, y_train, x_test, y_test):
    """
    Constructing a decision trees regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = DecisionTreeRegressor()
    def objective(params):
        max_depth = params
    	clf.set_params(max_depth=max_depth)
    return -np.mean(cross_val_score(reg, x_train, y_train, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))
    space = [(1, 100)]
    res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)
    print("""Best parameters:- max_depth=%d""" % (res_gp.x[0])
    clf = DecisionTreeRegressor(max_depth=res_gp.x[0])
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

    with open(DT_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)
    print(r2)
    return


def build_ridge(x_train, y_train, x_test, y_test):
    """
    Constructing a ridge regression model from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """
    clf = RidgeCV()
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

    with open(RR_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ridge_alpha, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_bayesian_rr(x_train, y_train, x_test, y_test):
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

    with open(BRR_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ridge_alpha, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return


def build_lasso(x_train, y_train, x_test, y_test):
    """
    Constructing a Lasso linear model with cross validation from input dataframe
    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_test: features dataframe for model testing
    :param y_test: target dataframe for model testing
    :return: None
    """

    clf = LassoCV()
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
    lasso_alpha = clf.alpha_

    with open(LASSO_PICKLE, 'wb') as results:
        pickle.dump(clf, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(lasso_alpha, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_pred, results, pickle.HIGHEST_PROTOCOL)

    return
