#!/usr/bin/env python

"""
Construct a neural network model and support vector regression model from the data
"""

import pickle

import numpy as np
from lasagne import nonlinearities
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet
from sklearn import svm, metrics

NODES = 10
NN_PICKLE = 'nn_data.pkl'
SVM_PICKLE = 'svm_data.pkl'


def build_nn(x_train, y_train, x_val, y_val):
    """
    Construct a regression neural network model from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_val: features dataframe for model validation
    :param y_val: target dataframe for model validation
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
    grid = sklearn.grid_search.GridSearchCV(net, param_grid, verbose=0,
                                            n_jobs=3, cv=3)
    grid.fit(x_train, y_train)

    y_pred = grid.predict(x_val)
    # Accuracy prediction score
    accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)

    with open(NN_PICKLE, 'wb') as results:
        pickle.dump(grid, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(net, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, results, pickle.HIGHEST_PROTOCOL)


def build_svm(x_train, y_train, x_val, y_val):
    """
    Construct a regression support vector regression model from input dataframe

    :param x_train: features dataframe for model training
    :param y_train: target dataframe for model training
    :param x_val: features dataframe for model validation
    :param y_val: target dataframe for model validation
    :return: None
    """

    clf = svm.SVR()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)

    # Mean absolute error regression loss
    mean_abs = metrics.mean_absolute_error(y_val, y_pred)
    # Mean squared error regression loss
    mean_sq = metrics.mean_squared_error(y_val, y_pred)
    # Median absolute error regression loss
    median_abs = metrics.median_absolute_error(y_val, y_pred)
    # R^2 (coefficient of determination) regression score function
    r2 = metrics.r2_score(y_val, y_pred)
    # Explained variance regression score function
    exp_var_score = sklearn.metrics.explained_variance_score(y_val, y_pred)

    with open(SVM_PICKLE, 'wb') as results:
        pickle.dump(mean_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_sq, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(median_abs, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(r2, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(exp_var_score, results, pickle.HIGHEST_PROTOCOL)
