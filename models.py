"""
Construct a neural network model and support vector regression model from a data frame
"""

import pickle

import numpy as np
import pandas as pd
from lasagne import nonlinearities
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import train_test_split
from sklearn import svm

NODES = 10
NN_PICKLE = 'nn_data.pkl'
SVM_PICKLE = 'svm_data.pkl'


def build_nn(x_train, y_train, x_val, y_val):
    """
    Construct a regression neural network model from input dataframe
    
    Parameters:
        x_train : features dataframe for model training
        y_train : target dataframe for model training
        x_val : features dataframe for model validation
        y_val : target dataframe for model validation
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

    param_grid = {'hidden0_num_units': [4, 17, 25],
                  'hidden0_nonlinearity': 
                  [nonlinearities.sigmoid, nonlinearities.softmax],
                  'hidden1_num_units': [4, 17, 25],
                  'hidden1_nonlinearity': 
                  [nonlinearities.sigmoid, nonlinearities.softmax],
                  'update_learning_rate': [0.01, 0.1, 0.5]}
    grid = sklearn.grid_search.GridSearchCV(net, param_grid, verbose=0)
    grid.fit(x_train, y_train)

    y_pred = grid.predict(x_val)
    accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)

    with open(NN_PICKLE, 'wb') as files:
        pickle.dump(grid_search, files, pickle.HIGHEST_PROTOCOL)
        pickle.dump(net, files, pickle.HIGHEST_PROTOCOL)
        pickle.dump(accuracy, files, pickle.HIGHEST_PROTOCOL)


def build_svm(x_train, y_train, x_val, y_val):
    """
    Construct a regression support vector regression model from input dataframe

    Parameters:
        x_train : features dataframe for model training
        y_train : target dataframe for model training
        x_val : features dataframe for model validation
        y_val : target dataframe for model validation
    """

    clf = svm.SVR()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)

    with open(SVM_PICKLE, 'wb') as files:
        pickle.dump(accuracy, files, pickle.HIGHEST_PROTOCOL)