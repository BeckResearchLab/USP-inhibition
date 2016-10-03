#!/usr/bin/env python

"""
Load a neural network model from a data frame
"""

import pickle

import numpy as np
import pandas as pd
from lasagne import nonlinearities
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

NN_PICKLE = 'nn_data.pkl'
SVM_PICKLE = 'svm_data.pkl'


def results():
    with open(NN_PICKLE, 'rb') as result:
        grid = pickle.load(result)
        net = pickle.load(result)
        accuracy = pickle.load(result)

    print(grid.grid_scores_)
    print(grid.best_estimator_)
    print(grid.best_score_)
    print("Best parameters set found on development set:")
    print(grid.best_params_)
    print("Accuracy prediction score:")
    print(accuracy)
    grid.save_params_to('/tmp/grid.params')
    net.save_params_to('/tmp/net.params')

    with open(SVM_PICKLE, 'rb') as result:
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)

    print("Mean absolute error regression loss:")
    print mean_abs
    print("Mean squared error regression loss:")
    print mean_sq
    print("Median absolute error regression loss:")
    print median_abs
    print("R^2 (coefficient of determination) regression score function:")
    print r2
    print("Explained variance regression score function:")
    print exp_var_score
