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
    with open(NN_PICKLE, 'rb') as file:
        grid_search = pickle.load(file)
        net = pickle.load(file)

    print(grid_search.grid_scores_)
    print(grid_search.best_estimator_)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    grid_search.save_params_to('/tmp/grid_search.params')
    net.save_params_to('/tmp/net.params')

    with open(SVM_PICKLE, 'rb') as file:
        mean_abs = pickle.load(file)
        mean_sq = pickle.load(file)
        median_abs = pickle.load(file)
        r2 = pickle.load(file)

    print mean_abs, mean_sq, median_abs, r2
