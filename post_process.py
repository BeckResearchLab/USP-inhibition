#!/usr/bin/env python

"""
Tasks performed after model building: Metric analysis and display
"""

import pickle

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

NN_PICKLE = 'nn_data.pkl'
SVM_PICKLE = 'svm_data.pkl'
DT_PICKLE = 'dt_data.pkl'
RR_PICKLE = 'rr_data.pkl'


def results():
    """
    Printing results and metrics from the pickles used with the learning
    models used in models.py
    :return: None
    """
    with open(NN_PICKLE, 'rb') as result:
        grid = pickle.load(result)
        net = pickle.load(result)
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        accuracy = pickle.load(result)

    grid.save_params_to('/tmp/grid.params')
    net.save_params_to('/tmp/net.params')

    print("A list of named tuples of scores for each set of parameter "
          "combinations in param_grid for the NN model:")
    print("[parameters, mean_validation_score over CV folds, the list of "
          "scores for each fold]")
    print(grid.grid_scores_)
    print("Estimator that was chosen by the search with the highest score for the NN model:")
    print(grid.best_estimator_)
    print("Score of best_estimator on the held out data for the NN model:")
    print(grid.best_score_)
    print("Parameter setting that gave the best results on the held out data for the NN model:")
    print(grid.best_params_)
    print("Scorer function used on the held out data to choose the best "
          "parameters for the NN model:")
    print(grid.scorer_)
    print("Mean absolute error regression loss for NN model:")
    print mean_abs
    print("Mean squared error regression loss for NN model:")
    print mean_sq
    print("Median absolute error regression loss for NN model:")
    print median_abs
    print("R^2 (coefficient of determination) regression score function for NN model:")
    print r2
    print("Explained variance regression score function for NN model:")
    print exp_var_score
    print("Accuracy prediction score for the NN model:")
    print accuracy

    with open(SVM_PICKLE, 'rb') as result:
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        accuracy = pickle.load(result)

    print("Mean absolute error regression loss for SVM model:")
    print mean_abs
    print("Mean squared error regression loss for SVM model:")
    print mean_sq
    print("Median absolute error regression loss for SVM model:")
    print median_abs
    print("R^2 (coefficient of determination) regression score function for SVM model:")
    print r2
    print("Explained variance regression score function for SVM model:")
    print exp_var_score
    print("Accuracy prediction score for the SVM model:")
    print accuracy

    with open(DT_PICKLE, 'rb') as result:
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        accuracy = pickle.load(result)

    print("Mean absolute error regression loss for tree model:")
    print mean_abs
    print("Mean squared error regression loss for tree model:")
    print mean_sq
    print("Median absolute error regression loss for tree model:")
    print median_abs
    print("R^2 (coefficient of determination) regression score function for tree model:")
    print r2
    print("Explained variance regression score function for tree model:")
    print exp_var_score
    print("Accuracy prediction score for the tree model:")
    print accuracy

    with open(RR_PICKLE, 'rb') as result:
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        accuracy = pickle.load(result)
        ridge_alpha = pickle.load(result)

    print("Mean absolute error regression loss for ridge regression model:")
    print mean_abs
    print("Mean squared error regression loss for ridge regression model:")
    print mean_sq
    print("Median absolute error regression loss for ridge regression model:")
    print median_abs
    print("R^2 (coefficient of determination) regression score function for ridge regression model:")
    print r2
    print("Explained variance regression score function for ridge regression model:")
    print exp_var_score
    print("Accuracy prediction score for the ridge regression model:")
    print accuracy
    print("Cross-validated value of the alpha parameter for ridge regression model:")
    print ridge_alpha

