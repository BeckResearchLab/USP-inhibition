#!/usr/bin/env python

"""
Tasks performed after model building
"""

import pickle

NN_PICKLE = 'nn_data.pkl'
SVM_PICKLE = 'svm_data.pkl'


def results():
    """
    Printing results and metrics from the pickles used with the learning
    models used in models.py
    :return: None
    """
    with open(NN_PICKLE, 'rb') as result:
        grid = pickle.load(result)
        net = pickle.load(result)
        accuracy = pickle.load(result)

    print("A list of named tuples of scores for each set of parameter "
          "combinations in param_grid:")
    print("[parameters, mean_validation_score over CV folds, the list of "
          "scores for each fold]")
    print(grid.grid_scores_)
    print("Estimator that was chosen by the search with the highest score:")
    print(grid.best_estimator_)
    print("Score of best_estimator on the held out data:")
    print(grid.best_score_)
    print("Parameter setting that gave the best results on the held out data:")
    print(grid.best_params_)
    print("Scorer function used on the held out data to choose the best "
          "parameters for the model:")
    print(grid.scorer_)
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
