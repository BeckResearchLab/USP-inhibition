#!/usr/bin/env python

"""
Tasks performed after model building: Metric analysis and display
"""

import matplotlib.pyplot as plt
import pickle

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

NN_PICKLE = 'data/trained_networks/nn_data.pkl'
SVM_PICKLE = 'data/trained_networks/svm_data.pkl'
DT_PICKLE = 'data/trained_networks/dt_data.pkl'
RR_PICKLE = 'data/trained_networks/rr_data.pkl'
BRR_PICKLE = 'data/trained_networks/brr_data.pkl'
LASSO_PICKLE = 'data/trained_networks/lasso_data.pkl'
XY_PICKLE = 'data/xy_data.pkl'


def results():
    """
    Printing results and metrics from the pickles used with the learning
    models used in models.py
    :return: None
    """
    with open(NN_PICKLE, 'rb') as result:
        clf = pickle.load(result)
        net = pickle.load(result)
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        y_pred_nn = pickle.load(result)

    clf.save_params_to('/tmp/clf.params')
    net.save_params_to('/tmp/net.params')

    print("A list of named tuples of scores for each set of parameter "
          "combinations in param_grid for the NN model:")
    print("[parameters, mean_validation_score over CV folds, the list of "
          "scores for each fold]")
    print(clf.grid_scores_)
    print("Estimator that was chosen by the search with the highest score for the NN model:")
    print(clf.best_estimator_)
    print("Score of best_estimator on the held out data for the NN model:")
    print(clf.best_score_)
    print("Parameter setting that gave the best results on the held out data for the NN model:")
    print(clf.best_params_)
    print("Scorer function used on the held out data to choose the best "
          "parameters for the NN model:")
    print(clf.scorer_)
    print("Mean absolute error regression loss for NN model:")
    print(mean_abs)
    print("Mean squared error regression loss for NN model:")
    print(mean_sq)
    print("Median absolute error regression loss for NN model:")
    print(median_abs)
    print("R^2 (coefficient of determination) regression score function for NN model:")
    print(r2)
    print("Explained variance regression score function for NN model:")
    print(exp_var_score)

    with open(SVM_PICKLE, 'rb') as result:
        clf = pickle.load(result)
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        y_pred_svm = pickle.load(result)

    print("Mean absolute error regression loss for SVM model:")
    print(mean_abs)
    print("Mean squared error regression loss for SVM model:")
    print(mean_sq)
    print("Median absolute error regression loss for SVM model:")
    print(median_abs)
    print("R^2 (coefficient of determination) regression score function for SVM model:")
    print(r2)
    print("Explained variance regression score function for SVM model:")
    print(exp_var_score)

    with open(DT_PICKLE, 'rb') as result:
        clf = pickle.load(result)
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        y_pred_dt = pickle.load(result)

    print("Mean absolute error regression loss for tree model:")
    print(mean_abs)
    print("Mean squared error regression loss for tree model:")
    print(mean_sq)
    print("Median absolute error regression loss for tree model:")
    print(median_abs)
    print("R^2 (coefficient of determination) regression score function for tree model:")
    print(r2)
    print("Explained variance regression score function for tree model:")
    print(exp_var_score)

    with open(RR_PICKLE, 'rb') as result:
        clf = pickle.load(result)
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        ridge_alpha = pickle.load(result)
        y_pred_rr = pickle.load(result)

    print("Mean absolute error regression loss for ridge regression model:")
    print(mean_abs)
    print("Mean squared error regression loss for ridge regression model:")
    print(mean_sq)
    print("Median absolute error regression loss for ridge regression model:")
    print(median_abs)
    print("R^2 (coefficient of determination) regression score function for ridge regression model:")
    print(r2)
    print("Explained variance regression score function for ridge regression model:")
    print(exp_var_score)
    print("Cross-validated value of the alpha parameter for ridge regression model:")
    print(ridge_alpha)

    with open(BRR_PICKLE, 'rb') as result:
        clf = pickle.load(result)
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        ridge_alpha = pickle.load(result)
        y_pred_brr = pickle.load(result)

    print("Mean absolute error regression loss for Bayesian ridge regression model:")
    print(mean_abs)
    print("Mean squared error regression loss for Bayesian ridge regression model:")
    print(mean_sq)
    print("Median absolute error regression loss for Bayesian ridge regression model:")
    print(median_abs)
    print("R^2 (coefficient of determination) regression score function for Bayesian ridge regression model:")
    print(r2)
    print("Explained variance regression score function for Bayesian ridge regression model:")
    print(exp_var_score)
    print("Cross-validated value of the alpha parameter for Bayesian ridge regression model:")
    print(ridge_alpha)

    with open(LASSO_PICKLE, 'rb') as result:
        clf = pickle.load(result)
        mean_abs = pickle.load(result)
        mean_sq = pickle.load(result)
        median_abs = pickle.load(result)
        r2 = pickle.load(result)
        exp_var_score = pickle.load(result)
        lasso_alpha = pickle.load(result)
        y_pred_lasso = pickle.load(result)

    print("Mean absolute error regression loss for Lasso model:")
    print(mean_abs)
    print("Mean squared error regression loss for Lasso model:")
    print(mean_sq)
    print("Median absolute error regression loss for Lasso model:")
    print(median_abs)
    print("R^2 (coefficient of determination) regression score function for Lasso model:")
    print(r2)
    print("Explained variance regression score function for Lasso model:")
    print(exp_var_score)
    print("Cross-validated value of the alpha parameter for Lasso model:")
    print(lasso_alpha)

    with open(XY_PICKLE, 'rb') as result:
        x_train = pickle.load(result)
        x_test = pickle.load(result)
        y_train = pickle.load(result)
        y_test = pickle.load(result)

    lw = 2
    plt.plot(y_test, y_pred_nn, lw=lw, label='ANN', marker='ro')
    plt.plot(y_test, y_pred_svm, lw=lw, label='SVM', marker='ro')
    plt.plot(y_test, y_pred_dt, lw=lw, label='Decision Trees', marker='ro')
    plt.plot(y_test, y_pred_rr, lw=lw, label='Ridge Regression', marker='ro')
    plt.plot(y_test, y_pred_brr, lw=lw, label='Bayesian Ridge Regression', marker='ro')
    plt.plot(y_test, y_pred_lasso, lw=lw, label='Lasso', marker='ro')
    plt.plot(y_test, y_test)
    plt.title("Comparison of predictions vs true values from different models", marker='ro')
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.style.use('ggplot')
    plt.show()
