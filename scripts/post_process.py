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


def results():
    """
    Printing results and metrics from the pickles used with the learning
    models used in models.py
    :return: None
    """

    model_choice = int(input("Choose model for which results are to be printed:" + "\n" +
                             "1 for Linear Regression" + "\n" +
                             "2 for Neural Network" + "\n" +
                             "3 for Support Vector Machine" + "\n" +
                             "4 for Decision Tree" + "\n" +
                             "5 for Ridge Regression" + "\n" +
                             "6 for Bayesian Ridge Regression" + "\n" +
                             "7 for Lasso:" + "\n" +
                             "8 for Random Forest Regressor:" + "\n"
                             ))
    n_features = int(input("Choose number of features the model was built on:" + "\n" +
                           "Pick from 25, 50, 75, 100, 150, 200, 250, 300" + "\n"))
    if model_choice == 1:
        with open('../trained_networks/lr_%d_data.pkl' % n_features, 'rb') as result:
            clf = pickle.load(result)
            mean_abs = pickle.load(result)
            mean_sq = pickle.load(result)
            median_abs = pickle.load(result)
            r2 = pickle.load(result)
            exp_var_score = pickle.load(result)
            y_pred = pickle.load(result)

        print("Mean absolute error regression loss for linear regression model:")
        print(mean_abs)
        print("Mean squared error regression loss for linear regression model:")
        print(mean_sq)
        print("Median absolute error regression loss for linear regression model:")
        print(median_abs)
        print("R^2 (coefficient of determination) regression score function for linear regression model:")
        print(r2)
        print("Explained variance regression score function for linear regression model:")
        print(exp_var_score)

    elif model_choice == 2:
        with open('../trained_networks/nn_%d_data.pkl' % n_features, 'rb') as result:
            clf = pickle.load(result)
            net = pickle.load(result)
            mean_abs = pickle.load(result)
            mean_sq = pickle.load(result)
            median_abs = pickle.load(result)
            r2 = pickle.load(result)
            exp_var_score = pickle.load(result)
            y_pred_nn = pickle.load(result)

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

    elif model_choice == 3:
        with open('../trained_networks/svm_%d_data.pkl' % n_features, 'rb') as result:
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

    elif model_choice == 4:
        with open('../trained_networks/dt_%d_data.pkl' % n_features, 'rb') as result:
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

    elif model_choice == 5:
        with open('../trained_networks/rr_%d_data.pkl' % n_features, 'rb') as result:
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

    elif model_choice == 6:
        with open('../trained_networks/brr_%d_data.pkl' % n_features, 'rb') as result:
            clf = pickle.load(result)
            mean_abs = pickle.load(result)
            mean_sq = pickle.load(result)
            median_abs = pickle.load(result)
            r2 = pickle.load(result)
            exp_var_score = pickle.load(result)
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
        print("Best parameters for Bayesian ridge regression model:")
        print(clf.best_params_)

    elif model_choice == 7:
        with open('../trained_networks/lasso_%d_data.pkl' % n_features, 'rb') as result:
            clf = pickle.load(result)
            mean_abs = pickle.load(result)
            mean_sq = pickle.load(result)
            median_abs = pickle.load(result)
            r2 = pickle.load(result)
            exp_var_score = pickle.load(result)
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
        print("Best parameters for Lasso model:")
        print(clf.best_params_)

    elif model_choice == 8:
        return
    else:
        print("Please choose from list of available models only")

    """lw = 2
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
    plt.show()"""
