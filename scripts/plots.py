#!/usr/bin/env python

"""
Collection of scripts for project plots
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('Agg')
plt.style.use('seaborn-pastel')


def plot_features(x_train, y_train, x_test, y_test):
    """
    Plotting each feature x and its corresponding value of target function y.
    :param x_train: Dataframe containing feature space of training set.
    :param y_train: Dataframe containing target/output of training set.
    :param x_test: Dataframe containing feature space of testing set.
    :param y_test: Dataframe containing target/output of testing set.
    """
    for column in x_train:
        plt.scatter(x_train[column], y_train, label='Training set', color='blue')
        plt.scatter(x_test[column], y_test, label='Test set', color='red')
        plt.axhspan(0, 40, facecolor='orange', alpha=0.3)
        plt.title('%s effect on inhibition activity score trend' % x_train[column].name)
        plt.xlabel('%s' % x_train[column].name)
        plt.ylabel('Activity score')
        plt.savefig('../plots/feature_plots/%s.png' % x_train[column].name, bbox_inches='tight')

    return


def plot_y_dist(y_train, y_test):

    """
    Plotting the counts of inhibition scores
    :param y_train: Target values used for training
    :param y_test: Target values used for testing
    :return: None
    """

    y_train = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                          'data/y_train_postprocessing.csv')
    y_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                         'data/y_test_postprocessing.csv')
    y_train.columns = ['ID', 'Activity_Score']
    df_train = y_train.groupby('Activity_Score')['ID'].nunique().to_frame()
    df_train['score'] = df_train.index
    df_train.columns = ['score_counts', 'score']
    df_train = df_train.reset_index(drop=True)

    y_test.columns = ['ID', 'Activity_Score']
    df_test = y_test.groupby('Activity_Score')['ID'].nunique().to_frame()
    df_test['score'] = df_test.index
    df_test.columns = ['score_counts', 'score']
    df_test = df_test.reset_index(drop=True)
    plt.plot(df_train['score'], df_train['score_counts'])
    plt.plot(df_test['score'], df_test['score_counts'])
    plt.title('Sample counts of unique inhibition scores')
    plt.xlabel('Inhibition score')
    plt.ylabel('Number of molecule samples')
    plt.axvspan(0, 40, facecolor='orange', alpha=0.2)
    plt.axvspan(40, 100, facecolor='violet', alpha=0.2)
    plt.savefig('./plots/score_counts.png', bbox_inches='tight')


def plot_results():
    """
    Plots the performance metrics, comparing the different methods over the number of features used.
    :return: 
    """
    n_features = [25, 50, 75, 100, 150, 200, 250, 300]
    mean_abs, mean_sq, median_abs, r2, exp_var_score = ({} for i in range(5))
    for n in n_features:
        with open('../trained_networks/lr_%d_data.pkl' % n, 'rb') as result:
            lr_clf = pickle.load(result)
            mean_abs["lr{0}".format(n)] = pickle.load(result)
            mean_sq["lr{0}".format(n)] = pickle.load(result)
            median_abs["lr{0}".format(n)] = pickle.load(result)
            r2["lr{0}".format(n)] = pickle.load(result)
            exp_var_score["lr{0}".format(n)] = pickle.load(result)
            lr_y_pred = pickle.load(result)

        with open('../trained_networks/nn_%d_data.pkl' % n, 'rb') as result:
            nn_clf = pickle.load(result)
            nn_net = pickle.load(result)
            mean_abs["nn{0}".format(n)] = pickle.load(result)
            mean_sq["nn{0}".format(n)] = pickle.load(result)
            median_abs["nn{0}".format(n)] = pickle.load(result)
            r2["nn{0}".format(n)] = pickle.load(result)
            exp_var_score["nn{0}".format(n)] = pickle.load(result)
            nn_y_pred = pickle.load(result)

        with open('../trained_networks/dt_%d_data.pkl' % n, 'rb') as result:
            dt_clf = pickle.load(result)
            mean_abs["dt{0}".format(n)] = pickle.load(result)
            mean_sq["dt{0}".format(n)] = pickle.load(result)
            median_abs["dt{0}".format(n)] = pickle.load(result)
            r2["dt{0}".format(n)] = pickle.load(result)
            exp_var_score["dt{0}".format(n)] = pickle.load(result)
            dt_y_pred = pickle.load(result)

        with open('../trained_networks/rr_%d_data.pkl' % n, 'rb') as result:
            rr_clf = pickle.load(result)
            mean_abs["rr{0}".format(n)] = pickle.load(result)
            mean_sq["rr{0}".format(n)] = pickle.load(result)
            median_abs["rr{0}".format(n)] = pickle.load(result)
            r2["rr{0}".format(n)] = pickle.load(result)
            exp_var_score["rr{0}".format(n)] = pickle.load(result)
            rr_ridge_alpha = pickle.load(result)
            rr_y_pred = pickle.load(result)

        with open('../trained_networks/brr_%d_data.pkl' % n, 'rb') as result:
            brr_clf = pickle.load(result)
            mean_abs["brr{0}".format(n)] = pickle.load(result)
            mean_sq["brr{0}".format(n)] = pickle.load(result)
            median_abs["brr{0}".format(n)] = pickle.load(result)
            r2["brr{0}".format(n)] = pickle.load(result)
            exp_var_score["brr{0}".format(n)] = pickle.load(result)
            brr_y_pred = pickle.load(result)

        with open('../trained_networks/lasso_%d_data.pkl' % n, 'rb') as result:
            lasso_clf = pickle.load(result)
            mean_abs["lasso{0}".format(n)] = pickle.load(result)
            mean_sq["lasso{0}".format(n)] = pickle.load(result)
            median_abs["lasso{0}".format(n)] = pickle.load(result)
            r2["lasso{0}".format(n)] = pickle.load(result)
            exp_var_score["lasso{0}".format(n)] = pickle.load(result)
            lasso_y_pred = pickle.load(result)

    plt.plot()

def pred_actual(y_actual, y_pred):
    """
    Plots the actual and predicted values of inhibition from the different models.
    :param y_actual: Dataframe containing target/output of training set.
    :param y_pred: Dataframe containing predicted target/output values.
    :return: None
    """
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
