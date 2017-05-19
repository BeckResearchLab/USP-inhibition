#!/usr/bin/env python

"""
Collection of scripts for project plots
"""

import matplotlib.pyplot as plt
import matplotlib
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
