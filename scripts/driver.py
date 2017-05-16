#!/usr/bin/env python

"""
Primary execution file for USP-Inhibition project
"""
import sys
sys.path.append("/home/pphilip/Tools/openbabel-install/lib")

import descriptors
import genalgo
import models
import numpy as np
import pandas as pd
import pickle
import post_process
import sklearn
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import utils

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

TARGET_COLUMN = 'Activity_Score'
XY_PICKLE = '../data/xy_data.pkl'


def main():
    """
    Module to execute the entire package from data retrieval to model results
    :return: None
    """

    choice = int(input("Type:" + "\n" +
                       "1 to prepare raw assay data and calculate molecular descriptors" + "\n" +
                       "2 to preprocess descriptor data" + "\n" +
                       "3 to create ML models" + "\n" +
                       "4 to display model results" + "\n" +
                       "5 to plot the data set using limited features" + "\n" +
                       "6 to run a genetic algorithm" + "\n"
                       ))
    if choice == 1:

        # Importing inhibitor notation data
        response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                                   'pphilip-usp-inhibition/data/compounds_smiles.txt')
        df_smiles = utils.create_notation_dataframe(response)

        # Importing inhibitor activity data
        response = pd.read_csv('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/data/AID_743255_datatable.csv')
        df_activity = utils.create_activity_dataframe(response)

        df = df_activity.merge(df_smiles)
        df.drop(df.index[[276743, 354142]], inplace=True)
        df.sort_values(by='CID', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Drop non-descriptor columns before feature space reduction
        df_x = df.drop(['Activity_Score', 'CID'], axis=1)

        # Creating target column
        df_y = df.drop(['SMILES', 'CID'], axis=1)
        # Extracting molecular descriptors for all compounds
        print("Starting descriptor calculation")
        descriptors.extract_all_descriptors(df_x, 'SMILES')
        # Joining separate descriptor data frames and target column
        df = utils.join_dataframes()
        df = df.join(df_y)
        df.to_csv('../data/df_preprocessing.csv')

    elif choice == 2:

        df = pd.read_csv('https://s3-us-west-2.amazonaws.com/'
                         'pphilip-usp-inhibition/data/df_preprocessing.csv')
        df.drop(df.columns[0], axis=1, inplace=True)

        # Copying column names to use after np array manipulation
        all_headers = list(df.columns.values)
        x_headers = list(df.columns.values)[:-1]

        # Train, validation and test split
        df_train, df_test = sklearn.cross_validation.train_test_split(df, test_size=0.25)
        # Reassign column name and index after randomized split
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_train = pd.DataFrame(df_train, columns=all_headers)
        df_test = pd.DataFrame(df_test, columns=all_headers)

        # Remove the classification column from the dataframe
        x_train = df_train.drop(TARGET_COLUMN, axis=1)
        x_test = df_test.drop(TARGET_COLUMN, axis=1)
        y_train = df_train[TARGET_COLUMN]
        y_test = df_test[TARGET_COLUMN]

        # Checking dataframe for NaN and infinite values
        x_train = utils.change_nan_infinite(x_train)
        y_train = utils.change_nan_infinite(y_train)
        x_test = utils.change_nan_infinite(x_test)
        y_test = utils.change_nan_infinite(y_test)

        y_train = pd.DataFrame(y_train, columns=['Activity_Score'])
        y_test = pd.DataFrame(y_test, columns=['Activity_Score'])
        y_train.to_csv('../data/y_train_postprocessing.csv')
        y_test.to_csv('../data/y_test_postprocessing.csv')

        # Transform all column values to mean 0 and unit variance
        clf = sklearn.preprocessing.StandardScaler().fit(x_train)
        x_train = clf.transform(x_train)
        x_test = clf.transform(x_test)
        y_train = np.array(y_train)
        # Feature selection and feature importance plot
        utils.choose_features(x_train, y_train, x_test, x_headers)

    elif choice == 3:
        n_features = int(input("Choose the number of features to be used in the model" + "\n" +
                               "Pick from 50, 100, 150, 200" + "\n"))
        feature_selector = str(input("The feature space can be reduced using" + "\n" +
                                     "Random Forest regressor or Randomized Lasso" + "\n" +
                                     "Type rfr for Random Forest Regressor" + "\n"
                                     "Type rl for Randomized Lasso" + "\n"))
        x_train = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                              'data/x_train_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        x_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                             'data/x_test_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        y_train = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                              'data/y_train_postprocessing.csv')
        y_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                              'data/y_test_postprocessing.csv')
        x_train.drop(x_train.columns[0], axis=1, inplace=True)
        x_test.drop(x_test.columns[0], axis=1, inplace=True)
        y_train.drop(y_train.columns[0], axis=1, inplace=True)
        y_test.drop(y_test.columns[0], axis=1, inplace=True)

        # Data to training task
        # Type check inputs for sanity
        if x_train or x_test or y_train or y_test is None:
            raise ValueError('Empty dataframes')
        if TARGET_COLUMN is None:
            raise ValueError('target_column is None')
        if not isinstance(TARGET_COLUMN, str):
            raise TypeError('target_column is not a string')
        if TARGET_COLUMN not in y_train.columns or y_test.columns:
            raise ValueError('target_column (%s) is not a valid column name' % TARGET_COLUMN)

        print("Generating models")
        models.run_models(x_train, y_train, x_test, y_test)

    elif choice == 4:
        post_process.results()

    elif choice == 5:
        n_features = int(input("Choose the number of features to be plotted" + "\n" +
                               "Pick from 50, 100, 150, 200" + "\n"))
        feature_selector = str(input("The feature space can be reduced using "
                                     "Random Forest regressor or Randomized Lasso" + "\n" +
                                     "Type rfr for Random Forest Regressor" + "\n"
                                     "Type rl for Randomized Lasso" + "\n"))
        x_train = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                              'data/x_train_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        x_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                             'data/x_test_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        y_train = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                              'data/y_train_postprocessing.csv')
        y_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                             'data/y_test_postprocessing.csv')
        x_train.drop(x_train.columns[0], axis=1, inplace=True)
        x_test.drop(x_test.columns[0], axis=1, inplace=True)
        y_train.drop(y_train.columns[0], axis=1, inplace=True)
        y_test.drop(y_test.columns[0], axis=1, inplace=True)

    elif choice == 6:
        genalgo.main()



if __name__ == "__main__":
    main()
