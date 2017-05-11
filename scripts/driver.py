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
    run = input("Type 1 to run ML models from raw assay data/calculated descriptor data "
                "or 0 to run ML models from stored processed data: ")
    if run == 1:

        run = input("Type 1 to use raw assay data or 0 to "
                    "use calculated descriptor data: ")
        if run == 1:

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

        else:
            df = pd.read_csv('https://s3-us-west-2.amazonaws.com/'
                             'pphilip-usp-inhibition/data/df_preprocessing.csv')
            df.drop(df.columns[0], axis=1, inplace=True)

        # Copying column names to use after np array manipulation
        x = df.drop(df.columns[-1], axis=1)
        headers = list(x.columns.values)

        # Train, validation and test split
        df_train, df_test = sklearn.cross_validation.train_test_split(df, test_size=0.25)
        # Reassign target column name and index after randomized split
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)

        # Remove the classification column from the dataframe
        x_train = np.array(df_train.drop(TARGET_COLUMN, axis=1))
        x_test = np.array(df_test.drop(TARGET_COLUMN, axis=1))
        y_train = np.array(df_train[TARGET_COLUMN])
        y_test = np.array(df_test[TARGET_COLUMN])

        # Transform all column values to mean 0 and unit variance
        clf = sklearn.preprocessing.StandardScaler().fit(x_train)
        x_train = clf.transform(x_train)
        x_test = clf.transform(x_test)

        # Checking dataframe for NaN and infinite values
        x_train = utils.change_nan_infinite(x_train)
        y_train = utils.change_nan_infinite(x_train)
        x_test = utils.change_nan_infinite(x_train)
        y_test = utils.change_nan_infinite(x_train)

        n_features = int(input("Choose the number of features to be used in the model" + "\n" +
                               "Pick from 50, 100, 150, 200" + "\n"))

        # Feature selection and feature importance plot
        x_train, y_train, x_test, y_test = utils.choose_features(x_train, y_train, x_test, y_test, headers,
                                                                 n_features)

    else:
        n_features = int(input("Choose the number of features to be used in the model" + "\n" +
                               "Pick from 50, 100, 150, 200" + "\n"))
        feature_selector = str(input("Choose the algorithm that is used to reduce the feature space." + "\n" +
                                     "Type rfr for Random Forest Regressor" + "\n"
                                     "Type rl for Randomized Lasso" + "\n"))
        x_train = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                              'data/x_train_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        y_train = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                              'data/y_train_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        x_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                             'data/x_test_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        y_test = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                             'data/y_test_postprocessing_%s_%d.csv' % (feature_selector, n_features))
        x_train.drop(x_train.columns[0], axis=1, inplace=True)
        y_train.drop(y_train.columns[0], axis=1, inplace=True)
        x_test.drop(x_test.columns[0], axis=1, inplace=True)
        y_test.drop(y_test.columns[0], axis=1, inplace=True)

    plot_input = input("Type 1 to plot feature space vs target column or 0 to skip: ")
    if plot_input == 1:
        utils.plot_features(x_train, y_train)

    # Data to training task
    # Type check inputs for sanity
    if df is None:
        raise ValueError('df is None')
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a dataframe')
    if TARGET_COLUMN is None:
        raise ValueError('target_column is None')
    if not isinstance(TARGET_COLUMN, str):
        raise TypeError('target_column is not a string')
    if TARGET_COLUMN not in df.columns:
        raise ValueError('target_column (%s) is not a valid column name' % TARGET_COLUMN)

    with open(XY_PICKLE, 'wb') as results:
        pickle.dump(x_train, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(x_test, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_train, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_test, results, pickle.HIGHEST_PROTOCOL)

    print("Generating models")
    models.run_models(x_train, y_train, x_test, y_test)

    results_display = input("Type 1 to print ML model and prediction results or 0 to skip: ")
    if results_display == 1:
        post_process.results()

    ga_input = input("Type 1 to find candidate drug molecule using genetic algorithms or 0 to skip: ")
    if ga_input == 1:
        ideal_mol_features = genalgo.main()
        ideal_mol_features.to_csv('../data/genalgo_results.csv')


if __name__ == "__main__":
    main()
