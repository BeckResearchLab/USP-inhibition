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
access_keys = pd.read_csv('../keys/accessKeys.csv')
AWS_ACCESS_KEY_ID = access_keys['Access key ID'][0]
AWS_ACCESS_SECRET_KEY = access_keys['Secret access key'][0]
BUCKET = 'pphilip-usp-inhibition'


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

            # Joining separate descriptor dataframes
            df_x = utils.join_dataframes()

            df_x.to_csv('../data/df_x_preprocessing.csv')
            df_y.to_csv('../data/df_y_preprocessing.csv')

        else:
            df_x = pd.read_csv('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/data/df_x_preprocessing.csv')
            df_y = pd.read_csv('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/data/df_y_preprocessing.csv')
            df_x.drop(df_x.columns[0], axis=1, inplace=True)
            df_y.drop(df_y.columns[0], axis=1, inplace=True)

        # Copying column names to use after np array manipulation
        headers = list(df_x.columns.values)
        # Checking dataframe for NaN and infinite values
        df_x = utils.change_nan_infinite(df_x)
        df_y = utils.change_nan_infinite(df_y)
        # Transform all column values to mean 0 and unit variance
        df_x = sklearn.preprocessing.scale(df_x)
        # Feature selection and feature importance plot
        df_x, coefficients = utils.choose_features(df_x, df_y)
        print(df_x, coefficients)
        df_x = pd.DataFrame(df_x)
        df_y = pd.DataFrame(df_y)
        coefficients = pd.DataFrame({'existence': coefficients, 'column names': headers})
        df_x.to_csv('../data/df_x_postprocessing.csv')
        df_y.to_csv('../data/df_y_postprocessing.csv')

        coefficients.to_csv('../data/feature_coefficients.csv')

    else:
        df_x = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                           'data/df_x_postprocessing.csv')
        df_y = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                           'data/df_y_postprocessing.csv')
        coefficients = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/'
                                   'data/feature_coefficients.csv')
        df_x.drop(df_x.columns[0], axis=1, inplace=True)
        df_y.drop(df_y.columns[0], axis=1, inplace=True)
        coefficients.drop(coefficients.columns[0], axis=1, inplace=True)

    coefficients['existence'] = coefficients['existence'].astype(int)
    df_x.columns = list(coefficients[coefficients['existence'] == 1]['column names'])
    df_y.columns = ['Activity_Score']
    plot_input = input("Type 1 to plot feature space vs target column or 0 to skip: ")
    if plot_input == 1:
        utils.plot_features(df_x, df_y)
    df = df_x.join(df_y)

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

    # Train, validation and test split
    df_train, df_test = sklearn.cross_validation.train_test_split(df, test_size=0.25)

    # Reassign target column name and index after randomized split
    df_train = df_train.rename(columns={-1: TARGET_COLUMN})
    df_test = df_test.rename(columns={-1: TARGET_COLUMN})
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    # Remove the classification column from the dataframe
    x_train = np.array(df_train.drop(TARGET_COLUMN, axis=1))
    x_test = np.array(df_test.drop(TARGET_COLUMN, axis=1))
    y_train = np.array(df_train[TARGET_COLUMN])
    y_test = np.array(df_test[TARGET_COLUMN])

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
