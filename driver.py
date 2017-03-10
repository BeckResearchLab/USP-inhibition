#!/usr/bin/env python

"""
Primary execution file for USP-Inhibition project
"""
import sys
sys.path.append("/home/pphilip/Tools/openbabel-install/lib")

import descriptors
import genalgo
import models
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
XY_PICKLE = 'data/xy_data.pkl'


def main():
    """
    Module to execute the entire package from data retrieval to model results
    :return: None
    """

    # Importing dataset from NCBI database to create dataframe
    """response = pd.read_csv('https://pubchem.ncbi.nlm.nih.gov/pcajax/pcget.cgi?query=download&record_type='
                           'datatable&response_type=save&aid=743255&version=1.1')"""

    # Importing inhibitor notation data
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/compounds_smiles.txt')
    df_smiles = utils.create_notation_dataframe(response)

    # Importing inhibitor activity data
    response = pd.read_csv('https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/AID_743255_datatable.csv')
    df_activity = utils.create_activity_dataframe(response)

    df = df_activity.merge(df_smiles)
    df.drop(df.index[276743], inplace=True)
    df.sort_values(by='CID', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop non-descriptor columns before feature space reduction
    df_x = df.drop(['Activity_Score', 'CID'], axis=1)

    # Creating target column
    df_y = df.drop(['SMILES', 'CID'], axis=1)

    # Extracting molecular descriptors for all compounds
    print("Starting descriptor calculation")
    descriptors.extract_all_descriptors(df_x, 'SMILES')
    print("Finished descriptor calculation")

    print("Joining dataframes")
    df_x = utils.join_dataframes()
    print("Joined dataframes")
    df_x.to_csv('data/df_x_preprocessing.csv')
    df_y.to_csv('data/df_y_preprocessing.csv')

    print("Checking dataframe for NaN and infinite values")
    df_x = utils.remove_nan_infinite(df_x)
    df_y = utils.remove_nan_infinite(df_y)
    print("Checked dataframe for NaN and infinite values")

    # Transform all column values to mean 0 and unit variance
    print("Transforming dataframe using mean and variance")
    df_x = sklearn.preprocessing.scale(df_x)
    df_y = sklearn.preprocessing.scale(df_y)
    print("Transformed dataframe using mean and variance")

    # Feature selection and space reduction
    print("Selecting best features in dataframe")
    df_x = utils.choose_features(df_x, df_y)
    print("Selected best features in dataframe")

    df_x = pd.DataFrame(df_x)
    utils.plot_features(df_x, df_y)

    df_x.to_csv('data/df_x_postprocessing.csv')
    df_y.to_csv('data/df_y_postprocessing.csv')
    df = df_x.join(df_y)
    # Data to training task
    # Type check inputs for sanity
    if df is None:
        raise ValueError('df is None')
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df is not a dataframe')
    if TARGET_COLUMN is None:
        raise ValueError('target_column is None')
    if not isinstance(TARGET_COLUMN, basestring):
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
    print("Generated models and saved results")

    print("Finding candidate drug molecule using genetic algorithm")
    ideal_mol_features = genalgo.main()
    ideal_mol_features.to_csv('data/genalgo_results.csv')
    print("Found candidate drug molecule using genetic algorithm")


if __name__ == "__main__":
    main()
