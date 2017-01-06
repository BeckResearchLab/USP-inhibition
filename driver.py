#!/usr/bin/env python

"""
Primary execution file for USP-Inhibition project
"""
import sys
sys.path.append("/home/pphilip/Tools/openbabel-install/lib")

import csv
import models
import pandas as pd
import pickle
import post_process
import sklearn
import urllib2
import utils

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

TARGET_COLUMN = 'Activity_Score'
XY_PICKLE = 'xy_data.pkl'


def main():
    """
    Module to execute the entire package from data retrieval to model
    performance metrics
    @:param: None
    :return: Post process results
    """
    # Importing inhibitor notation data
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/compounds_smiles.txt')
    df_compounds_smiles = utils.create_notation_dataframe(response)

    # Importing inhibitor activity data
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/AID_743255_datatable.csv')
    activity = utils.create_activity_dataframe(response)

    # Merging activity data and compound notation data
    df = activity.merge(df_compounds_smiles)
    df.sort_values(by='ID', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop non-descriptor columns before feature space reduction
    df_target = df.drop(['SMILES', 'ID', 'Phenotype'], axis=1)

    # Extracting molecular descriptors for all compounds
    print("Starting descriptor calculation")
    utils.extract_all_descriptors(df, 'SMILES')
    print("Finished descriptor calculation")

    # Importing feature sets
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_charge.csv')
    df_charge = pd.DataFrame(list(csv.reader(response)))
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_basak.csv')
    df_basak = pd.DataFrame(list(csv.reader(response)))
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_con.csv')
    df_con = pd.DataFrame(list(csv.reader(response)))
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_estate.csv')
    df_estate = pd.DataFrame(list(csv.reader(response)))
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_constitution.csv')
    df_constitution = pd.DataFrame(list(csv.reader(response)))
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_property.csv')
    df_property = pd.DataFrame(list(csv.reader(response)))
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_kappa.csv')
    df_kappa = pd.DataFrame(list(csv.reader(response)))
    response = urllib2.urlopen('https://s3-us-west-2.amazonaws.com/'
                               'pphilip-usp-inhibition/df_moe.csv')
    df_moe = pd.DataFrame(list(csv.reader(response)))

    print("Joining dataframes")
    df_descriptor = df_kappa.join(df_moe).join(df_constitution).\
        join(df_property).join(df_charge).join(df_estate).join(df_con).join(
        df_basak)
    print("Joined dataframes")

    print("Checking dataframe for NaN, infinite or too large values")
    df_descriptor = utils.remove_nan_infinite(df_descriptor)
    print("Checked dataframe for NaN, infinite or too large values")

    # Transform all column values to mean 0 and unit variance
    print("Transforming dataframe using mean and variance")
    df_descriptor = utils.transform_dataframe(df_descriptor)
    print("Transformed dataframe using mean and variance")

    # Feature selection and space reduction
    print("Selecting best features in dataframe")
    df_features = utils.select_features(df_descriptor, df_target)
    print("Selected best features in dataframe")

    df = df_features.join(df_target)

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
        raise ValueError('target_column (%s) is not a valid column name'
                         % TARGET_COLUMN)

    # Train, validation and test split
    df_train, df_test = sklearn.cross_validation.train_test_split(df, test_size=0.25)

    # Remove the classification column from the dataframe
    x_train = df_train.drop(TARGET_COLUMN, 1)
    x_test = df_test.drop(TARGET_COLUMN, 1)
    y_train = pd.DataFrame(df_train[TARGET_COLUMN])
    y_test = pd.DataFrame(df_test[TARGET_COLUMN])

    with open(XY_PICKLE, 'wb') as results:
        pickle.dump(x_train, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(x_test, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_train, results, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_test, results, pickle.HIGHEST_PROTOCOL)

    models.run_models(x_train, y_train, x_test, y_test)

    post_process.results()

if __name__ == "__main__":
    main()
