#!/usr/bin/env python

"""
Primary execution file for USP-Inhibition project
"""
import sys
sys.path.append("/home/pphilip/Tools/openbabel-install/lib")

import utils
import pandas as pd
import models
import pickle
import post_process
import sklearn

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
    # The SMILES and InChI logs of the same material have identical indices
    # Creating and joining the SMILES and InChI dataframes along the same index
    utils.check_files()
    df_compounds_smiles = utils.create_dataframe('data/chemical_notation_'
                                                 'data/compounds_smiles.txt',
                                                 'smiles')
    df_compounds = df_compounds_smiles.rename(columns={'ID': 'CID'})
    df_compounds = df_compounds.sort_values(by='CID')

    # Importing inhibitor activity data
    activity = pd.read_csv('data/activity_data/AID_743255_datatable.csv')
    activity = utils.clean_activity_dataframe(activity)

    # Merging activity data and compound notation data
    df = activity.merge(df_compounds)
    df = df.sort_values(by='CID')
    df = df.reset_index(drop=True)

    # Drop non-descriptor columns before feature space reduction
    df_target = df.drop(['SMILES', 'CID', 'Phenotype'], axis=1)

    # Extracting molecular descriptors for all compounds
    print("Sending data for descriptor calculation")
    # utils.extract_all_descriptors(df, 'SMILES')

    # Importing feature sets
    df_charge = pd.DataFrame.from_csv('data/df_charge.csv')
    df_basak = pd.DataFrame.from_csv('data/df_basak.csv')
    df_con = pd.DataFrame.from_csv('data/df_con.csv')
    df_estate = pd.DataFrame.from_csv('data/df_estate.csv')
    df_constitution = pd.DataFrame.from_csv('data/df_constitution.csv')
    df_property = pd.DataFrame.from_csv('data/df_property.csv')
    df_kappa = pd.DataFrame.from_csv('data/df_kappa.csv')
    df_moe = pd.DataFrame.from_csv('data/df_moe.csv')

    df_descriptor = df_kappa.join(df_moe).join(df_constitution).\
        join(df_property).join(df_charge).join(df_estate).join(df_con).join(
        df_basak)

    # Transform all column values to mean 0 and unit variance
    df_descriptor = utils.transform_dataframe(df_descriptor)

    # Feature selection and space reduction
    utils.select_features(df_descriptor, df_target)

    # Import optimal feature space from pickle

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
