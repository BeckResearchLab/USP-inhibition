import sys
sys.path.append("/home/pphilip/Tools/openbabel-install/lib")


import utils
import pandas as pd
import models
import post_process
from sklearn.cross_validation import train_test_split

TARGET_COLUMN = 'Activity_Score'

# To find the number of compounds tested; expected 389561
with open('data/chemical_notation_data/compounds_smiles.txt', 'r') as f:
    data = f.readlines()
    i = 1
    for line in data:
        words = line.split()
        i += 1
    print i


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

    df_compounds_smiles = utils.create_dataframe('data/chemical_notation_data/'
                                                 'compounds_smiles.txt', 'smiles')
    df_compounds = df_compounds_smiles.rename(columns={'ID': 'CID'})
    df_compounds = df_compounds.sort_values(by='CID')

    # Importing inhibitor activity data
    activity = pd.read_csv('data/activity_data/AID_743255_datatable.csv')
    activity = utils.clean_activity_dataframe(activity)

    # Merging activity data and compound notation data
    df = activity.merge(df_compounds)
    df = df.sort_values(by='CID')
    df = df.sample(frac=1).reset_index(drop=True)
    # Extracting molecular descriptors for all compounds
    df_descriptor = utils.extract_all_descriptors(df, 'SMILES')
    # Transform all column values to mean 0 and unit variance
    df_descriptor = utils.transform_dataframe(df_descriptor)
    # Drop non-descriptor columns before training
    df = df.drop(['SMILES', 'CID',	'Phenotype'], axis=1)
    df = df.join(df_descriptor)
    # Feature selection and reduction
    # Data to training task
    df.to_csv('data/descriptor_data.csv')

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
    df_train, df_test = train_test_split(df, test_size=0.25)
    df_train, df_val = train_test_split(df_train, test_size=0.333333)
    x_train, x_val, x_test = df_train, df_val, df_test

    # Remove the classification column from the dataframe
    x_train = x_train.drop(TARGET_COLUMN, 1)
    x_val = x_val.drop(TARGET_COLUMN, 1)
    x_test = x_test.drop(TARGET_COLUMN, 1)
    y_train = pd.DataFrame(df_train[TARGET_COLUMN])
    y_val = pd.DataFrame(df_val[TARGET_COLUMN])
    y_test = pd.DataFrame(df_test[TARGET_COLUMN])


    # models.build_nn(x_train, y_train, x_val, y_val)
    # models.build_svm(x_train, y_train, x_val, y_val)

    # post_process.results()

if __name__ == "__main__":
    main()
