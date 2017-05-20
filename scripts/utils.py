#!/usr/bin/env python

"""
Perform data manipulation tasks in project workflow
"""

import os
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

import boto
import numpy as np
import pandas as pd
import boto.s3
from boto.s3.key import Key
from sklearn.ensemble import RandomForestRegressor


__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"


def create_notation_dataframe(filename):
    """
    Returning Pandas dataframe of sample ID and molecular notation.
    :param filename: File object containing molecular notation indexed by sample ID
    :return: Dataframe of molecular notation indexed by sample ID.
    """
    df = []
    for line in filename:
        # Splits the line into it's key and molecular string
        words = line.split()
        z = [int(words[0]), words[1]]
        df.append(z)
    df = pd.DataFrame(df)
    df.columns = ['CID', 'SMILES']
    df.sort_values(by='CID', inplace=True)
    return df


def create_activity_dataframe(dataframe):
    """
    Performing useful transformations on the acquired data for use in subsequent algorithm.
    :param dataframe: Dataframe downloaded from NCBI database.
    :return: df: Cleaned and sorted dataframe.
    """

    # Eliminates first five text rows of csv
    for j in range(5):
        df = dataframe.drop(j, axis=0)
    df = df.drop(['PUBCHEM_ACTIVITY_URL', 'PUBCHEM_RESULT_TAG',
                  'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_SID',
                  'PUBCHEM_ASSAYDATA_COMMENT', 'Potency',
                  'Efficacy', 'Analysis Comment',
                  'Curve_Description', 'Fit_LogAC50',
                  'Fit_HillSlope', 'Fit_R2', 'Fit_ZeroActivity',
                  'Fit_CurveClass', 'Excluded_Points', 'Compound QC',
                  'Max_Response', 'Phenotype', 'Activity at 0.457 uM',
                  'Activity at 2.290 uM', 'Activity at 11.40 uM',
                  'Activity at 57.10 uM', 'PUBCHEM_ACTIVITY_OUTCOME',
                  'Fit_InfiniteActivity'], axis=1)
    df.rename(columns={'PUBCHEM_CID': 'CID'}, inplace=True)

    # Eliminates duplicate compound rows
    df['dupes'] = df.duplicated('CID')
    df = df[df['dupes'] == 0].drop(['dupes'], axis=1)
    df = df.sort_values(by='CID')
    return df


def upload_to_s3(aws_access_key_id, aws_secret_access_key, file_to_s3, bucket, key, callback=None, md5=None,
                 reduced_redundancy=False, content_type=None):
    """
    Uploads the given file to the AWS S3 bucket and key specified.
    :param aws_access_key_id: First part of AWS access key.
    :param aws_secret_access_key: Second part of AWS access key.
    :param file_to_s3: File object to be uploaded.
    :param bucket: S3 bucket name as string.
    :param key: Name attribute of the file object to be uploaded.
    :param callback: Function accepts two integer parameters, the first representing the number of bytes that have been
    successfully transmitted to S3 and the second representing the size of the to be transmitted object. Returns
    boolean indicating success/failure of upload.
    :param md5: MD5 checksum value to verify the integrity of the object.
    :param reduced_redundancy: S3 option that enables customers to reduce their costs
    by storing noncritical, reproducible data at lower levels of redundancy than S3's standard storage.
    :param content_type: Set the type of content in file object.
    :return: Boolean indicating success of upload.
    """
    try:
        size = os.fstat(file_to_s3.fileno()).st_size
    except:
        # Not all file objects implement fileno(), so we fall back on this
        file_to_s3.seek(0, os.SEEK_END)
        size = file_to_s3.tell()

    conn = boto.connect_s3(aws_access_key_id, aws_secret_access_key)
    bucket = conn.get_bucket(bucket, validate=True)
    k = Key(bucket)
    k.key = key
    if content_type:
        k.set_metadata('Content-Type', content_type)
    sent = k.set_contents_from_file(file_to_s3, cb=callback, md5=md5,
                                    reduced_redundancy=reduced_redundancy, rewind=True)

    # Rewind for later use
    file_to_s3.seek(0)

    if sent == size:
        return True
    return False


def join_dataframes():
    """
    Joining the dataframes of existing descriptor files from their urls into a single dataframe.
    :return: Dataframe after join over key column.
    """

    url_list = ['https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_constitution.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_con.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_kappa.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_estate.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_basak.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_property.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_charge.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_moe.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_burden.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_geary.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_moran.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_topology.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_geometric.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_cpsa.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_rdf.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_morse.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_whim.csv'
                ]

    url_exist_list = []
    for url in url_list:
        try:
            r = urllib2.urlopen(url)
        except urllib2.URLError as e:
            r = e
        if r.code < 400:
            url_exist_list.append(url)

    i = 0
    df = [0] * len(url_exist_list)
    for url in url_exist_list:
        df[i] = pd.read_csv(url)

        df[i].drop(df[i].columns[0], axis=1, inplace=True)
        df[i].reset_index(drop=True, inplace=True)
        i += 1

    joined_df = df[0]
    for i in df[1:]:
        joined_df = joined_df.join(i)
    return joined_df


def choose_features(x_train, y_train, x_test, column_names):
    """
    Selecting the features of high importance to reduce feature space.
    :param x_train: Training set of features.
    :param x_test: Test set of features.
    :param y_train: Training target values
    :param column_names: Names of columns in x
    """

    # Random forest feature importance
    clf = RandomForestRegressor(n_jobs=-1, random_state=1, n_estimators=20, max_depth=10)
    # Random state has int value for non-random sampling
    clf.fit(x_train, y_train)
    feature_importance = clf.feature_importances_
    scores_table = pd.DataFrame({'feature': column_names, 'scores':
                                 feature_importance}).sort_values(by=['scores'], ascending=False)
    scores = scores_table['scores'].tolist()
    n_features = [25, 50, 75, 100, 150, 200, 250, 300]
    for n in n_features:
        feature_scores = scores_table['feature'].tolist()
        selected_features = feature_scores[:n]
        x_train = pd.DataFrame(x_train, columns=column_names)
        desired_x_train = x_train[selected_features]
        x_test = pd.DataFrame(x_test, columns=column_names)
        desired_x_test = x_test[selected_features]

        desired_x_train.to_csv('../data/x_train_postprocessing_rfr_%d.csv' % n)
        desired_x_test.to_csv('../data/x_test_postprocessing_rfr_%d.csv' % n)
    pd.DataFrame(scores).to_csv('../data/feature_scores_rfr.csv')

    return


def change_nan_infinite(dataframe):
    """
    Replacing NaN and infinite values from the dataframe with zeros.
    :param dataframe: Dataframe containing NaN and infinite values.
    :return data: Data with no NaN or infinite values.
    """

    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = dataframe.fillna(0)

    return data
