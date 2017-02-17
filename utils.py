#!/usr/bin/env python

"""
Perform data manipulation tasks and create inputs for project workflow
"""

import csv
import multiprocessing
import os
import pickle
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

import boto
import numpy as np
import pandas as pd
import sklearn
from boto.s3.key import Key
from joblib import Parallel, delayed
from pychem import getmol
from sklearn.ensemble import RandomForestRegressor


__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"


def create_notation_dataframe(filename):
    """
    Returns Pandas dataframe of sample ID and molecular notation

    Inputs: filename
    filename - path to file containing molecular notation indexed by sample ID
    Input types: str

    Outputs: dataframe of molecular notation indexed by sample ID
    Output types: Pandas DataFrame
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
    df.drop(276743)
    """"# Extract SMILES strings from CID values
    smiles = []
    i = 0
    for cid in df['CID']:
        string = getmol.GetMolFromNCBI(cid='%d' % cid)
        smiles.append(string)
        print(i)
        i += 1
    df['SMILES'] = smiles
    df.to_csv('data/NCBIdata.csv')"""
    return df


def upload_to_s3(aws_access_key_id, aws_secret_access_key, file_to_s3, bucket, key, callback=None, md5=None,
                 reduced_redundancy=False, content_type=None):
    """
    Uploads the given file to the AWS S3 bucket and key specified.

    callback is a function of the form:

    def callback(complete, total)

    The callback should accept two integer parameters, the first representing the number of bytes that have been
    successfully transmitted to S3 and the second representing the size of the to be transmitted object.

    Returns boolean indicating success/failure of upload.
    """
    try:
        size = os.fstat(file_to_s3.fileno()).st_size
    except:
        # Not all file objects implement fileno(),
        # so we fall back on this
        file_to_s3.seek(0, os.SEEK_END)
        size = file_to_s3.tell()

    conn = boto.connect_s3(aws_access_key_id, aws_secret_access_key)
    bucket = conn.get_bucket(bucket, validate=True)
    k = Key(bucket)
    k.key = key
    if content_type:
        k.set_metadata('Content-Type', content_type)
    sent = k.set_contents_from_file(file_to_s3, cb=callback, md5=md5, reduced_redundancy=reduced_redundancy,
                                    rewind=True)

    # Rewind for later use
    file_to_s3.seek(0)

    if sent == size:
        return True
    return False


def join_dataframes():
    """
    Joins the dataframes of existing descriptor files from their urls into a single dataframe.
    :return: Dataframe after joining
    """

    url_list = ['https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_constitution.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_con.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_kappa.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_estate.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_basak.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_property.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_charge.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_moe.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_burden.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_geary.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_moran.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_topology.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_geometric.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_cpsa.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_rdf.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_morse.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/df_whim.csv'
                ]

    url_exist_list = []
    for url in url_list:
        try:
            r = urllib2.urlopen(url)
        except urllib2.URLError as e:
            r = e
        if r.code < 400:
            url_exist_list.append(url)
        else:
            None

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


def choose_features(x, y):
    """
    Selecting the features of high importance to reduce feature space.
    :param x: Dataframe of features
    :param y: Dataframe of target property
    :return desired x: Dataframe of short-listed features
    """

    # Random forest feature importance
    x = np.array(x)
    y = np.array(y)

    clf = RandomForestRegressor()
    sfm = sklearn.feature_selection.SelectFromModel(clf, threshold=0.15)
    sfm.fit(x, y)
    print 'here'
    desired_x = sfm.transform(x)

    return desired_x


def remove_nan_infinite(dataframe):
    """
    :param dataframe: Dataframe containing NaN and infinite values
    :return dataframe: Dataframe with no NaN or infinite values
    """

    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.fillna(0, inplace=True)

    return dataframe

