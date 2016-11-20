#!/usr/bin/env python

"""
Perform data manipulation tasks and create inputs for project workflow
"""

import os
import pickle
from multiprocessing import Process

import numpy as np
import pandas as pd
import sklearn.feature_selection as f_selection
from pychem import bcut, estate, basak, moran, geary, molproperty as mp
from pychem import charge, moe, constitution
from pychem import topology, connectivity as con, kappa
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, SVR

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

FS_PICKLE = 'fs_results.pkl'


def create_dict(filename, mol):
    """
    Returns dictionary of sample ID and molecular notation
    
    Inputs: filename, mol
    filename - path to file containing molecular notation indexed by sample ID
    mol - type of molecular notation
    Input types: str, str
    Outputs: dictionary of molecular notation indexed by sample ID
    Output types: Python dictionary
    """
    with open(filename, 'r') as f:
        # Reads the file line by line
        data = f.readlines()
        # Null dictionary
        df = (dict([]))
        for line in data[:]:
            # Splits the line into it's key and molecular string  
            words = line.split()
            if mol == 'smiles':
                z = (dict([(int(words[0]), [words[1]])]))
            elif mol == 'inchi':
                # This removes the 'InChI=' prefix to the InChI string
                z = (dict([(int(words[0]), [words[1][6:]])]))
            else:
                print('Invalid molecular notation. Choose from smiles or inchi.')
            # Appending dictionary            
            df.update(z)
        return df


def create_dataframe(filename, mol):
    """
    Returns Pandas dataframe of sample ID and molecular notation
    
    Inputs: filename, mol
    filename - path to file containing molecular notation indexed by sample ID
    mol - type of molecular notation
    Input types: str, str
    
    Outputs: dataframe of molecular notation indexed by sample ID
    Output types: Pandas DataFrame
    """
    with open(filename, 'r') as f:
        # Reads the file line by line
        data = f.readlines()
        # Null dataframe
        df = []
        for line in data[:]:
            # Splits the line into it's key and molecular string  
            words = line.split()
            if mol == 'smiles':
                z = [int(words[0]), words[1]]
            elif mol == 'inchi':
                # This removes the 'InChI=' prefix to the InChI string
                z = [int(words[0]), words[1][6:]]
            else:
                print('Invalid molecular notation. Choose from smiles or inchi.')
            # Appending dictionary            
            df.append(z)
        df = pd.DataFrame(df)
        df.columns = ['ID', mol.upper()]
        return df


def clean_activity_dataframe(activity_df):
    """

    :param activity_df:
    :return:
    """
    # Eliminates first five text rows of csv
    for j in range(5):
        activity_df = activity_df.drop(j, axis=0)
    activity_df = activity_df.drop(['PUBCHEM_ACTIVITY_URL',
                                    'PUBCHEM_RESULT_TAG',
                                    'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_SID',
                                    'PUBCHEM_ASSAYDATA_COMMENT', 'Potency',
                                    'Efficacy', 'Analysis Comment',
                                    'Curve_Description', 'Fit_LogAC50',
                                    'Fit_HillSlope', 'Fit_R2',
                                    'Fit_InfiniteActivity',
                                    'Fit_ZeroActivity', 'Fit_CurveClass',
                                    'Excluded_Points', 'Compound QC',
                                    'Max_Response',
                                    'Activity at 0.457 uM',
                                    'Activity at 2.290 uM',
                                    'Activity at 11.40 uM',
                                    'Activity at 57.10 uM',
                                    'PUBCHEM_ACTIVITY_OUTCOME'], axis=1)
    activity_df.rename(columns={'PUBCHEM_CID': 'CID'}, inplace=True)
    # Eliminates duplicate compound rows
    activity_df['dupes'] = activity_df.duplicated('CID')
    activity_df = activity_df[activity_df['dupes'] == 0].drop(['dupes'],
                                                              axis=1)
    activity_df = activity_df.sort_values(by='CID')
    return activity_df


def sort_features(x, y):
    """

    :param x: dataframe of features
    :param y: dataframe of target property
    :return: Sorted score of all features
    """

    # Random forest feature importance - Mean decrease impurity
    names = x.columns.values.tolist()
    rf = RandomForestRegressor()
    rf.fit(x, y)
    rf_sorted_score = sorted(zip(map(lambda d: round(d, 4), rf.feature_importances_),
                                 names), reverse=True)
    return rf_sorted_score


def select_features(x, y):
    """

    :param x: dataframe of features
    :param y: dataframe of target property
    :return: Outputs of feature selection process
    """
    x = pd.DataFrame(x)

    # Removing features with low variance
    var_threshold = f_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))

    # Kbest-based and Percentile-based feature selection using regression
    f_regress = f_selection.f_regression(x, y, center=False)
    kbest = f_selection.SelectKBest(score_func=f_regress, k=2)
    percent = f_selection.SelectPercentile(score_func=f_regress, percentile=10)

    # Tree-based feature selection using a number of randomized decision trees
    trees = f_selection.SelectFromModel(ExtraTreesRegressor, prefit=True)

    # "False positive rate"-based feature selection using regression
    fpr = f_selection.SelectFpr(score_func=f_regress, alpha=0.05)

    # PCA-component evaluation
    pca = PCA(n_components=2)

    # Recursive feature elimination and cross-validated feature selection
    estimator = SVR(kernel="linear")
    selector = f_selection.RFECV(estimator, step=1, cv=5)

    # Build estimator from PCA and Univariate selection:
    combined_features = FeatureUnion([("pca_based", pca), ("univ_kbest", kbest), ("false_positive_rate", fpr),
                                      ("percentile_based", percent), ("RFECV_selector", selector),
                                      ("variance_threshold", var_threshold), ("trees_based", trees)])
    x_union_features = combined_features.fit_transform(x, y)

    svm = SVC(kernel="linear")

    # Do grid search over all parameters:
    pipeline = Pipeline([("features", x_union_features), ("svm", svm)])

    grid = dict(features__pca_based__n_components=range(1, 101),
                features__univ_kbest__k=range(1, 101),
                features_false_positive_rate_alpha=range(0, 1, 0.01),
                features_percentile_based_percentile=range(1, 20, 1),
                features_RFECV_selector_cv=range(1, 5),
                features_variance_threshold_threshold=range(0, 1, 0.01),
                svm__C=[0.01, 0.1, 1.0, 10.0])

    grid_search = GridSearchCV(pipeline, param_grid=grid, verbose=0)
    x_features = grid_search.fit_transform(x, y)

    # Pickling feature reduction outputs
    with open(FS_PICKLE, 'wb') as result:
        pickle.dump(rf_sorted_score, result, pickle.HIGHEST_PROTOCOL)
        pickle.dump(grid_search.best_estimator_, result, pickle.HIGHEST_PROTOCOL)

    print(grid_search.best_estimator_)

    return x_features


def extract_constitution_descriptors(dataframe, column):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    constitution descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_constitution.csv') and os.access('data/df_constitution.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting constitution calculation")
        diction = []
        for line in dataframe[column][:]:
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = constitution.GetConstitutional(mol)
            diction.append(dic)
        df_constitution = pd.DataFrame(diction, columns=["nphos", "ndb", "nsb", "ncoi",
                                                         "ncarb", "nsulph", "ncof",
                                                         "nnitro", "ncobr", "naro",
                                                         "ndonr", "noxy", "nhet",
                                                         "nhev", "nhal", "naccr",
                                                         "nta", "ntb", "nring", "nrot",
                                                         "Weight", "PC2", "PC3", "PC1",
                                                         "PC6", "PC4", "PC5", "AWeight",
                                                         "ncocl", "nhyd"])

        df_constitution.to_csv('data/df_constitution.csv')
        print("done calculating constitution")
        return


def extract_topology_descriptors(dataframe, column):
    """
    Extracting molecular topology descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which topology
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_topology.csv') and os.access('data/df_topology.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting topology calculation")
        diction = []
        i = 0
        for line in dataframe[column][:]:
            smiles = line
            i += 1
            print("topology")
            print(i)
            mol = Chem.MolFromSmiles(smiles)
            dic = topology.GetTopology(mol)
            diction.append(dic)
        df_topology = pd.DataFrame(diction, columns=['GMTIV', 'AW', 'Geto', 'DZ',
                                                     'Gravto', 'IDET', 'Sitov',
                                                     'IDE', 'TIAC', 'Arto',
                                                     'Qindex', 'petitjeant',
                                                     'Hatov', 'diametert',
                                                     'BertzCT', 'IVDE', 'ISIZ',
                                                     'Platt', 'ZM2', 'Getov',
                                                     'ZM1', 'J', 'radiust',
                                                     'Tsch', 'Thara', 'W', 'MZM2',
                                                     'GMTI', 'MZM1', 'Ipc',
                                                     'Sito', 'Tigdi', 'Pol',
                                                     'Hato', 'Xu'])

        df_topology.to_csv('data/df_topology.csv')
        print("done calculating topology")
        return


def extract_con_descriptors(dataframe, column):
    """
    Extracting molecular connectivity descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    connectivity descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_con.csv') and os.access('data/df_con.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting con calculation")
        diction = []
        for line in dataframe[column][:]:
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = con.GetConnectivity(mol)
            diction.append(dic)
        df_con = pd.DataFrame(diction, columns=['Chi3ch', 'knotp', 'dchi3',
                                                'dchi2', 'dchi1', 'dchi0',
                                                'Chi5ch', 'Chiv4', 'Chiv7',
                                                'Chiv6', 'Chiv1', 'Chiv0',
                                                'Chiv3', 'Chiv2', 'Chi4c',
                                                'dchi4', 'Chiv4pc', 'Chiv3c',
                                                'Chiv8', 'Chi3c', 'Chi8',
                                                'Chi9', 'Chi2', 'Chi3', 'Chi0',
                                                'Chi1', 'Chi6', 'Chi7', 'Chi4',
                                                'Chi5', 'Chiv5', 'Chiv4c',
                                                'Chiv9', 'Chi4pc', 'knotpv',
                                                'Chiv5ch', 'Chiv3ch', 'Chiv10',
                                                'Chiv6ch', 'Chi10', 'Chi4ch',
                                                'Chiv4ch', 'mChi1', 'Chi6ch'])
        df_con.to_csv('data/df_con.csv')
        print("done calculating con")
        return


def extract_kappa_descriptors(dataframe, column):
    """
    Extracting molecular kappa descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Kappa
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_kappa.csv') and os.access('data/df_kappa.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting kappa calculation")
        diction = []
        for line in dataframe[column][:]:
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = kappa.GetKappa(mol)
            diction.append(dic)
        df_kappa = pd.DataFrame(diction, columns=['phi', 'kappa1', 'kappa3',
                                                  'kappa2', 'kappam1', 'kappam3',
                                                  'kappam2'])

        df_kappa.to_csv('data/df_kappa.csv')
        print("done calculating kappa")
        return


def extract_burden_descriptors(dataframe, column):
    """
    Extracting molecular burden descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Burden
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_burden.csv') and os.access('data/df_burden.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting burden calculation")
        diction = []
        i = 0
        for line in dataframe[column][:]:
            smiles = line
            i += 1
            print("burden")
            print(i)
            mol = Chem.MolFromSmiles(smiles)
            dic = bcut.GetBurden(mol)
            diction.append(dic)
        df_burden = pd.DataFrame(diction, columns=['bcutp8', 'bcutm9', 'bcutp9',
                                                   'bcutp5', 'bcutp6', 'bcutm8',
                                                   'bcutp1', 'bcutp2', 'bcutp3',
                                                   'bcutm7', 'bcute9', 'bcutv8',
                                                   'bcutv9', 'bcutv6', 'bcutm6',
                                                   'bcutv4', 'bcutm4', 'bcutm3',
                                                   'bcutm5', 'bcutm1', 'bcutv1',
                                                   'bcutv5', 'bcute8', 'bcutv2',
                                                   'bcutm2', 'bcutp4', 'bcute3',
                                                   'bcutv14', 'bcutv15',
                                                   'bcutv16', 'bcutv10',
                                                   'bcutv11', 'bcutv12',
                                                   'bcutv13', 'bcutp7',
                                                   'bcutp16', 'bcutp14',
                                                   'bcutp15', 'bcutp12',
                                                   'bcutp13', 'bcutp10',
                                                   'bcutp11', 'bcute16',
                                                   'bcute15', 'bcute14',
                                                   'bcute13', 'bcute12',
                                                   'bcute11', 'bcute10',
                                                   'bcutv3', 'bcute7', 'bcute6',
                                                   'bcute5', 'bcute4', 'bcutv7',
                                                   'bcute2', 'bcute1', 'bcutm16',
                                                   'bcutm15', 'bcutm14',
                                                   'bcutm13', 'bcutm12',
                                                   'bcutm11', 'bcutm10'])

        df_burden.to_csv('data/df_burden.csv')
        print("done calculating burden")
        return


def extract_estate_descriptors(dataframe, column):
    """
    Extracting molecular E-state descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which E-State
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_estate.csv') and os.access('data/df_estate.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting estate calculation")
        diction = []
        for line in dataframe[column][:]:
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = estate.GetEstate(mol)
            diction.append(dic)
        df_estate = pd.DataFrame(diction, columns=['Smax38', 'Smax39', 'Smax34',
                                                   'Smax35', 'Smax36', 'Smax37',
                                                   'Smax30', 'Smax31', 'Smax32',
                                                   'Smax33', 'S57', 'S56', 'S55',
                                                   'S54', 'S53', 'S52', 'S51',
                                                   'S50', 'S32', 'S59', 'S58',
                                                   'Smax8', 'Smax9', 'Smax0',
                                                   'Smax1', 'Smax2', 'Smax3',
                                                   'Smax4', 'Smax5', 'Smax6',
                                                   'Smax7', 'Smax29', 'Smax28',
                                                   'Smax23', 'Smax22', 'Smax21',
                                                   'Smax20', 'Smax27', 'Smax26',
                                                   'Smax25', 'Smax24', 'S44',
                                                   'S45', 'S46', 'S47', 'S40',
                                                   'S41', 'S42', 'S43', 'S48',
                                                   'S49', 'Smin78', 'Smin72',
                                                   'Smin73', 'Smin70', 'Smin71',
                                                   'Smin76', 'Smin77', 'Smin74',
                                                   'Smin75', 'S79', 'S78',
                                                   'Smin', 'Smax58', 'Smax59',
                                                   'Smax56', 'Smax57', 'S73',
                                                   'S72', 'S75', 'Smax53', 'S77',
                                                   'S76', 'Save', 'Smin69',
                                                   'Smin68', 'Shal', 'Smin61',
                                                   'Smin32', 'Smin63', 'Smin62',
                                                   'Smin65', 'Smin64', 'Smin67',
                                                   'Smin66', 'DS', 'Smin41',
                                                   'Smin40', 'Smax49', 'Smax48',
                                                   'S68', 'S69', 'Smax45',
                                                   'Smax44', 'Smax47', 'S65',
                                                   'Smax41', 'Smax40', 'Smax43',
                                                   'Smax42', 'Smin54', 'Smax52',
                                                   'Smin56', 'Smin57', 'Smin50',
                                                   'Smin51', 'Smin52', 'Smin53',
                                                   'Smin58', 'Smin59', 'Shev',
                                                   'Shet', 'Scar', 'Smin49',
                                                   'S9', 'S8', 'S3', 'S2', 'S1',
                                                   'Smin55', 'S7', 'S6', 'S5',
                                                   'S4', 'Smax78', 'S66', 'S67',
                                                   'Smax70', 'Smax71', 'Smax72',
                                                   'Smax73', 'Smax74', 'Smax75',
                                                   'Smax76', 'Smax77', 'Smin43',
                                                   'Smin42', 'S19', 'S18',
                                                   'Smin47', 'Smin46', 'Smin45',
                                                   'Smin44', 'S13', 'S12', 'S11',
                                                   'S10', 'S17', 'S16', 'S15',
                                                   'S14', 'S60', 'S64', 'Smin16',
                                                   'S61', 'Smax67', 'Smax66',
                                                   'Smax65', 'Smax64', 'Smax63',
                                                   'Smax62', 'Smax61', 'Smax60',
                                                   'Smax69', 'Smax68', 'Smin60',
                                                   'Smax', 'Smin36', 'Smin37',
                                                   'Smin34', 'Smin35', 'S62',
                                                   'Smin33', 'Smin30', 'Smin31',
                                                   'Smin38', 'Smin39', 'Smax12',
                                                   'Smax13', 'Smax10', 'Smax11',
                                                   'Smax16', 'Smax17', 'Smax14',
                                                   'Smax15', 'Smin20', 'Smax18',
                                                   'Smax19', 'S71', 'S63', 'S70',
                                                   'Smax54', 'Smax55', 'S39',
                                                   'S38', 'S35', 'S34', 'S37',
                                                   'S36', 'S31', 'S30', 'S33',
                                                   'S74', 'Smin25', 'Smin24',
                                                   'Smin27', 'Smin26', 'Smin21',
                                                   'Smax50', 'Smin23', 'Smin22',
                                                   'Smax51', 'Smin29', 'Smin28',
                                                   'Smin6', 'Smin7', 'Smin4',
                                                   'Smin5', 'Smin2', 'Smin3',
                                                   'Smin0', 'Smin1', 'Smin48',
                                                   'Smin8', 'Smin9', 'S22',
                                                   'S23', 'S20', 'S21', 'S26',
                                                   'S27', 'S24', 'S25', 'S28',
                                                   'S29', 'Smin10', 'Smin11',
                                                   'Smin12', 'Smin13', 'Smin14',
                                                   'Smin15', 'Smax46', 'Smin17',
                                                   'Smin18', 'Smin19'])
        df_estate.to_csv('data/df_estate.csv')
        print("done calculating estate")
        return


def extract_basak_descriptors(dataframe, column):
    """
    Extracting molecular basak descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Basak
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_basak.csv') and os.access('data/df_basak.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting basak calculation")
        diction = []
        i = 0
        for line in dataframe[column][:]:
            smiles = line
            i += 1
            print(i)
            mol = Chem.MolFromSmiles(smiles)
            dic = basak.Getbasak(mol)
            diction.append(dic)
        df_basak = pd.DataFrame(diction, columns=['CIC3', 'CIC6', 'SIC5', 'SIC4',
                                                  'SIC6', 'SIC1', 'SIC0', 'SIC3',
                                                  'SIC2', 'CIC5', 'CIC2', 'CIC0',
                                                  'CIC4', 'IC3', 'IC2', 'IC1',
                                                  'IC0', 'CIC1', 'IC6', 'IC5',
                                                  'IC4'])

        df_basak.to_csv('data/df_basak.csv')
        print("done calculating basak")
        return


def extract_moran_descriptors(dataframe, column):
    """
    Extracting molecular moran descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Moran
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_moran.csv') and os.access('data/df_moran.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting moran calculation")
        diction = []
        i = 0
        for line in dataframe[column][:]:
            smiles = line
            i += 1
            print("moran")
            print(i)
            mol = Chem.MolFromSmiles(smiles)
            dic = moran.GetMoranAuto(mol)
            diction.append(dic)
        df_moran = pd.DataFrame(diction, columns=['MATSv8', 'MATSp4', 'MATSp8',
                                                  'MATSv1', 'MATSp6', 'MATSv3',
                                                  'MATSv2', 'MATSv5', 'MATSv4',
                                                  'MATSv7', 'MATSv6', 'MATSm8',
                                                  'MATSp1', 'MATSm4', 'MATSm5',
                                                  'MATSm6', 'MATSm7', 'MATSm1',
                                                  'MATSm2', 'MATSm3', 'MATSe4',
                                                  'MATSe5', 'MATSe6', 'MATSe7',
                                                  'MATSe1', 'MATSe2', 'MATSe3',
                                                  'MATSe8', 'MATSp3', 'MATSp7',
                                                  'MATSp5', 'MATSp2'])

        df_moran.to_csv('data/df_moran.csv')
        print("done calculating moran")
        return


def extract_geary_descriptors(dataframe, column):
    """
    Extracting molecular geary descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Geary
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_geary.csv') and os.access('data/df_geary.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting geary calculation")
        diction = []
        i = 0
        for line in dataframe[column][:]:
            smiles = line
            i += 1
            print("geary")
            print(i)
            mol = Chem.MolFromSmiles(smiles)
            dic = geary.GetGearyAuto(mol)
            diction.append(dic)
        df_geary = pd.DataFrame(diction, columns=['GATSp8', 'GATSv3', 'GATSv2',
                                                  'GATSv1', 'GATSp6', 'GATSv7',
                                                  'GATSv6', 'GATSv5', 'GATSv4',
                                                  'GATSe2', 'GATSe3', 'GATSv8',
                                                  'GATSe6', 'GATSe7', 'GATSe4',
                                                  'GATSe5', 'GATSp5', 'GATSp4',
                                                  'GATSp7', 'GATSe1', 'GATSp1',
                                                  'GATSp3', 'GATSp2', 'GATSe8',
                                                  'GATSm2', 'GATSm3', 'GATSm1',
                                                  'GATSm6', 'GATSm7', 'GATSm4',
                                                  'GATSm5', 'GATSm8'])

        df_geary.to_csv('data/df_geary.csv')
        print("done calculating geary")
        return


def extract_property_descriptors(dataframe, column):
    """
    Extracting molecular property descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which property
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_property.csv') and os.access('data/df_property.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting property calculation")
        diction = []
        for line in dataframe[column]:
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = mp.GetMolecularProperty(mol)
            diction.append(dic)
        df_property = pd.DataFrame(diction, columns=['TPSA', 'Hy', 'LogP',
                                                     'LogP2', 'UI', 'MR'])

        df_property.to_csv('data/df_property.csv')
        print("done calculating property")
        return


def extract_charge_descriptors(dataframe, column):
    """
    Extracting molecular charge descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which charge
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_charge.csv') and os.access('data/df_charge.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting charge calculation")
        diction = []
        for line in dataframe[column][:]:
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = charge.GetCharge(mol)
            diction.append(dic)
        df_charge = pd.DataFrame(diction, columns=['QNmin', 'QOss', 'Mpc',
                                                   'QHss', 'SPP', 'LDI', 'QCmin',
                                                   'Mac', 'Qass', 'QNss',
                                                   'QCmax', 'QOmax', 'Tpc',
                                                   'Qmax', 'QOmin', 'Tnc',
                                                   'QHmin', 'QCss', 'QHmax',
                                                   'QNmax', 'Rnc', 'Rpc', 'Qmin',
                                                   'Tac', 'Mnc'])

        df_charge.to_csv('data/df_charge.csv')
        print("done calculating charge")
        return


def extract_moe_descriptors(dataframe, column):
    """
    Extracting molecular MOE-type descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which MOE
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :return: Descriptor dataframe
    """
    if os.path.exists('data/df_moe.csv') and os.access('data/df_moe.csv', os.R_OK):
        print("File exists and is readable")
        return
    else:
        print("starting moe calculation")
        diction = []
        for line in dataframe[column][:]:
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = moe.GetMOE(mol)
            diction.append(dic)
        df_moe = pd.DataFrame(diction, columns=['EstateVSA8', 'EstateVSA9',
                                                'EstateVSA4', 'EstateVSA5',
                                                'EstateVSA6', 'EstateVSA7',
                                                'EstateVSA0', 'EstateVSA1',
                                                'EstateVSA2', 'EstateVSA3',
                                                'PEOEVSA13', 'PEOEVSA12',
                                                'PEOEVSA11', 'PEOEVSA10',
                                                'VSAEstate0', 'VSAEstate1',
                                                'VSAEstate2', 'VSAEstate3',
                                                'VSAEstate4', 'VSAEstate5',
                                                'VSAEstate6', 'VSAEstate7',
                                                'VSAEstate8', 'LabuteASA',
                                                'PEOEVSA3', 'PEOEVSA2',
                                                'PEOEVSA1', 'PEOEVSA0',
                                                'PEOEVSA7', 'PEOEVSA6',
                                                'PEOEVSA5', 'PEOEVSA4',
                                                'MRVSA5', 'MRVSA4', 'PEOEVSA9',
                                                'PEOEVSA8', 'MRVSA1', 'MRVSA0',
                                                'MRVSA3', 'MRVSA2', 'MRVSA9',
                                                'TPSA1', 'slogPVSA10',
                                                'slogPVSA11', 'MRVSA8', 'MRVSA7',
                                                'MRVSA6', 'EstateVSA10',
                                                'slogPVSA2', 'slogPVSA3',
                                                'slogPVSA0', 'slogPVSA1',
                                                'slogPVSA6', 'slogPVSA7',
                                                'slogPVSA4', 'slogPVSA5',
                                                'slogPVSA8', 'slogPVSA9',
                                                'VSAEstate9', 'VSAEstate10'])

        df_moe.to_csv('data/df_moe.csv')
        print("done calculating moe")
        return


def extract_all_descriptors(df, column):
    """
    Extracting all molecular descriptors using PyChem package and
    SMILES strings of compounds.
    :param df: The dataframe containing SMILES info for which
                      all descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    p1 = Process(target=extract_constitution_descriptors, args=(df, column))
    p1.start()

    p2 = Process(target=extract_topology_descriptors, args=(df, column))
    p2.start()

    p3 = Process(target=extract_con_descriptors, args=(df, column))
    p3.start()

    p4 = Process(target=extract_kappa_descriptors, args=(df, column))
    p4.start()

    p5 = Process(target=extract_burden_descriptors, args=(df, column))
    p5.start()

    p6 = Process(target=extract_estate_descriptors, args=(df, column))
    p6.start()

    p7 = Process(target=extract_basak_descriptors, args=(df, column))
    p7.start()

    p8 = Process(target=extract_moran_descriptors, args=(df, column))
    p8.start()

    p9 = Process(target=extract_geary_descriptors, args=(df, column))
    p9.start()

    p10 = Process(target=extract_property_descriptors, args=(df, column))
    p10.start()

    p11 = Process(target=extract_charge_descriptors, args=(df, column))
    p11.start()

    p12 = Process(target=extract_moe_descriptors, args=(df, column))
    p12.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()

    return


def check_files():
    a = 0
    with open('data/chemical_notation_data/compounds_smiles.txt', 'r') as f:
        data = f.readlines()
        i = 1
        for line in data:
            i += 1
        if i == 389561:
            print("Compound information file check done")

    file_string = ['charge', 'con', 'constitution', 'estate', 'kappa', 'moe',
                   'property', 'basak', 'burden', 'geary', 'moran', 'topology']
    for string in file_string:
        if os.path.exists('data/df_%s.csv' % string) and os.access(
                        'data/df_%s.csv' % string, os.R_OK):
            df = pd.DataFrame.from_csv('data/df_%s.csv' % string)
            row = df.shape[0]
            if row == 389560:
                print("df_%s file check done" % string)
            else:
                print("Incorrect df_%s file length" % string)
        else:
            print("df_%s file does not exist" % string)


def remove_nan_infinite(dataframe):
    """

    :param dataframe: Dataframe undergoing further transformation and containing NaN and infinite values
    :return dataframe: Corrected dataframe with no NaN or infinite values
    """

    if np.any(np.isnan(dataframe)) == True and np.all(np.isfinite(dataframe)) == False:
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.fillna(0, inplace=True)

    return dataframe


def transform_dataframe(dataframe):

    """
    Function to read and standardize the dataframe with a
    mean 0 and unit variance on every column

    Parameters:
        dataframe : Input pandas dataframe
    Input types: pd.Dataframe
    Output types: pd.Dataframe

    """
    robust_scaler = RobustScaler()
    df = robust_scaler.fit_transform(dataframe)
    return df
