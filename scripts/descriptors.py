#!/usr/bin/env python

"""
Perform data manipulation tasks and create inputs for project workflow
"""

from multiprocessing import Process
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

import pandas as pd
import pybel as pyb
from pychem import bcut, estate, basak, moran, geary, molproperty as mp
from pychem import charge, moe, constitution, topology, kappa, whim
from pychem import connectivity as con, geometric, cpsa, rdf, morse
from pychem.pychem import Chem, GetARCFile
import numpy as np
import signal

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"


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
    url_list = ['https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_constitution.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_topology.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_con.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_kappa.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_burden.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_estate.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_basak.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_moran.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_geary.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_property.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_charge.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_moe.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_geometric.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_cpsa.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_rdf.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_morse.csv',
                'https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/df_whim.csv']

    p1 = Process(target=extract_constitution_descriptors, args=(df, column, url_list[0]))
    p1.start()

    # p2 = Process(target=extract_topology_descriptors, args=(df, column, url_list[1]))
    # p2.start()

    p3 = Process(target=extract_con_descriptors, args=(df, column, url_list[2]))
    p3.start()

    p4 = Process(target=extract_kappa_descriptors, args=(df, column, url_list[3]))
    p4.start()

    p5 = Process(target=extract_burden_descriptors, args=(df, column, url_list[4]))
    p5.start()

    p6 = Process(target=extract_estate_descriptors, args=(df, column, url_list[5]))
    p6.start()

    p7 = Process(target=extract_basak_descriptors, args=(df, column, url_list[6]))
    p7.start()

    p8 = Process(target=extract_moran_descriptors, args=(df, column, url_list[7]))
    p8.start()

    p9 = Process(target=extract_geary_descriptors, args=(df, column, url_list[8]))
    p9.start()

    p10 = Process(target=extract_property_descriptors, args=(df, column, url_list[9]))
    p10.start()

    p11 = Process(target=extract_charge_descriptors, args=(df, column, url_list[10]))
    p11.start()

    p12 = Process(target=extract_moe_descriptors, args=(df, column, url_list[11]))
    p12.start()

    """p1.join()
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
        p12.join()"""

    p13 = Process(target=extract_geometric_descriptors, args=(df, column, url_list[12]))
    p13.start()
    p13.join()

    p14 = Process(target=extract_cpsa_descriptors, args=(df, column, url_list[13]))
    p14.start()
    p14.join()

    p15 = Process(target=extract_rdf_descriptors, args=(df, column, url_list[14]))
    p15.start()
    p15.join()

    p16 = Process(target=extract_morse_descriptors, args=(df, column, url_list[15]))
    p16.start()
    p16.join()

    p17 = Process(target=extract_whim_descriptors, args=(df, column, url_list[16]))
    p17.start()
    p17.join()

    return


def extract_constitution_descriptors(dataframe, column, url):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    constitution descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting constitution calculation")
        diction = []
        columns = ["nphos", "ndb", "nsb", "ncoi",
                   "ncarb", "nsulph", "ncof",
                   "nnitro", "ncobr", "naro",
                   "ndonr", "noxy", "nhet",
                   "nhev", "nhal", "naccr",
                   "nta", "ntb", "nring", "nrot",
                   "Weight", "PC2", "PC3", "PC1",
                   "PC6", "PC4", "PC5", "AWeight",
                   "ncocl", "nhyd"]
        i = 0
        for line in dataframe[column]:
            i += 1
            print("constitution ", i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = constitution.GetConstitutional(mol)
            diction.append(dic)
        df_constitution = pd.DataFrame(diction, columns=columns)
        df_constitution.to_csv('../data/df_constitution.csv')
        print("Done calculating constitution")

        return


def extract_topology_descriptors(dataframe, column, url):
    """
    Extracting molecular topology descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which topology
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting topology calculation")
        diction = []
        columns = ['GMTIV', 'AW', 'Geto', 'DZ',
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
                   'Hato', 'Xu']
        i = 0
        for line in dataframe[column]:
            i += 1
            print("topology ", i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = topology.GetTopology(mol)
            diction.append(dic)
        df_topology = pd.DataFrame(diction, columns=columns)
        df_topology.to_csv('../data/df_topology.csv')
        print("Done calculating topology")

        return


def extract_con_descriptors(dataframe, column, url):
    """
    Extracting molecular connectivity descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    connectivity descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting con calculation")
        diction = []
        columns = ['Chi3ch', 'knotp', 'dchi3',
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
                   'Chiv4ch', 'mChi1', 'Chi6ch']
        i = 0
        for line in dataframe[column]:
            i += 1
            print("con ", i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = con.GetConnectivity(mol)
            diction.append(dic)
        df_con = pd.DataFrame(diction, columns=columns)
        df_con.to_csv('../data/df_con.csv')
        print("Done calculating con")

        return


def extract_kappa_descriptors(dataframe, column, url):
    """
    Extracting molecular kappa descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Kappa
    descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting kappa calculation")
        diction = []
        columns = ['phi', 'kappa1', 'kappa3', 'kappa2',
                   'kappam1', 'kappam3', 'kappam2']
        i = 0
        for line in dataframe[column]:
            i += 1
            print("kappa ", i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = kappa.GetKappa(mol)
            diction.append(dic)
        df_kappa = pd.DataFrame(diction, columns=columns)
        df_kappa.to_csv('../data/df_kappa.csv')
        print("Done calculating kappa")

        return


def extract_burden_descriptors(dataframe, column, url):
    """
    Extracting molecular burden descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Burden
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting burden calculation")
        diction = []
        columns = ['bcutp8', 'bcutm9', 'bcutp9',
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
                   'bcutm11', 'bcutm10']
        i = 0
        for line in dataframe[column]:
            i += 1
            print("burden ", i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = bcut.GetBurden(mol)
            diction.append(dic)
        df_burden = pd.DataFrame(diction, columns=columns)
        df_burden.to_csv('../data/df_burden.csv')
        print("Done calculating burden")

        return


def extract_estate_descriptors(dataframe, column, url):
    """
    Extracting molecular E-state descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which E-State
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting estate calculation")
        diction = []
        columns = ['Smax38', 'Smax39', 'Smax34',
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
                   'Smin18', 'Smin19']
        i = 0
        for line in dataframe[column]:
            i += 1
            print("estate ", i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = estate.GetEstate(mol)
            diction.append(dic)
        df_estate = pd.DataFrame(diction, columns=columns)
        df_estate.to_csv('../data/df_estate.csv')
        print("Done calculating estate")

        return


def extract_basak_descriptors(dataframe, column, url):
    """
    Extracting molecular basak descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Basak
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting basak calculation")
        diction = []
        columns = ['CIC3', 'CIC6', 'SIC5', 'SIC4',
                   'SIC6', 'SIC1', 'SIC0', 'SIC3',
                   'SIC2', 'CIC5', 'CIC2', 'CIC0',
                   'CIC4', 'IC3', 'IC2', 'IC1',
                   'IC0', 'CIC1', 'IC6', 'IC5',
                   'IC4']
        i = 0
        for line in dataframe[column]:
            i += 1
            print("basak ", i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = basak.Getbasak(mol)
            diction.append(dic)
        df_basak = pd.DataFrame(diction, columns=columns)
        df_basak.to_csv('../data/df_basak.csv')
        print("Done calculating basak")

        return


def extract_moran_descriptors(dataframe, column, url):
    """
    Extracting molecular moran descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Moran
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting moran calculation")
        diction = []
        columns = ['MATSv8', 'MATSp4', 'MATSp8',
                   'MATSv1', 'MATSp6', 'MATSv3',
                   'MATSv2', 'MATSv5', 'MATSv4',
                   'MATSv7', 'MATSv6', 'MATSm8',
                   'MATSp1', 'MATSm4', 'MATSm5',
                   'MATSm6', 'MATSm7', 'MATSm1',
                   'MATSm2', 'MATSm3', 'MATSe4',
                   'MATSe5', 'MATSe6', 'MATSe7',
                   'MATSe1', 'MATSe2', 'MATSe3',
                   'MATSe8', 'MATSp3', 'MATSp7',
                   'MATSp5', 'MATSp2']
        i = 0
        for line in dataframe[column]:
            smiles = line
            i += 1
            print("moran ", i)
            mol = Chem.MolFromSmiles(smiles)
            dic = moran.GetMoranAuto(mol)
            diction.append(dic)
        df_moran = pd.DataFrame(diction, columns=columns)
        df_moran.to_csv('../data/df_moran.csv')
        print("Done calculating moran")

        return


def extract_geary_descriptors(dataframe, column, url):
    """
    Extracting molecular geary descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which Geary
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting geary calculation")
        diction = []
        columns = ['GATSp8', 'GATSv3', 'GATSv2',
                   'GATSv1', 'GATSp6', 'GATSv7',
                   'GATSv6', 'GATSv5', 'GATSv4',
                   'GATSe2', 'GATSe3', 'GATSv8',
                   'GATSe6', 'GATSe7', 'GATSe4',
                   'GATSe5', 'GATSp5', 'GATSp4',
                   'GATSp7', 'GATSe1', 'GATSp1',
                   'GATSp3', 'GATSp2', 'GATSe8',
                   'GATSm2', 'GATSm3', 'GATSm1',
                   'GATSm6', 'GATSm7', 'GATSm4',
                   'GATSm5', 'GATSm8']
        i = 0
        for line in dataframe[column]:
            i += 1
            print('geary ', i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = geary.GetGearyAuto(mol)
            diction.append(dic)
        df_geary = pd.DataFrame(diction, columns=columns)
        df_geary.to_csv('../data/df_geary.csv')
        print("Done calculating geary")

        return


def extract_property_descriptors(dataframe, column, url):
    """
    Extracting molecular property descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which property
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting property calculation")
        diction = []
        columns = ['TPSA', 'Hy', 'LogP',
                   'LogP2', 'UI', 'MR']
        i = 0
        for line in dataframe[column]:
            i += 1
            print('property ', i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = mp.GetMolecularProperty(mol)
            diction.append(dic)
        df_property = pd.DataFrame(diction, columns=columns)
        df_property.to_csv('../data/df_property.csv')
        print("Done calculating property")

        return


def extract_charge_descriptors(dataframe, column, url):
    """
    Extracting molecular charge descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which charge
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting charge calculation")
        diction = []
        columns = ['QNmin', 'QOss', 'Mpc',
                   'QHss', 'SPP', 'LDI', 'QCmin',
                   'Mac', 'Qass', 'QNss',
                   'QCmax', 'QOmax', 'Tpc',
                   'Qmax', 'QOmin', 'Tnc',
                   'QHmin', 'QCss', 'QHmax',
                   'QNmax', 'Rnc', 'Rpc', 'Qmin',
                   'Tac', 'Mnc']
        i = 0
        for line in dataframe[column]:
            i += 1
            print('charge ', i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = charge.GetCharge(mol)
            diction.append(dic)
        df_charge = pd.DataFrame(diction, columns=columns)
        df_charge.to_csv('../data/df_charge.csv')
        print("Done calculating charge")

        return


def extract_moe_descriptors(dataframe, column, url):
    """
    Extracting molecular MOE-type descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which MOE
    descriptors info must be evaluated
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting moe calculation")
        diction = []
        columns = ['EstateVSA8', 'EstateVSA9',
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
                   'VSAEstate9', 'VSAEstate10']
        i = 0
        for line in dataframe[column]:
            i += 1
            print('moe ', i)
            smiles = line
            mol = Chem.MolFromSmiles(smiles)
            dic = moe.GetMOE(mol)
            diction.append(dic)
        df_moe = pd.DataFrame(diction, columns=columns)
        df_moe.to_csv('../data/df_moe.csv')
        print("Done calculating moe")

        return


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


def extract_geometric_descriptors(dataframe, column, url):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    constitution descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        i = 0
        # File does not exist in URL
        print("Starting geometric calculation")
        diction = []
        columns = ['W3DH', 'W3D', 'Petitj3D', 'GeDi', 'grav1', 'rygr',
                   'Harary3D', 'AGDD', 'SEig', 'SPAN', 'ASPAN', 'MEcc']
        signal.signal(signal.SIGALRM, timeout_handler)
        for line in dataframe[column]:
            i += 1
            print('geometric ', i)
            smiles = line
            signal.alarm(30)
            try:
                mol = pyb.readstring('smi', smiles)
                GetARCFile(mol)
                dic = geometric.GetGeometric(mol)
            except OSError or TimeoutException:
                dic = np.empty((len(columns), 1,))
                dic[:] = np.NAN
            else:
                signal.alarm(0)
            finally:
                diction.append(dic)
        df_geometric = pd.DataFrame(diction, columns=columns)
        df_geometric.to_csv('../data/df_geometric.csv')
        print("Done calculating geometric")

        return


def extract_cpsa_descriptors(dataframe, column, url):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    constitution descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting cpsa calculation")
        diction = []
        columns = ['ASA', 'MSA', 'PNSA1',
                   'PPSA1', 'PNSA2', 'PPSA2',
                   'PNSA3', 'PPSA3', 'DPSA1',
                   'DPSA2', 'DPSA3', 'FNSA1',
                   'FNSA2', 'FNSA3', 'FPSA1',
                   'FPSA2', 'FPSA3', 'WNSA1',
                   'WNSA2', 'WNSA3', 'WPSA1',
                   'WPSA2', 'WPSA3', 'TASA',
                   'PSA', 'FrTATP', 'RASA',
                   'RPSA', 'RNCS', 'RPCS']
        i = 0
        signal.signal(signal.SIGALRM, timeout_handler)
        for line in dataframe[column]:
            i += 1
            print('cpsa ', i)
            smiles = line
            signal.alarm(30)
            try:
                mol = pyb.readstring('smi', smiles)
                GetARCFile(mol)
                dic = geometric.GetCPSA(mol)
            except OSError or TimeoutException:
                dic = np.empty((len(columns), 1,))
                dic[:] = np.NAN
            else:
                signal.alarm(0)
            finally:
                diction.append(dic)

        df_cpsa = pd.DataFrame(diction, columns=columns)
        df_cpsa.to_csv('../data/df_cpsa.csv')
        print("Done calculating cpsa")

        return


def extract_rdf_descriptors(dataframe, column, url):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    constitution descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting rdf calculation")
        diction = []
        columns = map(str, range(1, 210 + 1))
        signal.signal(signal.SIGALRM, timeout_handler)
        for i in range(len(columns)):
            columns[i] = 'rdf' + columns[i]
        i = 0
        for line in dataframe[column]:
            i += 1
            print('rdf ', i)
            smiles = line
            signal.alarm(30)
            try:
                mol = pyb.readstring('smi', smiles)
                GetARCFile(mol)
                dic = rdf.GetRDF(mol)
            except OSError or TimeoutException:
                dic = np.empty((len(columns), 1,))
                dic[:] = np.NAN
            else:
                signal.alarm(0)
            finally:
                diction.append(dic)
        df_rdf = pd.DataFrame(diction, columns=columns)
        df_rdf.to_csv('../data/df_rdf.csv')
        print("Done calculating rdf")

        return


def extract_morse_descriptors(dataframe, column, url):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    constitution descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting morse calculation")
        diction = []
        columns = map(str, range(1, 210 + 1))
        signal.signal(signal.SIGALRM, timeout_handler)
        for i in range(len(columns)):
            columns[i] = 'morse' + columns[i]
        i = 0
        for line in dataframe[column]:
            i += 1
            print('morse ', i)
            smiles = line
            signal.alarm(30)
            try:
                mol = pyb.readstring('smi', smiles)
                GetARCFile(mol)
                dic = morse.GetMoRSE(mol)
            except OSError or TimeoutException:
                dic = np.empty((len(columns), 1,))
                dic[:] = np.NAN
            else:
                signal.alarm(0)
            finally:
                diction.append(dic)
        df_morse = pd.DataFrame(diction, columns=columns)
        df_morse.to_csv('../data/df_morse.csv')
        print("Done calculating morse")

        return


def extract_whim_descriptors(dataframe, column, url):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
    constitution descriptors info must be evaluated.
    :param column: the column containing SMILES info for the compounds
     in the dataframe.
    :param url: URL to descriptor file in S3 bucket.
    :return: Descriptor dataframe.
    """
    try:
        # Check if file exists in this url
        r = urllib2.urlopen(url)
    except urllib2.URLError as e:
        r = e
    if r.code < 400:
        # File already exists in URL
        return
    else:
        # File does not exist in URL
        print("Starting whim calculation")
        diction = []
        columns = map(str, range(1, 70 + 1))
        signal.signal(signal.SIGALRM, timeout_handler)
        for i in range(len(columns)):
            columns[i] = 'whim' + columns[i]
        i = 0
        for line in dataframe[column]:
            i += 1
            print('whim ', i)
            smiles = line
            signal.alarm(30)
            try:
                mol = pyb.readstring('smi', smiles)
                GetARCFile(mol)
                dic = whim.GetWHIM()
            except OSError or TimeoutException:
                dic = np.empty((len(columns), 1,))
                dic[:] = np.NAN
            else:
                signal.alarm(0)
            finally:
                diction.append(dic)
        df_whim = pd.DataFrame(diction, columns=columns)
        df_whim.to_csv('../data/df_whim.csv')
        print("Done calculating whim")

        return
