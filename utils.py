import pandas as pd
import sklearn.feature_selection as f_selection
# from scikit-feature/skfeature/function/similarity_based import fisher_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from pychem import constitution, topology, connectivity as con, kappa
from pychem import bcut, estate, basak, moran, geary, molproperty as mp
from pychem import charge, moe, geometric, cpsa, rdf, morse, whim, fingerprint
from pychem.pychem import Chem
from sklearn.preprocessing import RobustScaler, LabelEncoder


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
                print 'Invalid molecular notation. Choose from smiles or inchi.'
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
                print 'Invalid molecular notation. Choose from smiles or inchi.'
            # Appending dictionary            
            df.append(z)
        df = pd.DataFrame(df)
        df.columns = ['ID', mol.upper()]
        return df


def select_features(x, y, num_fea):
    """

    :param x: dataset of features
    :param y: dataset of target property
    :param num_fea: desired number of features
    :return:
    """

    # Fisher's test
    f_score = fisher_score.fisher_score(x, y)
    idx = fisher_score.feature_ranking(f_score)
    x_fisher = x[:, idx[0:num_fea]]

    # Removing features with low variance
    var_threshold = f_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
    x_var_threshold = var_threshold.fit_transform(x)

    # Kbest-based feature selection using regression
    f_regress = f_selection.f_regression(x, y, center=False)
    # For classification: x_kbest = f_selection.SelectKBest(f_selection.chi2, k=2).fit_transform(x, y)
    x_kbest = f_selection.SelectKBest(score_func=f_regress, k=2).fit_transform(x, y)

    # Tree-based feature selection
    clf = ExtraTreesClassifier.fit(x, y)
    x_trees = f_selection.SelectFromModel(clf, prefit=True).transform(x)

    # Percentile-based feature selection using regression
    x_percentile = f_selection.SelectPercentile(score_func=f_regress,
                                                percentile=10).fit_transform(x, y)

    # "False positive rate"-based feature selection using regression
    x_percentile = f_selection.SelectFpr(score_func=f_regress,
                                         alpha=0.05).fit_transform(x, y)

    # This data set is way to high-dimensional. Better do PCA:
    pca = PCA(n_components=2)

    # Maybe some original features where good, too?
    kbest = f_selection.SelectKBest(score_func=f_regress, k=2)

    # Build estimator from PCA and Univariate selection:

    combined_features = FeatureUnion([("pca", pca), ("univ_kbest", kbest)])

    # Use combined features to transform dataset:
    x_features = combined_features.fit(x, y).transform(x)

    svm = SVC(kernel="linear")

    # Do grid search over k, n_components and C:

    pipeline = Pipeline([("features", x_features), ("svm", svm)])

    param_grid = dict(features__pca__n_components=[1, 2, 3],
                      features__univ_kbest__k=[1, 2],
                      svm__C=[0.1, 1, 10])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    grid_search.fit(x, y)
    print(grid_search.best_estimator_)

    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(x, y)

    return fisher_score, idx, x_fisher, x_var_threshold, x_kbest, x_trees, x_percentile, selector.support_


def extract_constitution_descriptors(dataframe, column):
    """
    Extracting molecular constitution descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      constitutional descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
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
    print df_constitution


def extract_topology_descriptors(dataframe, column):
    """
    Extracting molecular topology descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      topological descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    diction = []
    for line in dataframe[column][:]:
        smiles = line
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
    print df_topology


def extract_con_descriptors(dataframe, column):
    """
    Extracting molecular connectivity descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      connectivity descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
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
    print df_con


def extract_kappa_descriptors(dataframe, column):
    """
    Extracting molecular kappa descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      kappa descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    diction = []
    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        dic = kappa.GetKappa(mol)
        diction.append(dic)
    df_kappa = pd.DataFrame(diction, columns=['phi', 'kappa1', 'kappa3',
                                              'kappa2', 'kappam1', 'kappam3',
                                              'kappam2'])
    print df_kappa


def extract_burden_descriptors(dataframe, column):
    """
    Extracting molecular burden descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      burden descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    diction = []
    for line in dataframe[column][:]:
        smiles = line
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
    print df_burden


def extract_estate_descriptors(dataframe, column):
    """
    Extracting molecular E-state descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      E-state descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
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
    print df_estate


def extract_basak_descriptors(dataframe, column):
    """
    Extracting molecular basak descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      basak descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    diction = []
    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        dic = basak.Getbasak(mol)
        print dic
        diction.append(dic)
    df_basak = pd.DataFrame(diction, columns=['CIC3', 'CIC6', 'SIC5', 'SIC4',
                                              'SIC6', 'SIC1', 'SIC0', 'SIC3',
                                              'SIC2', 'CIC5', 'CIC2', 'CIC0',
                                              'CIC4', 'IC3', 'IC2', 'IC1',
                                              'IC0', 'CIC1', 'IC6', 'IC5',
                                              'IC4'])
    print df_basak


def extract_moran_descriptors(dataframe, column):
    """
    Extracting molecular moran descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      moran descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    diction = []
    for line in dataframe[column][:]:
        smiles = line
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
    print df_moran


def extract_geary_descriptors(dataframe, column):
    """
    Extracting molecular geary descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      geary descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    diction = []
    for line in dataframe[column][:]:
        smiles = line
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
    print df_geary


def extract_property_descriptors(dataframe, column):
    """
    Extracting molecular property descriptors using PyChem package and
    SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which
                      property descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in
                    the dataframe.
    :return: Descriptor dataframe
    """
    diction = []
    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        dic = mp.GetMolecularProperty(mol)
        diction.append(dic)
    df_property = pd.DataFrame(diction, columns=['TPSA', 'Hy', 'LogP',
                                                 'LogP2', 'UI', 'MR'])
    print df_property


def transform_dataframe(dataframe, target_column):

    """
    Function to read dataframe and standardize the dataframe with
    a mean 0 and unit variance on every column except target_column

    Parameters:
        dataframe : Input pandas dataframe
        target_column : Identity of the column in df with target data
    Input types: (pd.Dataframe, str)
    Output types: pd.Dataframe

    """
    cols = [col for col in dataframe.columns if col not in
            [target_column]]
    robust_scaler = RobustScaler()
    df = robust_scaler.fit_transform(dataframe[cols])
    dataframe.columns = df
    return dataframe
