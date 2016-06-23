import pandas as pd
import sklearn.feature_selection as f_selection
#from scikit-feature/skfeature/function/similarity_based import fisher_score
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
        dict = constitution.GetConstitutional(mol)
        diction.append(dict)
    df_constitution = pd.DataFrame(diction, columns=["nphos", "ndb", "nsb", "ncoi",
                                               "ncarb", "nsulph", "ncof",
                                               "nnitro","ncobr", "naro",
                                               "ndonr", "noxy", "nhet",
                                               "nhev", "nhal", "naccr",
                                               "nta", "ntb","nring", "nrot",
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
        dict = topology.GetTopology(mol)
        diction.append(dict)
    df_topology = pd.DataFrame(diction, columns=['GMTIV','AW', 'Geto', 'DZ',
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
        dict = con.GetConnectivity(mol)
        diction.append(dict)
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
        dict = kappa.GetKappa(mol)
        print dict
        diction.append(dict)
    df_kappa = pd.DataFrame(diction, columns=["nphos", "ndb", "nsb", "ncoi",
                                               "ncarb", "nsulph", "ncof",
                                               "nnitro","ncobr", "naro",
                                               "ndonr", "noxy", "nhet",
                                               "nhev", "nhal", "naccr",
                                               "nta", "ntb","nring", "nrot",
                                               "Weight", "PC2", "PC3", "PC1",
                                               "PC6", "PC4", "PC5", "AWeight",
                                               "ncocl", "nhyd"])
    print df_kappa


def extract_bcut_descriptors(dataframe, column):
    """
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        res = bcut.CalculateBurdenVDW(mol)
        res = bcut.CalculateBurdenPolarizability(mol)


def extract_electronic_state_descriptors(dataframe, column):
    """
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        res = estate.CalculateHeavyAtomEState(mol)
        res = estate.CalculateMaxEState(mol)
        res = estate.CalculateHalogenEState(mol)
        res = estate.GetEstate(mol)


def extract_basak_descriptors(dataframe, column):
    """
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        res = basak.CalculateBasakCIC1(mol)
        res = basak.CalculateBasakSIC2(mol)
        res = basak.CalculateBasakSIC3(mol)
        res = basak.Getbasak(mol)


def extract_moran_descriptors(dataframe, column):
    """
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        res = moran.CalculateMoranAutoVolume(mol)
        res = moran.GetMoranAuto(mol)

        res = geary.CalculateGearyAutoMass(mol)

        mp = mp.CalculateMolLogP(mol)
        mp = mp.CalculateMolMR(mol)
        mp = mp.CalculateTPSA(mol)
        mp = mp.GetMolecularProperty(mol)
        res = charge.CalculateLocalDipoleIndex(mol)
        res = charge.CalculateAllSumSquareCharge(mol)
        res = charge.GetCharge(mol)
        res = moe.CalculateTPSA(mol)
        res = moe.CalculatePEOEVSA(mol)
        res = moe.GetMOE(mol)
        print molweight


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
