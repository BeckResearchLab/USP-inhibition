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
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        constitution = constitution.GetConstitutional(mol)
        print constitution


def extract_topology_descriptors(dataframe, column):
    """
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        balaban = topology.CalculateBalaban(mol)
        mzagreb1 = topology.CalculateMZagreb1(mol)
        harary = topology.CalculateHarary(mol)


def extract_connectivity_descriptors(dataframe, column):
    """
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        res = con.CalculateChi2(mol)
        res = con.CalculateMeanRandic(mol)
        res = con.GetConnectivity(mol)


def extract_kappa_descriptors(dataframe, column):
    """
    Extracting molecular descriptors using PyChem package and SMILES strings of compounds.
    :param dataframe: The dataframe containing SMILES info for which descriptors info must be evaluated.
    :param column:  The column containing SMILES info for the compounds in the dataframe.
    :return:
    """

    for line in dataframe[column][:]:
        smiles = line
        mol = Chem.MolFromSmiles(smiles)
        res = kappa.CalculateKappa1(mol)
        res = kappa.CalculateKappa2(mol)
        res = kappa.GetKappa(mol)


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