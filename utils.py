import pandas as pd
import sklearn.feature_selection as f_selection
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import ExtraTreesClassifier


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

    # Univariate feature selection
    f_regress = f_selection.f_regression(x, y, center=False)
    # For classification: x_kbest = f_selection.SelectKBest(f_selection.chi2, k=2).fit_transform(x, y)
    x_kbest = f_selection.SelectKBest(score_func=f_regress, k=2).fit_transform(x, y)

    # Tree-based feature selection
    clf = ExtraTreesClassifier.fit(x, y)
    clf.feature_importances_
    x_trees = f_selection.SelectFromModel(clf, prefit=True).transform(x)

    return fisher_score, idx, x_fisher, x_var_threshold, x_kbest, x_trees
