import sys
sys.path.append("/home/pphilip/Tools/openbabel-install/lib")


import utils
import pandas as pd
import nn_model
import post_process  


# To find the number of compounds tested
with open('chemical_notation_data/compounds_inchi.txt', 'r') as f:
    data = f.readlines()
    i = 1
    for line in data:
        words = line.split()
        i += 1
    print i 

# Expected: 389561


def main():
    """
    Module to execute the entire neural network model from data retrieval to
    model performance metrics
    @:param: None
    :return: Post process results
    """
    # The SMILES and InChI logs of the same material have identical indices
    # Creating and joining the SMILES and InChI dataframes along the same index

    df_compounds_smiles = utils.create_dataframe('chemical_notation_data/'
                                           'compounds_smiles.txt', 'smiles')
    df_compounds_inchi = utils.create_dataframe('chemical_notation_data/'
                                          'compounds_inchi.txt', 'inchi')

    df_compounds = pd.concat([df_compounds_smiles, df_compounds_inchi['INCHI']],
                             axis=1).rename(columns={'ID': 'CID'})

    activity = pd.read_csv('activity_data/AID_743255_datatable.csv')
    for j in range(5):
        activity = activity.drop(j, axis=0)
    activity = activity.drop(['PUBCHEM_ACTIVITY_URL', 'PUBCHEM_RESULT_TAG',
                              'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_SID',
                              'PUBCHEM_ASSAYDATA_COMMENT', 'Potency',
                              'Efficacy', 'Analysis Comment',
                              'Curve_Description', 'Fit_LogAC50',
                              'Fit_HillSlope', 'Fit_R2', 'Fit_InfiniteActivity',
                              'Fit_ZeroActivity', 'Fit_CurveClass',
                              'Excluded_Points', 'Compound QC', 'Max_Response',
                              'Activity at 0.457 uM', 'Activity at 2.290 uM',
                              'Activity at 11.40 uM', 'Activity at 57.10 uM',
                              'PUBCHEM_ACTIVITY_OUTCOME'], axis=1)
    activity.rename(columns={'PUBCHEM_CID': 'CID'}, inplace=True)
    activity['dupes'] = activity.duplicated('CID')
    activity = activity[activity['dupes'] == 0].drop(['dupes'], axis=1)
    df_compounds = df_compounds.sort_values(by='CID')
    activity = activity.sort_values(by='CID')
    df = activity.merge(df_compounds)
    df = df.sort_values(by='CID')
    df.to_csv('activity_data/merged_data.csv')
    utils.extract_constitution_descriptors(df, 'SMILES')
    #nn_model.build_nn(df, 'class')
    #post_process.results()

if __name__ == "__main__":
    main()
