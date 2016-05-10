import pandas as pd

from utils import create_dict, create_dataframe

# To find the number of compounds tested
with open('chemical_notation_data/compounds_inchi.txt', 'r') as f:
    data = f.readlines()
    i = 1
    for line in data:
        words = line.split()
        i += 1
    print i 

# Expected: 389561


# The SMILES and InChI logs of the same compound have identical keys 
# Creating and joining the SMILES and InChI dictionaries along the same index

dict_compounds = {key: value + create_dict('chemical_notation_data/compounds_inchi.txt',
                                           'inchi')[key] for key, value in create_dict(
    'chemical_notation_data/compounds_smiles.txt', 'smiles').iteritems()}
dict_compounds_active = {key: value + create_dict('chemical_notation_data/compounds_'
                                                  'active_inchi.txt', 'inchi')[key] for
                         key, value in create_dict('chemical_notation_data/compounds_active'
                                                   '_smiles.txt', 'smiles').iteritems()}

# The SMILES and InChI logs of the same material have identical indices 
# Creating and joining the SMILES and InChI dataframes along the same index

df_compounds_smiles = create_dataframe('chemical_notation_data/compounds_smiles.txt', 'smiles')
df_compounds_inchi = create_dataframe('chemical_notation_data/compounds_inchi.txt', 'inchi')

df_compounds = pd.concat([df_compounds_smiles, df_compounds_inchi['INCHI']], axis=1).rename(columns={'ID': 'CID'})


# In[13]:

activity = pd.read_csv('activity_data/AID_743255_datatable.csv')
for i in range(5):
    activity = activity.drop(i, axis=0)
activity = activity.drop(['PUBCHEM_ACTIVITY_URL', 'PUBCHEM_RESULT_TAG', 'PUBCHEM_ACTIVITY_SCORE', 
                          'PUBCHEM_SID', 'PUBCHEM_ASSAYDATA_COMMENT', 'Potency', 'Efficacy',
                          'Analysis Comment', 'Curve_Description', 'Fit_LogAC50', 'Fit_HillSlope',
                          'Fit_R2', 'Fit_InfiniteActivity', 'Fit_ZeroActivity', 'Fit_CurveClass',
                          'Excluded_Points', 'Compound QC'], axis=1)
activity.reset_index(['PUBCHEM_RESULT_TAG'], drop=True) 
activity['dupes'] = activity.duplicated('PUBCHEM_CID')
activity = activity[activity['dupes'] == 0].drop(['dupes'], axis=1)
activity.to_csv('activity_data/cleaned_data.csv')
print activity
