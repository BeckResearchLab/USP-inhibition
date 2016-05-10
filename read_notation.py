
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

# To find the number of compounds tested
with open('chemical_notation_data/compounds_inchi.txt', 'r') as f:
    data = f.readlines()
    i = 1
    for line in data:
        words = line.split()
        i += 1
    print i 

# Expected: 389561


# In[3]:

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
        for line in data[:100]:
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


# In[4]:

# The SMILES and InChI logs of the same compound have identical keys 
# Creating and joining the SMILES and InChI dictionaries along the same index

dict_compounds = {key: value + create_dict('chemical_notation_data/compounds_inchi.txt', 'inchi')[key] for key, value in 
                create_dict('chemical_notation_data/compounds_smiles.txt','smiles').iteritems()}
dict_compounds_active = {key: value + create_dict('chemical_notation_data/compounds_active_inchi.txt', 'inchi')[key] for key, value in 
                create_dict('chemical_notation_data/compounds_active_smiles.txt', 'smiles').iteritems()}


# In[5]:

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


# In[6]:

# The SMILES and InChI logs of the same material have identical indices 
# Creating and joining the SMILES and InChI dataframes along the same index

df_compounds_smiles = create_dataframe('chemical_notation_data/compounds_smiles.txt', 'smiles')
df_compounds_inchi = create_dataframe('chemical_notation_data/compounds_inchi.txt','inchi')

df_compounds = pd.concat([df_compounds_smiles, df_compounds_inchi['INCHI']], axis=1).rename(columns = {'ID':'CID'})


# In[7]:

activity = pd.read_csv('activity_data/AID_743255_datatable.csv')
for i in range(5):
    activity = activity.drop(i, axis=0)
activity = activity.drop(['PUBCHEM_ACTIVITY_URL', 'PUBCHEM_SID', 'PUBCHEM_ASSAYDATA_COMMENT', 
                          'Potency', 'Efficacy','Analysis Comment', 'Curve_Description',
                          'Fit_LogAC50','Fit_HillSlope','Fit_R2','Fit_InfiniteActivity',
                          'Fit_ZeroActivity','Fit_CurveClass', 'Excluded_Points', 'Compound QC'], axis=1)
activity.reset_index(['PUBCHEM_RESULT_TAG'], drop=True).to_csv('activity_data/cleaned_data.csv')


# In[8]:

activity.shape


# In[9]:

print df_compounds.shape


# In[ ]:



