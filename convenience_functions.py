# Convenience functions
# Requires python 3.5

import os
import pickle
import string
import regex as re
import numpy as np
import pandas as pd



def write_pickle(obj, relnm):
    """ Serialize object to pickle and write to disk at relnm """
   
    with open(relnm, 'wb') as f:
        pickle.dump(obj, f, protocol=-1)
    return 'Serialized object to disk at {}'.format(relnm)


def read_pickle(relnm):
    """ Read serialized object from pickle ondisk at relnm """
   
    with open(relnm, 'rb') as f:
        obj = pickle.load(f)
    return obj



def ensure_dir(relnm):
    """ Accept relative filepath string, create it if it doesnt already exist
        return filepath string
    """
    
    d = os.path.dirname(relnm)
    if not os.path.exists(d):
        os.makedirs(d)
        
    return relnm


def snakey_lowercase(s):
    """ Clean and standardise a string to snakey lowercase
        Convert '-' to '_' and preserve existing '_'
        Useful for the often messy column names present in Excel tables
    """
    punct_to_remove = string.punctuation.replace('_', '')
    s1 = s.replace('-', '_')
    s2 = re.sub('[{}]'.format(re.escape(punct_to_remove)), '', s1)
    return '_'.join(s2.lower().split())


def custom_describe(df, nrows=3, nfeats=20):
    ''' Concat transposed topN rows, numerical desc & dtypes '''

    print(df.shape)
    rndidx = np.random.randint(0,len(df),nrows)
    dfdesc = df.describe().T

    for col in ['mean','std']:
        dfdesc[col] = dfdesc[col].apply(lambda x: np.round(x,2))
 
    dfout = pd.concat((df.iloc[rndidx].T, dfdesc, df.dtypes),axis=1, join='outer')
    dfout = dfout.loc[df.columns.values]
    dfout.rename(columns={0:'dtype'}, inplace=True)
    
    # add count nonNAN, min, max for string cols
    dfout['count'] = df.shape[0] - df.isnull().sum()
    dfout['min'] = df.min().apply(lambda x: x[:6] if type(x) == str else x)
    dfout['max'] = df.max().apply(lambda x: x[:6] if type(x) == str else x)
    
    return dfout.iloc[:nfeats,:]


