# Convenience functions
# Requires python 3.5

import os
import pickle
import string
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors.kde import KernelDensity
import pymc3 as pm


def write_pickle(obj, relnm):
    """ Serialize object to pickle and write to disk at relnm """
   
    with open(relnm, 'wb') as f:
        pickle.dump(obj, f, protocol=-1)
    return 'Serialized object to disk at {}'.format(relnm)


def read_pickle(relnm):
    """ Read serialized object from pickle ondisk at relnm """
   
    with open(relnm, 'rb') as f:
        obj = pickle.load(f)
        
    print('Loaded object from disk at {}'.format(relnm))
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


def strip_derived_rvs(rvs):
    '''Convenience fn: remove PyMC3-generated RVs from a list'''
    ret_rvs = []
    for rv in rvs:
        if not (re.search('_log',rv.name) or re.search('_interval',rv.name)):
            ret_rvs.append(rv)     
    return ret_rvs


def trace_median(x):
    return pd.Series(np.median(x,0), name='median')


def plot_traces_pymc(trcs, varnames=None):
    ''' Convenience fn: plot traces with overlaid means and values '''

    nrows = len(trcs.varnames)
    if varnames is not None:
        nrows = len(varnames)
        
    ax = pm.traceplot(trcs, varnames=varnames, figsize=(12,nrows*1.4)
        ,lines={k: v['mean'] for k, v in 
            pm.df_summary(trcs,varnames=varnames).iterrows()})

    for i, mn in enumerate(pm.df_summary(trcs, varnames=varnames)['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')    

        
def plot_stan_trc(dftrc):
    """
       Create simple plots of parameter distributions and traces from 
       output of pystan sampling. Emulates pymc traceplots.
    """

    fig, ax2d = plt.subplots(nrows=dftrc.shape[1], ncols=2, figsize=(14, 1.8*dftrc.shape[1]),
                                facecolor='0.99', edgecolor='k')
    fig.suptitle('Distributions and traceplots for {} samples'.format(
                                dftrc.shape[0]),fontsize=14)
    fig.subplots_adjust(wspace=0.2, hspace=0.5)

    k = 0
    
    # create density and traceplot, per parameter coeff
    for i, (ax1d, col) in enumerate(zip(ax2d, dftrc.columns)):

        samples = dftrc[col].values
        scale = (10**np.round(np.log10(samples.max() - samples.min()))) / 20
        kde = KernelDensity(bandwidth=scale).fit(samples.reshape(-1, 1))
        x = np.linspace(samples.min(), samples.max(), 100).reshape(-1, 1)
        y = np.exp(kde.score_samples(x))
        clr = sns.color_palette()[0]

        # density plot
        ax1d[0].plot(x, y, color=clr, linewidth=1.4)
        ax1d[0].vlines(np.percentile(samples, [2.5, 97.5]), ymin=0, ymax=y.max()*1.1,
                       alpha=1, linestyles='dotted', colors=clr, linewidth=1.2)
        mn = np.mean(samples)
        ax1d[0].vlines(mn, ymin=0, ymax=y.max()*1.1,
                       alpha=1, colors='r', linewidth=1.2)
        ax1d[0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')    
        ax1d[0].set_title('{}'.format(col), fontdict={'fontsize':10})


        # traceplot
        ax1d[1].plot(np.arange(len(samples)),samples, alpha=0.2, color=clr, linestyle='solid'
                              ,marker=',', markerfacecolor=clr, markersize=10)
        ax1d[1].hlines(np.percentile(samples,[2.5, 97.5]), xmin=0, xmax=len(samples),
                       alpha=1, linestyles='dotted', colors=clr)
        ax1d[1].hlines(np.mean(samples), xmin=0, xmax=len(samples), alpha=1, colors='r')

        k += 1
                
        ax1d[0].set_title('{}'.format(col), fontdict={'fontsize':14})#,'fontweight':'bold'})
        #ax1d[0].legend(loc='best', shadow=True)
        
        _ = [ax1d[j].axes.grid(True, linestyle='-', color='lightgrey') for j in range(2)]
            
    plt.subplots_adjust(top=0.94)
    plt.show()
    

