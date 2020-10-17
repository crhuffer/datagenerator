import pandas as pd
import os
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# %%

path_project = os.getcwd()
path_data = path_project + '/data/'
path_rawdata = path_data + 'raw_data/'
path_processeddata = path_data + 'processed_data/'
path_datasetmetadata = path_data + 'datasetmetadata/'

# %%


def plotresults_rfecv(rfecv, figandax=None, name_data=None):

    if figandax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figandax

    plt.sca(ax=ax)
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Cross validation score (nb of correct classifications)")
    if name_data is None:
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, marker='.')
    else:
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, marker='.', label=name_data)
    ax.grid()
    ax.set_xlim(0, len(rfecv.grid_scores_) + 1)
    return fig, ax

# %%

list_dfnames = ['df1.csv', 'df2.csv', 'df3.csv', 'df4.csv', 'df5.csv']

list_dfs = [pd.read_csv(path_rawdata + dataframename, index_col=False) for dataframename in list_dfnames]

dict_dfs = {}
for index in range(len(list_dfnames)):
    dataframename = list_dfnames[index]
    df = list_dfs[index]
    dict_dfs[dataframename] = df

# %%
#
# df1 = pd.read_csv(path_rawdata + 'df1.csv'); list_dfs.append(df1)
# df2 = pd.read_csv(path_rawdata + 'df2.csv'); list_dfs.append(df1)
# df3 = pd.read_csv(path_rawdata + 'df3.csv'); list_dfs.append(df1)
# df4 = pd.read_csv(path_rawdata + 'df4.csv'); list_dfs.append(df1)
# df5 = pd.read_csv(path_rawdata + 'df5.csv'); list_dfs.append(df1)

# %%

linearmodel = LinearRegression()

rfecv = RFECV(linearmodel, verbose=1)

# %%
#
# df1 = list_dfs[0]
# X1 = df1.loc[:, df1.columns.drop('y')]
# y1 = df1.loc[:, 'y']
#
# # %%
#
# rfecv.fit(X1, y1)
#
# # %%
#
# plotresults_rfecv(rfecv)

# %%

fig, ax = plt.subplots()
for dfname in dict_dfs.keys():
    df = dict_dfs[dfname]
    X = df.loc[:, df.columns.drop('y')]
    y = df.loc[:, 'y']
    rfecv.fit(X, y)
    plotresults_rfecv(rfecv, figandax=(fig, ax), name_data=dfname)

ax.legend()

# %%

