from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt

import os

# %%

path_project = os.getcwd()
path_data = path_project + '/data/'
path_rawdata = path_data + 'raw_data/'
path_processeddata = path_data + 'processed_data/'
path_datasetmetadata = path_data + 'datasetmetadata/'


# %%

def proxy_make_regression(n_samples=100, n_features=100, *, n_informative=10, n_targets=1, bias=0.0,
                          effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                          random_state=None):
    """
    This proxy creates a wrapper over the makeregression function to enable saving of the input parameters along with
    the data that they generated.

    arguments should match the make_regression function inside sklearn.
    """

    dict_arguments = {'n_samples': n_samples, 'n_features': n_features, 'n_informative': n_informative,
                      'n_targets': n_targets, 'bias': bias, 'effective_rank': effective_rank,
                      'tail_strength': tail_strength, 'noise': noise, 'shuffle': shuffle, 'coef': coef,
                      'random_state': random_state}

    X, y, coef = make_regression(**dict_arguments)

    df = pd.DataFrame(X)
    df['y'] = y

    dict_datasetmetadata = dict_arguments
    dict_datasetmetadata['coef'] = str(list(coef))

    return df, dict_datasetmetadata


# %%

df1, dict_datasetmetadata1 = proxy_make_regression(n_samples=10000,
                                                   n_features=20,
                                                   n_informative=5,
                                                   coef=True,
                                                   random_state=42)

df_datasetmetadata = pd.DataFrame(dict_datasetmetadata1, index=[0])

# %%

df2, dict_datasetmetadata2 = proxy_make_regression(n_samples=10000,
                                                   n_features=20,
                                                   n_informative=5,
                                                   effective_rank=10,
                                                   coef=True,
                                                   random_state=42)

df_datasetmetadata = df_datasetmetadata.append(dict_datasetmetadata2, ignore_index=True)

# %%

df3, dict_datasetmetadata3 = proxy_make_regression(n_samples=10000,
                                                   n_features=20,
                                                   n_informative=4,
                                                   effective_rank=10,
                                                   coef=True,
                                                   random_state=42)

df_datasetmetadata = df_datasetmetadata.append(dict_datasetmetadata3, ignore_index=True)

# %%

df4, dict_datasetmetadata4 = proxy_make_regression(n_samples=10000,
                                                   n_features=20,
                                                   n_informative=4,
                                                   effective_rank=10,
                                                   coef=True,
                                                   random_state=42)

df_datasetmetadata = df_datasetmetadata.append(dict_datasetmetadata4, ignore_index=True)

# %%

df5, dict_datasetmetadata5 = proxy_make_regression(n_samples=10000,
                                                   n_features=20,
                                                   n_informative=4,
                                                   effective_rank=10,
                                                   noise=0.5,
                                                   coef=True,
                                                   random_state=42)

df_datasetmetadata = df_datasetmetadata.append(dict_datasetmetadata5, ignore_index=True)

# %%

df_datasetmetadata.to_csv(path_datasetmetadata + 'df_datasetmetadata.csv')

# %%

df1.head()

# %%

plt.imshow(df1.corr(), cmap='hot', interpolation='nearest')
plt.imshow(df2.corr(), cmap='hot', interpolation='nearest')
plt.imshow(df3.corr(), cmap='hot', interpolation='nearest')
plt.imshow(df4.corr(), cmap='hot', interpolation='nearest')
plt.imshow(df5.corr(), cmap='hot', interpolation='nearest')

# %%

df1.hist()
df2.hist()
df3.hist()
df4.hist()
df5.hist()

# %%

df1.to_csv(path_rawdata + 'df1')
df2.to_csv(path_rawdata + 'df2')
df3.to_csv(path_rawdata + 'df3')
df4.to_csv(path_rawdata + 'df4')
df5.to_csv(path_rawdata + 'df5')

# %%
