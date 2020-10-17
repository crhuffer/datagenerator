
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt

import os

# %%

path_project = os.getcwd()
path_data = path_project + '/data/'
path_rawdata = path_data + 'raw_data/'
path_processeddata = path_data + 'processed_data/'

# %%

def regressiontodf(regression):
    X, y, coef = regression
    df = pd.DataFrame(X)
    df['y'] = y
    return X, y, coef, df

# %%

# def proxy_make_regression(n_samples=100, n_features=100, *, n_informative=10, n_targets=1, bias=0.0,
#                           effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
#                           random_state=None):
#
#     dict_arguments = {'n_samples': n_samples, 'n_features': n_features, 'n_informative': n_informative,
#                       'n_targets': n_targets, 'bias': bias, 'effective_rank': effective_rank,
#                       'tail_strength': tail_strength, 'noise': noise, 'shuffle': shuffle, 'coef': coef,
#                       'random_state': random_state}
#
#     make_regression(*)

# %%

X1, y1, coef1 = make_regression(n_samples=10000,
                                n_features=20,
                                n_informative=5,
                                coef=True,
                                random_state=42)

df1 = pd.DataFrame(X1); df1['y']=y1

# %%

X2, y2, coef2 = make_regression(n_samples=10000,
                                n_features=20,
                                n_informative=5,
                                effective_rank=10,
                                coef=True,
                                random_state=42)

df2 = pd.DataFrame(X2); df2['y'] = y2

# %%

X3, y3, coef3, df3 = regressiontodf(make_regression(n_samples=10000,
                                                    n_features=20,
                                                    n_informative=4,
                                                    effective_rank=10,
                                                    coef=True,
                                                    random_state=42))

# %%

X4, y4, coef4, df4 = regressiontodf(make_regression(n_samples=10000,
                                                    n_features=20,
                                                    n_informative=4,
                                                    effective_rank=10,
                                                    coef=True,
                                                    random_state=42))

# %%

X5, y5, coef5, df5 = regressiontodf(make_regression(n_samples=10000,
                                                    n_features=20,
                                                    n_informative=4,
                                                    effective_rank=10,
                                                    noise=0.5,
                                                    coef=True,
                                                    random_state=42))

# %%

coef1

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
