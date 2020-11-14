import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

from DataGenerators import DataGeneratorReconstructor
from timer import Timer

from tenacity import retry, stop_after_attempt

@pd.api.extensions.register_dataframe_accessor("plotter")
class PlotterAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.huecountthreshold = 10

    @staticmethod
    def _validate(obj):
        pass

    def plot(self, hue: [str, None] = None, replacelabelwithhuename=True, verbose=False, *args, **kwargs):
        # plot this array's data on a map, e.g., using Cartopy

        if verbose:
            print('hue: {}'.format(hue))
        if hue is not None:
            df_value_counts = self._obj[hue].value_counts()
            list_names = list(df_value_counts.index)
            # list_counts = list(df_value_counts.values)
            numberofuniquevalues = len(list_names)

            if numberofuniquevalues > self.huecountthreshold:
                error_string = '''
                               The hue column that was passed has {} unique values. This is higher than the current huecountthreshold = {}. Excedding the threshold will likely result in a bad plot or performance. If you want to overwrite the threshold overwrite the property obj.huecountthreshold.
                               '''.format(numberofuniquevalues, self.huecountthreshold)

                raise ValueError(error_string)

            for huename in list_names:
                indexer = self._obj[hue] == huename
                if replacelabelwithhuename:
                    self._obj.loc[indexer, :].plot(*args, **kwargs, label=huename)
                else:
                    self._obj.loc[indexer, :].plot(*args, **kwargs)
        else:
            self._obj.plot(*args, **kwargs)

        if 'ax' in kwargs.keys() and 'y' in kwargs.keys():
            kwargs['ax'].set_ylabel(kwargs['y'])

# %%

figsize_1 = 10, 5
figsize_1on2 = 5, 5

# %%

path_project = os.getcwd().replace('\\', '/')
print('Project path: {}'.format(path_project))
path_data = path_project + '/data/'
path_datasetmetadata = path_data + 'datasetmetadata/'

# %%

df_datasetmetadata = pd.read_csv(path_datasetmetadata + 'df_datasetmetadatav2.csv')

# %%

for columnname in df_datasetmetadata.columns:
    # the true index won't exist if there aren't nulls.
    try:
        if df_datasetmetadata[columnname].isnull().value_counts()[True] > 0:
            print('columnname: {} has nans converting to None'.format(columnname))
            indexer = df_datasetmetadata[columnname].notnull()
            df_temp = df_datasetmetadata[columnname].copy()
            df_datasetmetadata[columnname] = None
            df_datasetmetadata.loc[indexer, columnname] = df_temp
    except KeyError:
        pass

# %%

dict_generators = dict()
for counter, index in enumerate(df_datasetmetadata.index):
    dict_datasetmetadata = df_datasetmetadata.iloc[index, :].to_dict()
    # converts a string representation of a list to an actual python list.
    dict_datasetmetadata['coefficients'] = np.array([[float(x)] for x in
                                                     dict_datasetmetadata['coefficients'].
                                                    replace('[', '').replace(']', '').replace(' ', '').split(',')])

    dict_generators['generatorv{}'.format(counter)] = DataGeneratorReconstructor(**dict_datasetmetadata)

# %%

list_numberofsamples = [10, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
list_numberofsamples = list(np.arange(5, 51, 1))
list_numberofsamples = list(np.arange(5, 51, 1)) + [int(x) for x in [1e2, 2e2, 4e2, 1e3]]
repetitions = 5
# list_numberofsamples = list(np.arange(5, 51, 10))
# repetitions = 1

# %%

array_alphas = 10**np.linspace(10,-2,100)*0.5
array_alphas

# %%

@retry(stop=stop_after_attempt(7))
def trylasso5times(array_alphas, cv=5, max_iter=1e6, normalize=True):
    lassocv = LassoCV(alphas=array_alphas, cv=5, max_iter=1e6, normalize=True)
    lassocv.fit(X_train, y_train)
    lasso = Lasso()
    lasso.set_params(alpha=lassocv.alpha_)
    lasso.fit(X_train, y_train)
    return lasso, lassocv

# %%

dict_results = dict()
for repetition in range(repetitions):
    for generatorname in dict_generators.keys():
        generator = dict_generators[generatorname]
        for numberofsamples in list_numberofsamples:
            datetime_modeling = datetime.datetime.now()
            experimentname = '_'.join([generatorname, 'n' + str(numberofsamples)])
            timer = Timer(taskname='Training Experiment {}'.format(experimentname))
            # print('generator name: {}'.format(generatorname))
            # print('number of samples: {}'.format(numberofsamples))
            df_train = generator.generatesamples(n_samples=numberofsamples,
                                                 random_state_generator=np.random.randint(0, 1e6, 1)[0])
            df_test = generator.generatesamples(n_samples=10000,
                                                random_state_generator=np.random.randint(0, 1e6, 1)[0])

            y_train = df_train['y']
            X_train = df_train.drop(columns=['y'])
            y_test = df_test['y']
            X_test = df_test.drop(columns=['y'])

            attempts = 0

            lasso, lassocv = trylasso5times(array_alphas=array_alphas)

            y_train_predict = lasso.predict(X_train)
            y_test_predict = lasso.predict(X_test)

            timer.stopAndPrint()
            dict_resultscurrentexperiment = dict()
            dict_resultscurrentexperiment['generatorname'] = generatorname
            dict_resultscurrentexperiment['numberofsamples'] = numberofsamples
            dict_resultscurrentexperiment['mse_train'] = mean_squared_error(y_train, y_train_predict)
            dict_resultscurrentexperiment['mse_test'] = mean_squared_error(y_test, y_test_predict)
            dict_resultscurrentexperiment['modelingduration'] = timer.duration_seconds
            dict_resultscurrentexperiment['datetimemodeling'] = datetime_modeling
            dict_results[experimentname] = dict_resultscurrentexperiment

df_results = pd.DataFrame(dict_results).T
df_results = df_results.set_index('datetimemodeling')

# %%

df_mse_path = pd.DataFrame(lassocv.mse_path_)
df_mse_path.index = lassocv.alphas_

# %%

df_mse_path.head()

# %%

fig, ax = plt.subplots()
df_mse_path.plot(ax=ax, marker='.')
ax.grid()
ax.set_xscale('log')
fig.show()

# %%

columnname_y = 'mse_test'
fig, ax = plt.subplots()
df_results.loc[:, columnname_y].plot(ax=ax)
ax.set_ylabel(columnname_y)
ax.grid()
fig.show()

# %%

columnname_y = 'modelingduration'
fig, ax = plt.subplots()
df_results.loc[:, columnname_y].plot(ax=ax)
ax.set_ylabel(columnname_y)
ax.grid()
fig.show()

# %%

columnname_y = 'modelingduration'
fig, ax = plt.subplots()
df_results.plot(x='numberofsamples', y=columnname_y, marker='o', ax=ax, linewidth=0, alpha=0.2)
ax.set_ylabel(columnname_y)
ax.grid()
ax.set_xscale('log')
fig.show()

# %%

# columnname_y = 'mse_test'
# fig, ax = plt.subplots()
# df_results.plot(x='numberofsamples', y=columnname_y, marker='o', ax=ax, linewidth=0, alpha=0.2, hue='generatorname')
# ax.set_ylabel(columnname_y)
# ax.grid()
# ax.set_xscale('log')
# fig.show()

# %%

fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1on2)
ax = axes[0]
sns.scatterplot(x='numberofsamples', y='mse_test', marker='o', ax=ax, data=df_results, hue='generatorname')
ax.grid()

ax = axes[1]
sns.scatterplot(x='numberofsamples', y='mse_train', marker='o', ax=ax, data=df_results, hue='generatorname')
ax.grid()

# ax.set_xscale('log')
fig.show()

# %%

fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize_1on2)

df_results.plotter.plot(x='mse_train', y='mse_test', marker='o', ax=ax, hue='generatorname')
ax.grid()

ax.set_yscale('log')
fig.show()

# %%

fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize_1on2)

df_results.plotter.plot(x='mse_train', y='mse_test', marker='o', ax=ax)
ax.grid()

ax.set_yscale('log')
fig.show()

# %%

fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize_1on2)

df_results.plotter.plot(x='mse_train', y='mse_test', marker='o', ax=ax, hue='generatorname', linewidth=0)
ax.grid()

ax.set_yscale('log')
fig.show()

# %%

fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1on2)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples', y='mse_test', marker='o', ax=ax, hue='generatorname')
ax.grid()

ax = axes[1]
df_results.plotter.plot(x='numberofsamples', y='mse_train', marker='o', ax=ax, hue='generatorname')
ax.grid()

# ax.set_xscale('log')
fig.show()

# %%


fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1, sharey=True)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples', y='mse_test', marker='o', ax=ax, hue='generatorname')
ax.grid()
ax.set_ylabel('mse_test')

ax = axes[1]
df_results.plotter.plot(x='numberofsamples', y='mse_train', marker='o', ax=ax, hue='generatorname')
ax.set_ylabel('mse_train')
ax.grid()

# ax.set_xscale('log')
ax.set_yscale('log')
fig.show()

# %%

fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1, sharey=True)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples', y='mse_test', marker='o', ax=ax, hue='generatorname')
ax.grid()
ax.set_ylabel('mse_test')

ax = axes[1]
df_results.plotter.plot(x='numberofsamples', y='mse_train', marker='o', ax=ax, hue='generatorname')
ax.set_ylabel('mse_train')
ax.grid()

ax.set_yscale('log')
ax.set_ylim(1e-3, 1e5)
fig.suptitle('Lasso Regression')
fig.show()

# %%

marker='.'
alpha=0.3
fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1, sharey=True)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples', y='mse_test', marker=marker, ax=ax, hue='generatorname', alpha=alpha)
ax.grid()
ax.set_ylabel('mse_test')

ax = axes[1]
df_results.plotter.plot(x='numberofsamples', y='mse_train', marker=marker, ax=ax, hue='generatorname', alpha=alpha)
ax.set_ylabel('mse_train')
ax.grid()

ax.set_yscale('log')
# ax.set_ylim(1e-5, 1e5)
fig.suptitle('Lasso Regression')
fig.show()

# %%

marker='.'
alpha=0.7
fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1, sharey=True)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples', y='mse_test', marker=marker, ax=ax, hue='generatorname', alpha=alpha)
ax.grid()
ax.set_ylabel('mse_test')

ax = axes[1]
df_results.plotter.plot(x='numberofsamples', y='mse_train', marker=marker, ax=ax, hue='generatorname', alpha=alpha)
ax.set_ylabel('mse_train')
ax.grid()

ax.set_yscale('log')
# ax.set_ylim(1e-5, 1e5)
ax.set_xlim(0, 75)
fig.suptitle('Lasso Regression')
fig.show()


# %%

marker='.'
alpha=0.3
fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1, sharey=True)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples', y='mse_test', marker=marker, ax=ax, hue='generatorname', alpha=alpha)
ax.grid()
ax.set_ylabel('mse_test')

ax = axes[1]
df_results.plotter.plot(x='numberofsamples', y='mse_train', marker=marker, ax=ax, hue='generatorname', alpha=alpha)
ax.set_ylabel('mse_train')
ax.grid()

ax.set_yscale('log')
ax.set_ylim(1e3, 1e4)
fig.suptitle('Lasso Regression')
fig.show()

# %%

df_results.head()

# %%
