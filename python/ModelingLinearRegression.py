import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from tenacity import retry, stop_after_attempt

from DataGenerators import Adapter_DatasetMetaData, createdictofdatasetgenerators
from modelresultevaluators import evaluatemodelresultsv1
from timer import Timer

# %%

modeltype = LinearRegression.__name__


# %%

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

@retry(stop=stop_after_attempt(7))
def fitmodel(X_train, y_test):
    model.fit(X_train, y_train)
    return model


# %%

df_datasetmetadata = Adapter_DatasetMetaData().run(pd.read_csv(path_datasetmetadata + 'df_datasetmetadatav2.csv'))
dict_datasetgenerators = createdictofdatasetgenerators(df_datasetmetadata)

# %%

list_numberofsamples = [10, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000]
list_numberofsamples = list(np.arange(5, 51, 1)) + [int(x) for x in [1e2, 2e2, 4e2, 1e3]]
repetitions = 5
# list_numberofsamples = list(np.arange(5, 51, 10))
# repetitions = 1

# %%

numberofsamples_test = 10000
dict_results = dict()
for repetition in range(repetitions):
    for generatorname in dict_datasetgenerators.keys():
        generator = dict_datasetgenerators[generatorname]
        for numberofsamples_train in list_numberofsamples:
            datetime_modeling = datetime.datetime.now()
            experimentname = '_'.join([generatorname, 'n' + str(numberofsamples_train)])
            timer = Timer(taskname='Training Experiment {}'.format(experimentname))
            # print('generator name: {}'.format(generatorname))
            # print('number of samples: {}'.format(numberofsamples))

            seed_train = np.random.randint(0, 1e6, 1)[0]
            df_train = generator.generatesamples(n_samples=numberofsamples_train,
                                                 random_state_generator=seed_train)
            seed_test = np.random.randint(0, 1e6, 1)[0]
            df_test = generator.generatesamples(n_samples=numberofsamples_test,
                                                random_state_generator=seed_test)

            y_train = df_train['y']
            X_train = df_train.drop(columns=['y'])
            y_test = df_test['y']
            X_test = df_test.drop(columns=['y'])

            model = LinearRegression()

            timer_training = Timer('Training the model')
            model = fitmodel(X_train, y_train)
            timer_training.stopAndPrint()

            timer_scoring = Timer('Scoring the model')
            y_train_predict = model.predict(X_train)
            y_test_predict = model.predict(X_test)
            timer_scoring.stopAndPrint()

            timer.stopAndPrint()

            dict_resultscurrentexperiment = dict()
            dict_resultscurrentexperiment['generatorname'] = generatorname
            dict_resultscurrentexperiment['numberofsamples_train'] = numberofsamples_train
            dict_resultscurrentexperiment['numberofsamples_test'] = numberofsamples_test
            dict_resultscurrentexperiment['seed_train'] = seed_train
            dict_resultscurrentexperiment['seed_test'] = seed_test

            dict_resultscurrentexperiment['modeltype'] = modeltype

            dict_metrics_train = evaluatemodelresultsv1(y_train, y_train_predict, suffix='_train')
            dict_resultscurrentexperiment.update(dict_metrics_train)

            dict_metrics_test = evaluatemodelresultsv1(y_test, y_test_predict, suffix='_test')
            dict_resultscurrentexperiment.update(dict_metrics_test)

            dict_resultscurrentexperiment['trainingduration'] = timer_training.duration_seconds
            dict_resultscurrentexperiment['scoringduration'] = timer_scoring.duration_seconds
            dict_resultscurrentexperiment['datetimemodeling'] = datetime_modeling

            dict_results[experimentname] = dict_resultscurrentexperiment

df_results = pd.DataFrame(dict_results).T
df_results = df_results.set_index('datetimemodeling')

# %%

columnname_y = 'rmse_test'
fig, ax = plt.subplots()
df_results.loc[:, columnname_y].plot(ax=ax)
ax.set_ylabel(columnname_y)
ax.grid()
fig.show()

# %%

columnname_y = 'trainingduration'
fig, ax = plt.subplots()
df_results.loc[:, columnname_y].plot(ax=ax)
ax.set_ylabel(columnname_y)
ax.grid()
fig.show()

# %%

columnname_y = 'trainingduration'
fig, ax = plt.subplots()
df_results.plot(x='numberofsamples_train', y=columnname_y, marker='o', ax=ax, linewidth=0, alpha=0.2)
ax.set_ylabel(columnname_y)
ax.grid()
ax.set_xscale('log')
fig.show()

# %%

fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1on2)
ax = axes[0]
sns.scatterplot(x='numberofsamples_train', y='rmse_test', marker='o', ax=ax, data=df_results, hue='generatorname')
ax.grid()

ax = axes[1]
sns.scatterplot(x='numberofsamples_train', y='rmse_train', marker='o', ax=ax, data=df_results, hue='generatorname')
ax.grid()

# ax.set_xscale('log')
fig.show()

# %%

fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize_1on2)

sns.scatterplot(x='rmse_train', y='rmse_test', marker='o', ax=ax, data=df_results, hue='generatorname')
ax.grid()

ax.set_yscale('log')
fig.show()

# %%

fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize_1on2)

df_results.plotter.plot(x='rmse_train', y='rmse_test', marker='o', ax=ax)
ax.grid()

ax.set_yscale('log')
fig.show()

# %%

fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize_1on2)

df_results.plotter.plot(x='rmse_train', y='rmse_test', marker='o', ax=ax, hue='generatorname', linewidth=0)
ax.grid()

ax.set_yscale('log')
fig.show()

# %%

fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1on2)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples_train', y='rmse_test', marker='o', ax=ax, hue='generatorname')
ax.grid()

ax = axes[1]
df_results.plotter.plot(x='numberofsamples_train', y='rmse_train', marker='o', ax=ax, hue='generatorname')
ax.grid()

# ax.set_xscale('log')
fig.show()

# %%

fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1, sharey=True)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples_train', y='rmse_test', marker='o', ax=ax, hue='generatorname')
ax.grid()
ax.set_ylabel('mse_test')

ax = axes[1]
df_results.plotter.plot(x='numberofsamples_train', y='rmse_train', marker='o', ax=ax, hue='generatorname')
ax.set_ylabel('mse_train')
ax.grid()

# ax.set_xscale('log')
ax.set_yscale('log')
fig.show()

# %%

fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize_1, sharey=True)
ax = axes[0]
df_results.plotter.plot(x='numberofsamples_train', y='rmse_test', marker='o', ax=ax, hue='generatorname')
ax.grid()
ax.set_ylabel('mse_test')

ax = axes[1]
df_results.plotter.plot(x='numberofsamples_train', y='rmse_train', marker='o', ax=ax, hue='generatorname')
ax.set_ylabel('mse_train')
ax.grid()

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-3, 1e5)
fig.suptitle('Linear Regression')
fig.show()

# %%
