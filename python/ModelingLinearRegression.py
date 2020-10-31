import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from DataGenerators import DataGeneratorReconstructor
from timer import Timer

# %%

path_project = os.getcwd().replace('\\', '/')
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

list_numberofsamples = [10, 100, 1000, 10000]

# %%

dict_results = dict()
for generatorname in dict_generators.keys():
    generator = dict_generators[generatorname]
    for numberofsamples in list_numberofsamples:
        datetime_modeling = datetime.datetime.now()
        experimentname = '_'.join([generatorname, 'n' + str(numberofsamples)])
        timer = Timer(taskname='Training Experiment {}'.format(experimentname))
        print('generator name: {}'.format(generatorname))
        print('number of samples: {}'.format(numberofsamples))
        df = generator.generatesamples(n_samples=numberofsamples,
                                       random_state_generator=np.random.randint(0, 1e6, 1)[0])
        y = df['y']
        X = df.drop(columns=['y'])
        model = LinearRegression()
        model.fit(X, y)
        y_predict = model.predict(X)

        timer.stopAndPrint()
        dict_resultscurrentexperiment = dict()
        dict_resultscurrentexperiment['generatorname'] = generatorname
        dict_resultscurrentexperiment['numberofsamples'] = numberofsamples
        dict_resultscurrentexperiment['mse'] = mean_squared_error(y, y_predict)
        dict_resultscurrentexperiment['modelingduration'] = timer.duration_seconds
        dict_resultscurrentexperiment['datetimemodeling'] = datetime_modeling
        dict_results[experimentname] = dict_resultscurrentexperiment

# %%

df_results = pd.DataFrame(dict_results).T

# %%

df_results = df_results.set_index('datetimemodeling')

# %%

columnname_y = 'mse'
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
