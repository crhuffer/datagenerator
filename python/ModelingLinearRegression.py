import datetime
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from DataGenerators import DataGeneratorReconstructor
from timer import Timer

# %%

path_project = os.getcwd().replace('\\', '/')
path_data = path_project + '/data/'
path_datasetmetadata = path_data + 'datasetmetadata/'

# %%

df_datasetmetadata = pd.read_csv(path_datasetmetadata + 'df_datasetmetadatav2.csv')

# %%

df_datasetmetadata.dtypes

# %%

# indexer = df_datasetmetadata['effective_rank'].isnull()
# df_datasetmetadata['effective_rank'].fillna(value='None', inplace=True)
# df_datasetmetadata.loc[indexer, 'effective_rank'] = None

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
    dict_datasetmetadata['coefficients'] = np.array([[float(x)] for x in
                                                     dict_datasetmetadata['coefficients'].replace('[', '').replace(']',
                                                                                                                   '').replace(
                                                         ' ', '').split(',')])

    dict_generators['generatorv{}'.format(counter)] = DataGeneratorReconstructor(**dict_datasetmetadata)

# %%

datasetgenerator = DataGeneratorReconstructor(**dict_datasetmetadata)
datasetgenerator.coefficients
datasetgenerator.datasetmetadata
df = datasetgenerator.generatesamples()

# %%

list_numberofsamples = [10, 100, 1000, 10000]

# %%

generatorname = 'generatorv0'
numberofsamples = 10

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
        timer.stopAndPrint()
        dict_resultscurrentexperiment = dict()
        dict_resultscurrentexperiment['generatorname'] = generatorname
        dict_resultscurrentexperiment['numberofsamples'] = numberofsamples
        dict_resultscurrentexperiment['mse'] = generatorname
        dict_resultscurrentexperiment['modelingduration'] = timer.duration_seconds
        dict_resultscurrentexperiment['datetime_modeling'] = datetime_modeling
        dict_results['experimentname'] = experimentname
