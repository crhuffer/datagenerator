
UML diagram of DataGenerators.py.

![DataGeneratorsUML](https://github.com/crhuffer/modelingongenerateddata/blob/master/images/DataGeneratorsUML.png)

Why use data generators?

Have you ever finished a round of experiments, run your data on the holdout dataset and gotten results you weren't happy with?  
Continuing to experiment on the dataset risks contaminating the holdout dataset and makes future conclusions less trustworthy.
In the earl world maybe you could collect additional dataset, but in kaggle and practice problems we don't have the ability to do that, and in the real world it is likely that this might be too expensive.

The goal of of the data generator is to store metadata about the dataset so that additional samples can be manufactured using the same parameters.
This enables some types of research that might be harder to do without a dataset generator for example can we quantify the impact of making decisions on the holdout dataset using a third holdout dataset?
 
 
Flow of the code:
DataGenerators.py: Creates the datasets and saves their metadata.
Modeling... .py: Loads the metadata and uses DataGeneratorReconstructors to build data sets and then builds models on those datasets. 