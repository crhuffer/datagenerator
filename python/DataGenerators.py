from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state


# %%


class DataGenerator(ABC):
    """
    An abstract class for building generators that can be repeatedly called to continue generating data from the same
    distribution.

    Built out of the logic of sklearn.datasets.make_regression
    """

    def __init__(self, n_features: int = 100, n_informative: int = 10, n_targets: int = 1,
                 bias: float = 0.0, effective_rank: [int, None] = None, tail_strength: float = 0.5, noise: float = 0.0,
                 random_state_initialization: int = 42):
        self.n_features = n_features
        self.n_targets = n_targets
        self.bias = bias
        self.effective_rank = effective_rank
        self.tail_strength = tail_strength
        self.noise = noise
        self.random_state_initialization = random_state_initialization

        self.n_informative = min(n_features, n_informative)

    @property
    @abstractmethod
    def coefficients(self):
        return NotImplementedError

    def generatesamples(self, n_samples=100, random_state_generator=None) -> pd.DataFrame:
        from sklearn.datasets._samples_generator import make_low_rank_matrix

        seed = check_random_state(random_state_generator)

        if self.effective_rank is None:
            # Randomly generate a well conditioned input set
            X = seed.randn(n_samples, self.n_features)

        else:
            # Randomly generate a low rank, fat tail input set
            X = make_low_rank_matrix(n_samples=n_samples,
                                     n_features=self.n_features,
                                     effective_rank=self.effective_rank,
                                     tail_strength=self.tail_strength,
                                     random_state=seed)

        y = np.dot(X, self.coefficients) + self.bias

        # Add noise
        if self.noise > 0.0:
            y += seed.normal(scale=self.noise, size=y.shape)

        y = np.squeeze(y)

        df = pd.DataFrame(X)
        df['y'] = y

        return df

    @property
    def datasetmetadata(cls):
        dict_arguments = cls.__dict__
        dict_datasetmetadata = dict_arguments
        dict_datasetmetadata['coefficients'] = str(list(datasetgenerator.coefficients[:, 0]))
        return dict_datasetmetadata


# %%


class DataGeneratorConstructor(DataGenerator):
    """This subclass will construct a repeatable, random set of coefficients to build a regression problem off of."""

    def __init__(self, *args, **kwargs):
        super(DataGeneratorConstructor, self).__init__(*args, **kwargs)

    @property
    def coefficients(self):
        seed = check_random_state(self.random_state_initialization)

        # Generate a ground truth model with only n_informative features being non
        # zeros (the other features are not correlated to y and should be ignored
        # by a sparsifying regularizers such as L1 or elastic net)
        coefficients = np.zeros((self.n_features, self.n_targets))
        coefficients[:self.n_informative, :] = 100 * seed.rand(self.n_informative, self.n_targets)

        # shuffle the column names so that the informative features aren't the last features.
        columnnames = np.arange(self.n_features)
        seed.shuffle(columnnames)
        coefficients = coefficients[columnnames]
        return coefficients


# %%

class DataGeneratorReconstructor(DataGenerator):
    """This subclass will reconstruct a regression problem off of an inputted set of coefficients."""

    def __init__(self, coefficients, *args, **kwargs):
        super(DataGeneratorReconstructor, self).__init__(*args, **kwargs)

        self.input_coefficients = coefficients

        self.n_features = len(self.coefficients)
        self.n_informative = None
        # TODO: build a calculation for self.n_informative.

    @property
    def coefficients(self):
        return self.input_coefficients


# %%

if __name__ == '__main__':
    import os

    datasetgenerator = DataGeneratorConstructor(n_features=20,
                                                n_informative=5,
                                                random_state_initialization=42)

    df1 = datasetgenerator.generatesamples(n_samples=10000)
    df_datasetmetadata = pd.DataFrame(datasetgenerator.datasetmetadata, index=[0])

    datasetgenerator = DataGeneratorConstructor(n_features=20,
                                                n_informative=5,
                                                effective_rank=10,
                                                random_state_initialization=42)

    df2 = datasetgenerator.generatesamples(n_samples=10000)
    df_datasetmetadata = df_datasetmetadata.append(datasetgenerator.datasetmetadata, ignore_index=True)

    datasetgenerator = DataGeneratorConstructor(n_features=20,
                                                n_informative=4,
                                                effective_rank=10,
                                                random_state_initialization=42)

    df3 = datasetgenerator.generatesamples(n_samples=10000)
    df_datasetmetadata = df_datasetmetadata.append(datasetgenerator.datasetmetadata, ignore_index=True)

    datasetgenerator = DataGeneratorConstructor(n_features=20,
                                                n_informative=4,
                                                effective_rank=10,
                                                random_state_initialization=42)

    df4 = datasetgenerator.generatesamples(n_samples=10000)
    df_datasetmetadata = df_datasetmetadata.append(datasetgenerator.datasetmetadata, ignore_index=True)

    datasetgenerator = DataGeneratorConstructor(n_features=20,
                                                n_informative=4,
                                                effective_rank=10,
                                                noise=0.5,
                                                random_state_initialization=42)

    df5 = datasetgenerator.generatesamples(n_samples=10000)
    df_datasetmetadata = df_datasetmetadata.append(datasetgenerator.datasetmetadata, ignore_index=True)

    path_project = os.getcwd()
    path_data = path_project + '/data/'
    path_rawdata = path_data + 'raw_data/'
    path_processeddata = path_data + 'processed_data/'
    path_datasetmetadata = path_data + 'datasetmetadata/'

    df_datasetmetadata.to_csv(path_datasetmetadata + 'df_datasetmetadatav2.csv')