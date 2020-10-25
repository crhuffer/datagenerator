import numpy as np

from DataGenerators import DataGeneratorReconstructor, DataGeneratorConstructor, DataGenerator

# %%

# This code should fail!!!!
# DataGenerator doesn't have a way to know what the coefficients are, those are built into the subclasses.
# Therefore we made DataGenerator abstract so it will fail if you try instantiate it.
seed = np.random.randint(0, 100, 1)[0]
datagenerator = DataGenerator()
print(seed, datagenerator.coefficients)

# %% We can use a constructor to create a new dataset generator (it will determine the coefficients randomly.

datagenerator = DataGeneratorConstructor(n_features=5,
                                         n_informative=3,
                                         random_state_initialization=42)

# %% Or we can use a reconstructor to build more random data using a set of coefficients we pass in.

seed = np.random.randint(0, 100, 1)[0]
datagenerator = DataGeneratorReconstructor(coefficients=[0.0, 0.0, 73.1, 72.1, 21.5])
print(seed, datagenerator.coefficients)

# %%

# %% In the next step we build a dataset with a constructor, we grab the coefficients and plug them into
# the reconstructor. Then we generate the same data to show that the two classes produce equivalent data.

datagenerator = DataGeneratorConstructor(n_features=5,
                                         n_informative=3,
                                         random_state_initialization=42)

# %%

print(datagenerator.coefficients)

# %%

coefficients = datagenerator.coefficients

# %%

seed = np.random.randint(0, 100, 1)[0]
datagenerator = DataGeneratorReconstructor(coefficients=coefficients)
print(seed, datagenerator.coefficients)

# %% Now we can experiment with different inputs.

datagenerator = DataGeneratorReconstructor(coefficients=coefficients, n_targets=1, bias=0.0,
                                           effective_rank=None, tail_strength=0.5, noise=0.0,
                                           random_state_initialization=42)
print(seed, datagenerator.coefficients)
datagenerator.generatesamples(n_samples=100, random_state_generator=42)

# %%

datagenerator = DataGeneratorConstructor(n_features=5, n_informative=3, n_targets=1, bias=0.0,
                                         effective_rank=None, tail_strength=0.5, noise=0.0,
                                         random_state_initialization=42)
print(seed, datagenerator.coefficients)
datagenerator.generatesamples(n_samples=100, random_state_generator=42)

# %%
