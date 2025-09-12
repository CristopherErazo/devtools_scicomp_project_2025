Usage
=====


The Spin Sampler library provides a general purpose ``Sampler`` class for Gibbs sampling of a spin system of :math:`N` spins.

To initialize the ``Sampler`` we need to define the physical system, which is fully characterized in our case by its coupling matrix :math:`\boldsymbol{J} = (J_{ij}) \in \mathbb{R}^{N\times N}`. The matrix has to be symmetric, otherwise a ``TypeError`` will be returned. The matrix can be customized by the user, nonetheless, the library provides two classical spin glass systems integrated that can be accesed with the functions ``spin_sampler.define_hopfield_model`` and  ``spin_sampler.define_SK_model``. See :ref:`api` for more information.

The ``Sampler`` admits 3 different ``modes`` that we can use:

- ``single_chain``: This mode runs a single Markov chain for Gibbs sampling. It is the default mode and is suitable for most use cases where a single chain is sufficient to explore the state space.

- ``multi_chains``: This mode runs multiple Markov chains in parallel where all of them share the same coupling matrix. It is useful to explore the state space with different initial conditions or to obtain fully statistically independent samples from the distribution.

- ``multi_couplings``: This mode runs multiple Markov chains in parallel where each chain has a different coupling matrix. It is useful when we want to study properties of an ensemble of systems when :math:`\boldsymbol{J}` comes from a distribution :math:`\mathbb{P}(\boldsymbol{J})`. 

Changing the mode also implies modifying the dimensions of the arrays used and this has to be checked beforehand. The variable ``N_walkers`` define the number of chains and the shapes of the arrays should be as follows:

+--------------------+--------------------+----------------------------+----------------------------+
| **Mode**           | ``single_chain``   | ``multi_chains``           | ``multi_couplings``        |
+====================+====================+============================+============================+
| **Shape of J**     | :math:`(N, N)`     | :math:`(N, N)`             | :math:`(N_{walkers}, N, N)`|
+--------------------+--------------------+----------------------------+----------------------------+
| **Shape of S**     | :math:`(N,)`       | :math:`(N_{walkers}, N)`   | :math:`(N_{walkers}, N)`   |
+--------------------+--------------------+----------------------------+----------------------------+



Example 1: Using `Sampler.sample`
---------------------------------

The `Sampler.sample` method is a generator that can be iterated to produce samples one at a time.
This is particularly useful for preliminary tests when we want to perform computations at each step to evaluate convergence of the algorithm or to analize physical properties over time.

.. code-block:: python

    from spin_sampler import Sampler , define_SK_model , initialize_spins
    import numpy as np

    # Define parameters
    N = 100     # Number of spins
    T = 1.0     # Temperature

    # The couplings
    J = define_SK_model(N)  

    # Initialize the sampler
    sampler = Sampler(J, T, mode="single_chain")

    # Generate samples
    initial_state = np.random.choice([-1, 1], size=N)
    for sample in sampler.sample(initial_state, N_samples=10):
        result = computation(sample)
        magnetiaztion = (1/N) * sample.sum()
        print(f'Results of computation with sample = {results}')
        print(f'The magnetization is = {magnetization}')

Example 2: Using `Sampler.run_gibbs`
------------------------------------

The `Sampler.run_gibbs` method executes the generator and returns the full chain of values.

.. code-block:: python

    # Run Gibbs sampling and get the full chain
    chain = sampler.run_gibbs(initial_state, N_samples=10)
    print(chain)