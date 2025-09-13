Usage
=====


The Spin Sampler library provides a general purpose :class:`Sampler <spin_sampler.Sampler>` class for Gibbs sampling of a spin system of :math:`N` spins.

To initialize the ``Sampler`` we need to define the physical system, which is fully characterized in our case by its coupling matrix :math:`\boldsymbol{J} = (J_{ij}) \in \mathbb{R}^{N\times N}`. The matrix has to be symmetric, otherwise a ``TypeError`` will be returned. The matrix can be customized by the user, nonetheless, the library provides two classical spin glass systems integrated that can be accesed with the functions :func:`define_hopfield_model <spin_sampler.define_hopfield_model>` and :func:`define_SK_model <spin_sampler.define_SK_model>`. See :ref:`api` for more information.

The :class:`Sampler <spin_sampler.Sampler>` admits 3 different ``modes`` that we can use:

- ``single_chain``: This mode runs a single Markov chain for Gibbs sampling. It is the default mode and is suitable for most use cases where a single chain is sufficient to explore the state space.

- ``multi_chains``: This mode runs multiple Markov chains in parallel where all of them share the same coupling matrix. It is useful to explore the state space with different initial conditions or to obtain fully statistically independent samples from the distribution.

- ``multi_couplings``: This mode runs multiple Markov chains in parallel where each chain has a different coupling matrix. It is useful when we want to study properties of an ensemble of systems when :math:`\boldsymbol{J}` comes from a distribution :math:`\mathbb{P}(\boldsymbol{J})`. 

Changing the mode also implies modifying the dimensions of the arrays used and this has to be checked beforehand. The variable ``N_walkers`` (default = 1) defines the number of chains and the shapes of the arrays should be as follows:

+--------------------+--------------------+----------------------------+----------------------------+
| **Mode**           | ``single_chain``   | ``multi_chains``           | ``multi_couplings``        |
+====================+====================+============================+============================+
| **Shape of J**     | :math:`(N, N)`     | :math:`(N, N)`             | :math:`(N_{walkers}, N, N)`|
+--------------------+--------------------+----------------------------+----------------------------+
| **Shape of S**     | :math:`(N,)`       | :math:`(N_{walkers}, N)`   | :math:`(N_{walkers}, N)`   |
+--------------------+--------------------+----------------------------+----------------------------+

The build-in functions satisfy the right shapes for each case. 


Example 1: Simple case
----------------------

The simplest usage of the package consist of extracting a set of samples from the Boltzmann distribution as follows:

.. code-block:: python

    from spin_sampler import Sampler , define_SK_model , initialize_spins
    import numpy as np

    # Define parameters of the system
    N = 100     # Number of spins
    T = 1.0     # Temperature

    # Define the couplings and initialize spins
    J = define_SK_model(N)
    initial_state = initialize_spins(N)

    # Create a sampler instance
    sampler = Sampler(J, T)
    # Perform sampling
    final_state = sampler.run_gibbs(initial_state,N_samples=10,dt_samples=1,store=True)
    # Get the samples
    chain = sampler.get_chain()
     # Clean up the sampler
    sampler.reset_chain()
    # chain.shape = (10,100) = (N_samples,N)


The :meth:`run_gibbs <spin_sampler.Sampler.run_gibbs>` method returns the last spin configuration as output and if ``storage = True``, the chain of values gets saved on the atribute ``sampler.chain`` as a list and can be accesed to with the :meth:`get_chain <spin_sampler.Sampler.get_chain>` method as an array.

The variable ``dt_samples`` controls how many gibbs steps we do before saving the next sample in the chain and can be used to reduce the time correlation between consecutive samples when needed.

Example 2: General Case
------------------------------------

The :class:`Sampler <spin_sampler.Sampler>` class allows for different backends to perform the main Gibbs update loop. The default option is ``numpy``, but it also includes a ``numba`` precompiled version of the numpy code and a ``jax`` implementation with a ``@jit`` compiled function that uses the optimized ``jax.lax.fori_loop()`` function (see :ref:`gibbs_steps`).

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from spin_sampler import Sampler , define_SK_model , initialize_spins

    # Define parameters of the system
    N = 100                 # Number of spins
    T = 1.0                 # Temperature
    N_walkers = 2           # Number of parallel chains

    # Define parameters of the sampler
    mode = 'multi_chain'    # 'single_chain', 'multi_chain' of 'multi_couplings'
    backend = 'jax'       # 'numpy', 'numba' or 'jax'
    seed = 42               # Random seed for reproducibility (mandatory for jax backend)
    N_samples = 10          # Number of samples to draw
    dt_samples = 5          # Number of Monte Carlo steps between samples (reduce time correlation)
    rnd_ord = True          # Randomize the order of spin updates (True recommended)
    store = True            # Store the samples 
    progress = True         # Show a progress bar (tqdm required)


    # Define the couplings and initialize spins
    J = define_SK_model(N,N_walkers,mode,backend,seed)
    initial_state = initialize_spins(N,N_walkers,mode,backend,seed)

    # Create a sampler instance
    sampler = Sampler(J, T,mode,backend)
    # Perform sampling
    sampler.run_gibbs(initial_state,N_samples,dt_samples,rnd_ord,seed,store,progress);
    # Get the samples
    chain = sampler.get_chain()
    # Clean up the sampler
    sampler.reset_chain()
    # chain.shape = (2,10,100) = (N_walkers,N_samples,N)


Example 3: Generator
--------------------

The :meth:`Sampler.sample <spin_sampler.Sampler.sample>` method can be used as a generator to perform Gibbs sampling and yield the spin state at each step. This allows you to perform computations with the current spin state at each step.

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from spin_sampler import Sampler, define_SK_model, initialize_spins

    # Define parameters
    N = 100                 # Number of spins
    T = 1.0                 # Temperature
    mode = 'single_chain'   # Sampling mode
    backend = 'numpy'       # Backend ('numpy', 'numba', or 'jax')
    seed = 42               # Random seed for reproducibility
    N_samples = 10          # Number of samples to draw
    dt_samples = 1          # Number of steps between samples
    rnd_ord = True          # Randomize the order of spin updates (True recommended)
    store = False           # Store the samples 
    progress = False        # Show a progress bar (tqdm required)


    # Define the coupling matrix and initialize spins
    J = define_SK_model(N, backend=backend, seed=seed)
    initial_state = initialize_spins(N, backend=backend, seed=seed)

    # Create a sampler instance and define the generator
    sampler = Sampler(J, T, mode, backend)
    generator = sampler.sample(initial_state, N_samples, dt_samples,rnd_ord, seed, store, progress)

    # Use the sample method as a generator
    for current_spin_state in generator:
        # Perform some computation with the current spin state
        magnetization = np.mean(current_spin_state)  # Example computation: magnetization
        print(f"Computation result: {magnetization}")
        
This setup gives a lot of flexibility to perform online computations, monitore the progress of the sampler or to dynamically save results in memory if needed instead of waiting until the end of the sampling as with :meth:`run_gibbs <spin_sampler.Sampler.run_gibbs>`.