import numpy as np
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from line_profiler import profile
from spin_sampler import Sampler, define_hopfield_model , initialize_spins, read_config


@profile
def main():
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,
                        help='Configuration file for the experiment')
    args = parser.parse_args()
    config_path = args.config

    # Read the configuration file
    conf = read_config(config_path)
    print(conf)

    # Define parameters
    mode = conf['mode']
    N = conf['N']
    N_walkers = conf['N_walkers']
    N_samples = conf['N_samples']
    dt_samples = conf['dt_samples']
    T = conf['T']
    seed = conf['seed']
    p = conf['p']

    # Run
    backend = 'numpy'
    print("Running with numpy backend")
    J , _ = define_hopfield_model(N,p,N_walkers if mode=='multi_couplings' else 1,mode=mode,backend=backend,seed=seed)
    S0 = initialize_spins(N,N_walkers,mode=mode,backend=backend,seed=seed)
    sampler = Sampler(J, T, mode = mode,backend=backend)
    sampler.run_gibbs(S0, N_samples=N_samples, dt_samples=dt_samples,seed=seed, store=True,progress=True);

    backend = 'numba'
    print("Running with numba backend")
    J , _ = define_hopfield_model(N,p,N_walkers if mode=='multi_couplings' else 1,mode=mode,backend=backend,seed=seed)
    S0 = initialize_spins(N,N_walkers,mode=mode,backend=backend,seed=seed)
    sampler = Sampler(J, T, mode = mode,backend=backend)
    sampler.run_gibbs(S0, N_samples=N_samples, dt_samples=dt_samples,seed=seed, store=True,progress=True);

    backend = 'jax'
    print("Running with jax backend")
    J , _ = define_hopfield_model(N,p,N_walkers if mode=='multi_couplings' else 1,mode=mode,backend=backend,seed=seed)
    S0 = initialize_spins(N,N_walkers,mode=mode,backend=backend,seed=seed)
    sampler = Sampler(J, T, mode = mode,backend=backend)
    sampler.run_gibbs(S0, N_samples=N_samples, dt_samples=dt_samples,seed=seed, store=True,progress=True);


if __name__ == "__main__":
    main()
