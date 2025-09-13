import numpy as np
import time 
import pickle
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from line_profiler import profile
from spin_sampler import Sampler, define_SK_model , initialize_spins, read_config


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
    Ns = conf['Ns']
    backends = conf['backends']
    modes = conf['modes']
    N_walkers = conf['N_walkers']
    N_samples = conf['N_samples']
    dt_samples = conf['dt_samples']
    T = conf['T']
    N_iterations = conf['N_iterations']


    # Dictionary to store results
    data = conf.copy()

    for mode in modes:
        print(f"Running with {mode} mode")
        data[mode] = {}
        for backend in backends:
            print(f"Using {backend} backend")
            times = np.zeros(shape=(N_iterations,len(Ns)))
            for it in range(N_iterations):
                print(f"Iteration {it+1}/{N_iterations}")
                for iN , N in enumerate(Ns):
                    # print(f"N={N}")
                    seed = int(time.time())
                    J = define_SK_model(N,N_walkers if mode=='multi_couplings' else 1,mode=mode,backend=backend,seed=seed)
                    S0 = initialize_spins(N,1 if mode == 'single_chain' else N_walkers,mode=mode,backend=backend,seed=seed)
                    sampler = Sampler(J, T, mode = mode,backend=backend)
                    t0 = time.time()
                    sampler.run_gibbs(S0, N_samples=N_samples, dt_samples=dt_samples,seed=seed, store=False,progress=True);
                    t1 = time.time()
                    sampler.reset_chain()
                    elapsed = t1 - t0
                    times[it,iN] = elapsed

            data[mode][backend] = times

    print(data.keys())
    # Save the dictionary to a file
    with open('logs/data.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()