import numpy as np
import argparse
from spin_sampler.sampling import Sampler
from spin_sampler.utils import read_config
from line_profiler import profile
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax
from spin_sampler.spin_systems import *



plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def get_patterns(N_walkers, N, p , multi_patterns = True):
    """Generate random patterns of size N with p patterns."""
    # Generate random patterns
    if multi_patterns:
        patterns = np.random.choice([-1, 1], size=(N_walkers,N, p))
    else:
        patterns = np.random.choice([-1, 1], size=(N, p))
    
    if multi_patterns and N_walkers==1: patterns = patterns[0]

    return patterns


def weight_matrix(patterns):
    """Return the weight matrix."""
    # Initialize weights
    if len(patterns.shape) == 2:
        N , p = patterns.shape
        J = 1/N * (patterns @ patterns.T - p*np.eye(N))
    else:
        _ , N , p = patterns.shape
        J = 1/N * np.array( [pat @ pat.T - p*np.eye(N) for pat in patterns] )
    return J


# @profile

def test_gibbs_step_functionality():
    N = 10
    N_walkers = 3
    T = 2.0 
    for backend in ['numba']:
        key = jax.random.PRNGKey(42) if backend == 'jax' else 1
        for mode in ['single_chain', 'multi_chain', 'multi_couplings']:
            print(backend,mode)

            Nw = N_walkers if mode == 'multi_couplings' else 1
            J = define_SK_model(N, N_walkers=Nw, mode=mode, backend=backend, seed=42)
            sampler = Sampler(J, T, mode=mode, backend=backend)

            Nw = 1 if mode == 'single_chain' else N_walkers
            S_in = initialize_spins(N,N_walkers=Nw,mode=mode,backend=backend,seed=42)
            print(S_in.shape,J.shape)
            print(S_in.dtype,J.dtype)
            # S_out , key = sampler.gibbs_step(S_in.copy(), J, T,True,key)
            
            # # State should change
            # if backend == 'jax':
            #     assert jnp.array_equal(S_in,S_in)
            # else:
            #     assert np.array_equal(S_in,S_in) 

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
    seed = 1234
    p = 10
    # Define system and initial condition
    patterns = get_patterns(N_walkers,N,p)
    J = weight_matrix(patterns)


    S0 = np.random.choice([-1,1], size = (N_walkers,N))[0]
    patterns_jx = jnp.array(patterns)
    J_jx = jnp.array(J)
    S0_jx = jnp.array(S0)

    print(patterns.shape)
    backends = ['numpy','numba','jax']
    data = {}
    for backend in backends:
        print(f'Running with {backend}')
        if backend == 'jax':
            J = J_jx
            patterns = patterns_jx  
            S0 = S0_jx
        sampler = Sampler(J, T, mode = mode,backend=backend)
        sampler.run_gibbs(S0, N_samples=N_samples, dt_samples=dt_samples,seed=seed, store=True,progress=True);
        sampler.run_gibbs(S0, N_samples=N_samples, dt_samples=dt_samples,seed=seed, store=True,progress=True);
        S = sampler.get_chain()
        M = 1/N * S @ patterns
        data[backend] = M.T

    
    fig , axes = plt.subplots(nrows=len(backends),figsize=(6,4),sharex=True)

    colors = plt.get_cmap('YlGnBu')(np.linspace(0.1,1,p))
    for i , backend in enumerate(backends):
        ax = axes[i]
        if i == len(backends)-1: ax.set_xlabel(r'Gibbs steps')
        ax.set_ylabel(rf'{backend}')
        M = data[backend]
        for j in range(p): 
            ax.plot(M[j],color=colors[j],lw=1,alpha=0.75)
    fig.suptitle(r'Magnetization onto different patterns')
    fig.savefig('logs/backends.png',dpi=250,bbox_inches='tight')





if __name__ == "__main__":
    # main()
    test_gibbs_step_functionality()

