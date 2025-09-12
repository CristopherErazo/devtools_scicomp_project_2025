import numpy as np
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from line_profiler import profile
from spin_sampler import Sampler, define_hopfield_model , initialize_spins, read_config


plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


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

