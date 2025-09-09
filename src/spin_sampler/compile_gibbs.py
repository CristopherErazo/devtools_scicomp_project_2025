from numba.pycc import CC
from numba import njit , types
import numpy as np
from spin_sampler.utils import prob_plus_numba

import warnings
from numba.core.errors import NumbaPerformanceWarning

# Suppress NumbaDebugInfoWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


# Create a compilation module
cc = CC('compiled_gibbs')  # This will generate `compiled_gibbs.so`

# Define the Numba-compiled single chain function




@cc.export('gibbs_step_single_chain_numba',
           types.Tuple((types.float64[:], types.int8))(
               types.float64[:], types.float64[:, :], types.float64, types.boolean, types.int8
           ))
@njit
def gibbs_step_single_chain(S, J, T, rnd_ord = True , key = None):
    """
    One full update of spin state using Gibbs sampling for 
    single chain with J.

    Parameters:
    ----------
    - S: Spin configuration shape (N,).
    - J: Coupling matrix (shape (N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: Dummy variable for compatibility with JAX functions.

    Returns:
    ----------
    - Updated spin configuration and dummy variable.
    """
    beta = 1 / T  
    N = len(S)

    # Define the update order
    idx = np.random.permutation(N) if rnd_ord else np.arange(N)

    # Update sequentially the spins
    for i in idx:
        h_i = S @ J[i]
        p_plus = prob_plus_numba(beta * h_i)
        rand_vals = np.random.rand()
        S[i] = 1 if rand_vals < p_plus else -1
    return S , key

# Define the Numba-compiled multi-chain function
@cc.export('gibbs_step_multi_chain_numba',
           types.Tuple((types.float64[:,:], types.int8))(
               types.float64[:,:], types.float64[:, :], types.float64, types.boolean, types.int8
           ))
@njit
def gibbs_step_multi_chain(S, J, T, rnd_ord = True,  key = None):
    """
    One full update of spin state using Gibbs sampling for 
    multiple chains with same J.

    Parameters:
    ----------
    - S: Spin configuration shape (N_walkers,N).
    - J: Coupling matrix (shape (N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: Dummy variable for compatibility with JAX functions.

    Returns:
    ----------
    - Updated spin configuration and dummy variable.
    """
    beta = 1 / T  
    N_walkers , N = S.shape

    # Define the update order
    idx = np.random.permutation(N) if rnd_ord else np.arange(N)

    # Update sequentially the spins
    for i in idx:
        h_i = S @ J[i]
        p_plus = prob_plus_numba(beta * h_i)
        rand_vals = np.random.rand(N_walkers)
        S[:, i] = np.where(rand_vals < p_plus, 1, -1)
    return S , key


# Define the Numba-compiled multi-couplings function
@cc.export('gibbs_step_multi_couplings_numba',
           types.Tuple((types.float64[:,:], types.int8))(
               types.float64[:,:], types.float64[:, :,:], types.float64, types.boolean, types.int8
           ))
@njit
def gibbs_step_multi_couplings(S, J, T, rnd_ord = True , key = None):
    """
    One full update of spin state using Gibbs sampling for 
    multiple chains with different J.

    Parameters:
    ----------
    - S: Spin configuration (shape (N_walkers, N)).
    - J: Coupling matrices (shape (N_walkers, N, N)).
    - T: Temperature.
    - rnd_ord: If True, update spins in random order.
    - key: Dummy variable for compatibility with JAX functions.

    Returns:
    ----------
    - Updated spin configuration and dummy variable.
    """
    beta = 1 / T  
    N_walkers, N = S.shape

    # Define the update order
    idx = np.random.permutation(N) if rnd_ord else np.arange(N)

    # Update sequentially the spins
    for i in idx:
        h_i = np.sum(J[:, i] * S, axis=1)
        p_plus = prob_plus_numba(beta * h_i)
        rand_vals = np.random.rand(N_walkers)
        S[:, i] = np.where(rand_vals < p_plus, 1, -1)
    return S , key


# Compile the module
if __name__ == '__main__':
    cc.compile()