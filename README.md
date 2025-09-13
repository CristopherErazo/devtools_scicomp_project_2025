Name: Cristopher Ricardo Erazo Vallejos
email: cerazova@sissa.it
course: TSDS PhD.

# Spin Sampler Library

The **Spin Sampler** library is a Python package designed to study the thermodynamical properties of spin glasses. It provides tools to sample from the Boltzmann distribution of a spin system using Gibbs sampling. This library is particularly useful for researchers and students in statistical physics and computational science.

## Key Features

- **Sampling from the Boltzmann Distribution**: The library implements Gibbs sampling to explore the state space of spin systems.
- **Support for Multiple Modes**: Includes `single_chain`, `multi_chains`, and `multi_couplings` modes for different use cases.
- **Multiple Backends**: Supports `numpy`, `numba`, and `jax` backends for flexibility and performance optimization.
- **Profiling and Optimization**: Includes profiling tools to identify bottlenecks and optimize performance.

## Installation

To install the Spin Sampler library, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/CristopherErazo/devtools_scicomp_project_2025.git
    ```

2. Navigate to the root directory and install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the package:
    ```bash
    pip install .
    ```

For more details, refer to the [Installation Guide](docs/INSTALLATION.rst).

## Usage

The library provides a `Sampler` class to perform Gibbs sampling. Below is a simple example:

```python
from spin_sampler import Sampler, define_SK_model, initialize_spins

# Define parameters
N = 100
T = 1.0
J = define_SK_model(N)
initial_state = initialize_spins(N)

# Create a sampler instance
sampler = Sampler(J, T)
final_state = sampler.run_gibbs(initial_state, N_samples=10, dt_samples=1, store=True)
chain = sampler.get_chain()
sampler.reset_chain()
```

For detailed usage examples, refer to the [Usage Documentation](docs/USAGE.rst).

## Main Equations

The system is governed by the Hamiltonian:

$$
H(\boldsymbol{s}) = -\frac{1}{2} \sum_{i=1}^N J_{ij} s_i s_j
$$

The equilibrium configurations follow the Boltzmann distribution:

$$
\mu(\boldsymbol{s}) = \frac{e^{-\beta H(\boldsymbol{s})}}{Z_\beta}
$$

The Gibbs sampling algorithm updates spins using the conditional probability:

$$
P(s_i = +1 \mid \boldsymbol{s}_{\setminus i}) = \frac{1}{1 + \exp(-2\beta h_i)}
$$

where the local field is:

$$
h_i = \sum_{j \neq i} J_{ij} s_j
$$

## Profiling and Optimization

Using the `line_profiler`, we identified that the bottleneck of the algorithm is the single Gibbs step update. This is due to the sequential nature of the spin updates, which prohibits parallelization. Below is an excerpt from the profiling logs:

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
--------------------------------------------------------------
    200       100       500000   5000.0     95.0  S, key = self.gibbs_step(S, self.J, self.T, rnd_ord, key)
```

## Performance Results

The library supports three backends (`numpy`, `numba`, and `jax`) to optimize performance. The following image shows the timing results for different modes and backends as the system size increases:

![Timing Results](logs/timing.png)

The `jax` backend demonstrates significant performance improvements for larger systems due to its just-in-time (JIT) compilation.

## Documentation

For more information, refer to the full documentation:

- [Introduction](docs/INTRODUCTION.rst)
- [Usage](docs/USAGE.rst)
- [Installation](docs/INSTALLATION.rst)

## Conclusion

The Spin Sampler library is a powerful tool for studying spin systems. It combines flexibility, performance, and ease of use, making it suitable for both research and educational purposes. Profiling and optimization techniques have been applied to ensure efficient execution, and the results demonstrate the scalability of the library.
