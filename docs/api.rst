.. _api:

API Reference
=============

Below are the main functions and classes provided by the Spin Sampler library.

Sampler Class
-------------

.. autoclass:: spin_sampler.Sampler
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: spin_sampler.define_hopfield_model
.. autofunction:: spin_sampler.define_SK_model
.. autofunction:: spin_sampler.initialize_spins

.. _gibbs_steps:
 
Gibbs steps
-----------

.. autofunction:: spin_sampler.gibbs_steps.gibbs_step_single_chain
.. autofunction:: spin_sampler.gibbs_steps.gibbs_step_multi_chain
.. autofunction:: spin_sampler.gibbs_steps.gibbs_step_multi_couplings
.. autofunction:: spin_sampler.gibbs_steps.gibbs_step_single_chain_jax
.. autofunction:: spin_sampler.gibbs_steps.gibbs_step_multi_chain_jax
.. autofunction:: spin_sampler.gibbs_steps.gibbs_step_multi_couplings_jax
   