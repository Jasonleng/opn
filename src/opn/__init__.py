"""Shared code for the omission (OPN) analyses.

Modules
-------
paths        repository folders (networks, data, results)
networks     loading the trained LAN / omission networks
likelihoods  JAX log-likelihoods built from those networks
ops          wrapping the log-likelihoods as PyTensor Ops for PyMC
simulate     simulating, contaminating and splitting datasets at a deadline
cli          command-line options and result files shared by the simulation scripts
hierarchical hierarchical fits of any dataset (prepare_data, ModelSpec, fit)

Set ``pytensor.config.floatX = "float32"`` before building any Op: PyTensor's
JAX backend enables 64-bit JAX when floatX is float64, which breaks sampling
with these float32 networks.
"""

import os as _os


def set_cpu_devices(n):
    """Give JAX n CPU devices, so that n MCMC chains run in parallel.

    Call before JAX is imported (i.e. before importing pymc or the other opn modules);
    has no effect afterwards or if XLA_FLAGS is already set.
    """
    _os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={int(n)}")
