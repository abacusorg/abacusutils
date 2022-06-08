Simulation Metadata
===================

The ``abacusnbody.metadata`` module contains the parameters and states
for Abacus simulations like AbacusSummit. One can use this module to
query information about simulations without actually downloading or
opening any simulation files.

The main entry point is the ``get_meta(simname, redshift=z)`` function.
Examples and the API are below.

Examples
--------

Omega(z)
~~~~~~~~
::

    import abacusnbody.metadata
    meta = abacusnbody.metadata.get_meta('AbacusSummit_base_c000_ph000', redshift=0.1)
    print(meta['OmegaNow_m'])  # Omega_M(z=0.1)
    print(meta['OmegaNow_DE'])  # Omega_DE(z=0.1)


Growth Factors in AbacusSummit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Growth factors in AbacusSummit are a bit of a special case, because the parameters
actually contain a pre-computed table of :math:`D(z)` for all the output epochs
and the ICs. This table is a ``dict`` called ``GrowthTable``.  Since AbacusSummit input
power spectra (i.e. CLASS power spectra) are generated at :math:`z=1`, one can
compute the linear power spectrum at a different epoch via the ratio :math:`D(z)/D(1)`:

::

    import abacusnbody.metadata
    meta = abacusnbody.metadata.get_meta('AbacusSummit_base_c000_ph000')
    Dz = meta['GrowthTable']
    ztarget = 0.1
    linear_pk = input_pk * (Dz[ztarget] / Dz[1.])**2


Developer Details
-----------------
The metadata is stored in an ASDF file in the ``abacusnbody/metadata``
directory. The metadata files are built with ``scripts/metadata/gather_metadata.py``,
and then compressed with ``scripts/metadata/compress.py`` (using compression on
the pickled representation). Internally, the time-independent parameters are separated
from the time-varying state values, but the two sets are combined into a single ``dict``
that is passed to the user.

API
---

.. automodule:: abacusnbody.metadata
   :members:
   :undoc-members:
   :show-inheritance:
