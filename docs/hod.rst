AbacusHOD
=========

The AbacusHOD module loads halo catalogs from the AbacusSummit
simulations and outputs multi-tracer mock galaxy catalogs.
The code is highly efficient and contains a large set of HOD
extensions such as secondary biases (assembly biases),
velocity biases, and satellite profile flexibilities. The baseline
HODs are based on those from `Zheng et al. 2007 <https://arxiv.org/abs/astro-ph/0703457>`_
and `Alam et al. 2020 <http://arxiv.org/abs/1910.05095>`_.
The HOD extensions are first explained in `Yuan et al. 2018 <https://arxiv.org/abs/1802.10115>`_, and more
recently summarized in `Yuan et al. 2020b <https://arxiv.org/abs/2010.04182>`_ .
This HOD code also supports RSD and incompleteness. The code is fast,
completeling a :math:`(2Gpc/h)^3` volume in 80ms per tracer on a 32 core
desktop system, and the performance should be scalable. The module also
provides efficient correlation function and power spectrum calculators.
This module is particularly suited for efficiently sampling HOD parameter
space. We provide examples of docking it onto ``emcee`` and ``dynesty``
samplers.

The module defines one class, ``AbacusHOD``, whose constructor
takes the path to the simulation volume, and a set of HOD
parameters, and runs the ``staging`` function to compile the
simulation halo catalog as a set of arrays that are saved on
memory. The ``run_hod`` function can then be called to
generate galaxy catalogs.

The output takes the format of a dictionary of dictionaries,
where each subdictionary corresponds to a different tracer.
Currently, we have enabled tracer types: LRG, ELG, and QSO.
Each subdictionary contains all the mock galaxies of that
tracer type, recording their properties with keys ``x``, ``y``
, ``z``, ``vx``, ``vy``, ``vz``, ``mass``, ``id``, ``Ncent``.
The coordinates are in Mpc/h, and the velocities are in km/s.
The ``mass`` refers to host halo mass and is in units of Msun/h.
The ``id`` refers to halo id, and the ``Ncent`` key refers to number of
central galaxies for that tracer. The first ``Ncent`` galaxies
in the catalog are always centrals and the rest are satellites.

The galaxies can be written to disk by setting the
``write_to_disk`` flag to ``True`` in the argument of
``run_hod``. However, the I/O is slow and the ``write_to_disk``
flag defaults to ``False``.

The core of the AbacusHOD code is a two-pass memory-in-place algorithm.
The first pass of the halo+particle subsample computes the number
of galaxies generated in total. Then an empty array for these galaxies
is allocated in memory, which is then filled on the second pass of
the halos+particles. Each pass is accelerated with numba parallel.
The default threading is set to 16.


Theory
------
The baseline HOD for LRGs comes from Zheng et al. 2007:

.. math:: \bar{n}_{\mathrm{cent}}(M) = \frac{1}{2}\mathrm{erfc} \left[\frac{\ln(M_{\mathrm{cut}}/M)}{\sqrt{2}\sigma}\right],
.. math:: \bar{n}_{\textrm{sat}}(M) = \left[\frac{M-\kappa M_{\textrm{cut}}}{M_1}\right]^{\alpha}\bar{n}_{\mathrm{cent}}(M),

The baseline HOD for ELGs and QSOs
comes from Alam et al. 2020. The actual calculation
is complex and we refer the readers
to section 3 of `said paper <http://arxiv.org/abs/1910.05095>`_ for details.

In the baseline implementation, the central galaxy is assigned to the center
of mass of the halo, with the velocity vector also set to that of the center
of mass of the halo. Satellite galaxies are assigned to particles of the
halo with equal weights. When multiple tracers are enabled, each halo/particle
can only host a single tracer type. However, we have not yet implemented any
prescription of conformity.

The secondary bias (assembly bias) extensions follow the recipes described in
`Xu et al. 2020 <https://arxiv.org/abs/2007.05545>`_ , where the secondary halo
property (concentration or local overdensity) is directly tied to the mass
parameters in the baseline HOD (:math:`M_{\mathrm{cut}}` and :math:`M_1`):

.. math:: \log_{10} M_{\mathrm{cut}}^{\mathrm{mod}} = \log_{10} M_{\mathrm{cut}} + A_c(c^{\mathrm{rank}} - 0.5) + B_c(\delta^{\mathrm{rank}} - 0.5)
.. math:: \log_{10} M_{1}^{\mathrm{mod}} = \log_{10} M_{1} + A_s(c^{\mathrm{rank}} - 0.5) + B_s(\delta^{\mathrm{rank}} - 0.5)

where :math:`c` and :math:`\delta` represent the halo concentration and local
overdensity, respectively. These secondary properties are ranked within narrow
halo mass bins, and the rank are normalized to range from 0 to 1, as noted by
the :math:`\mathrm{rank}` superscript. :math:`(A_c, B_c, A_s, B_s)` form the
four parameters describing secondary biases in the HOD model. The default for
these parameters are 0.

The velocity bias extension follows the common prescription as described in
`Guo et al. 2015 <https://arxiv.org/abs/1407.4811>`_ .

.. math:: \sigma_c = \alpha_c \sigma_h
.. math:: v_s - v_h = \alpha_s (v_p - v_h)

where the central velocity bias parameter :math:`\alpha_c` sets the ratio of
central velocity dispersion vs. halo velocity dispersion. The satellite
velocity bias parameter :math:`\alpha_c` sets the ratio between the satellite
peculiar velocity to the particle peculiar velocity. The default for these two
parameters are 0 and 1, respectively.

We additionaly introduce a set of satellite profile parameters
:math:`(s, s_v, s_p, s_r)` that allow for flexibilities in how satellite
galaxies are distributed within a halo. They respecctively allow the galaxy
weight per particle to depend on radial position (:math:`s`), peculair velocity
(:math:`s_v`), perihelion distance of the particle orbit (:math:`s_p`), and
the radial velocity (:math:`s_v`). The default values for these parameters are
1. A detailed description of these parameters are available in
`Yuan et al. 2018 <https://arxiv.org/abs/1802.10115>`_, and more
recently in `Yuan et al. 2020b <https://arxiv.org/abs/2010.04182>`_ .


Some brief examples and technical details about the module
layout are presented below, followed by the full module API.


Short Example
-------------

The first step is to create the configuration file such as ``config/abacus_hod.yaml``,
which provides the full customizability of the HOD code. By default, it lives in your
current work directory under a subdirectory ``./config``. A template with
default settings are provided under ``abacusutils/scripts/hod/config``.

With the first use, you should define which simulation box, which redshift,
the path to simulation data, the path to output datasets, the various HOD
flags and an initial set of HOD parameters. Other decisions that need to be
made initially (you can always re-do this but it would take some time) include:
do you only want LRGs or do you want other tracers as well?
Do you want to enable satellite profile flexibilities (the :math:`s, s_v, s_p, s_r`
parameters)? If so, you need to turn on ``want_ranks`` flag in the config file.
If you want to enable secondary bias, you need to set ``want_AB`` flag to true in the
config file. The local environment is defined by total mass within 5 Mpc/h but beyond
``r98``.

.. tip::
   Running this code is a two-part process. First, you need to run the ``prepare_sim``
   code, which generates the necessary data files for that simulation. Then you can run the actual
   HOD code. The first step only needs to be done once for a simulation box, but it can be slow,
   depending on the downsampling and the features you choose to enable.

So first, you need to run the ``prepare_sim`` script, this extracts the simulation outputs
and organizes them into formats that are suited for the HOD code. This code can take
approximately an hour depending on your configuration settings and system capabilities.
We recommend setting the ``Nthread_load`` parameter to ``min(sys_core_count, memoryGB_divided_by_30)``.
You can run ``load_sims`` on command line with ::

    python -m abacusnbody.hod.prepare_sim --path2config PATH2CONFIG


Within Python, you can run the same script with ``from abacusnbody.hod import prepare_sim``
and then ``prepare_sim.main("/path/to/config.yaml")``.

If your config file lives in the default location, i.e. ``./config``, then you
can ignore the ``-path2config`` flag.
Once that is finished, you can construct the ``AbacusHOD`` object and run fast
HOD chains. A code template is given in ``abacusutils/scripts/run_hod.py`` for
running a few example HODs and ``abacusutils/scripts/run_emcee.py`` for integrating
with the ``emcee`` sampler.

To use the given ``run_hod.py`` script to run a custom configuration file, you can
simply run the given script in bash ::

    python run_hod.py --path2config PATH2CONFIG

You can also consruct the AbacusHOD object yourself within Python and run HODs from
there. Here we show the scripts within ``run_hod.py`` for reference.::

    import os
    import glob
    import time
    import yaml
    import numpy as np
    import argparse

    from abacusnbody.hod.abacus_hod import AbacusHOD

    path2config = 'config/abacus_hod.yaml' # path to config file

    # load the config file and parse in relevant parameters
    config = yaml.safe_load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']

    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']

    # create a new AbacusHOD object
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # first hod run, slow due to compiling jit, write to disk
    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = 16)

    # run the 10 different HODs for timing
    for i in range(10):
        newBall.tracers['LRG']['alpha'] += 0.01
        print("alpha = ",newBall.tracers['LRG']['alpha'])
        start = time.time()
        mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False, Nthread = 64)
        print("Done iteration ", i, "took time ", time.time() - start)

The class also provides fast 2PCF calculators. For example to compute the
redshift-space 2PCF (:math:`\xi(r_p, \pi)`): ::

    # load the rp pi binning from the config file
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']    # the pi binning is configrured by pi_max and bin size

    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk)
    xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size)

Light Cones
-----------
AbacusHOD supports generating HOD mock catalogs from halo light cone catalogs
(`PR #28 <https://github.com/abacusorg/abacusutils/pull/28>`_).  Details on the usage
will be provided here soon.

Notes
~~~~~
Currently, when RSD effects are enabled in the HOD code for the halo light cones, the
factor ``velz2kms``, determining the size of the RSD correction to the position along
the line of sight, is the same for all galaxies at a given redshift catalog.


Parallelism
-----------
Some of the HOD routines accept an ``Nthread`` argument to enable parallelism. This parallelism
is backed by `Numba <https://numba.readthedocs.io/en/stable/user/index.html>`_, for the most part.

This has implications if the HOD module is used as part of a larger code. Numba supports multiple
`threading backends <https://numba.readthedocs.io/en/stable/user/threading-layer.html#selecting-a-threading-layer-for-safe-parallel-execution>`_,
some of which will not work if the larger code uses other flavors of parallelism. In such cases,
you may need to instruct Numba to use a fork-safe and/or thread-safe backend (called ``forksafe``
and ``threadsafe``), depending on whether the larger code is forking subprocesses or spawning
threads. This can be accomplished by setting the environment variable ``NUMBA_THREADING_LAYER=forksafe``
or assigning ``numba.config.THREADING_LAYER='forksafe'`` in Python before importing the HOD
code (likewise for ``threadsafe``).  There is also a ``safe`` backend that is both fork- and thread-safe,
but it is only available if Intel TBB is available. See the `Numba threading layer docs <https://numba.readthedocs.io/en/stable/user/threading-layer.html>`_
for more info.


API
---

.. automodule:: abacusnbody.hod.abacus_hod
   :members:
   :undoc-members:
   :show-inheritance:
