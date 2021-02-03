"""
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
``run_hod``. However, the I/O is slow and the``write_to_disk`` 
flag defaults to ``False``.

The core of the AbacusHOD code is a two-pass memory-in-place algorithm.
The first pass of the halo+particle subsample computes the number
of galaxies generated in total. Then an empty array for these galaxies 
is allocated in memory, which is then filled on the second pass of 
the halos+particles. Each pass is accelerated with numba parallel.
The default threading is set to 16. 


Theory
======
The baseline HOD for LRGs comes from Zheng et al. 2007:

.. math:: \\bar{n}_{\\mathrm{cent}}(M) = \\frac{1}{2}\\mathrm{erfc} \\left[\\frac{\\ln(M_{\\mathrm{cut}}/M)}{\\sqrt{2}\\sigma}\\right],
.. math:: \\bar{n}_{\\textrm{sat}}(M) = \\left[\\frac{M-\\kappa M_{\\textrm{cut}}}{M_1}\\right]^{\\alpha}\\bar{n}_{\\mathrm{cent}}(M),

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
parameters in the baseline HOD (:math:`M_{\\mathrm{cut}}` and :math:`M_1`):

.. math:: \\log_{10} M_{\\mathrm{cut}}^{\\mathrm{mod}} = \\log_{10} M_{\\mathrm{cut}} + A_c(c^{\\mathrm{rank}} - 0.5) + B_c(\\delta^{\\mathrm{rank}} - 0.5)
.. math:: \\log_{10} M_{1}^{\\mathrm{mod}} = \\log_{10} M_{1} + A_s(c^{\\mathrm{rank}} - 0.5) + B_s(\\delta^{\\mathrm{rank}} - 0.5)

where :math:`c` and :math:`\\delta` represent the halo concentration and local 
overdensity, respectively. These secondary properties are ranked within narrow
halo mass bins, and the rank are normalized to range from 0 to 1, as noted by 
the :math:`\\mathrm{rank}` superscript. :math:`(A_c, B_c, A_s, B_s)` form the 
four parameters describing secondary biases in the HOD model. The default for
these parameters are 0. 

The velocity bias extension follows the common prescription as described in 
`Guo et al. 2015 <https://arxiv.org/abs/1407.4811>`_ . 

.. math:: \\sigma_c = \\alpha_c \\sigma_h
.. math:: v_s - v_h = \\alpha_s (v_p - v_h)

where the central velocity bias parameter :math:`\\alpha_c` sets the ratio of
central velocity dispersion vs. halo velocity dispersion. The satellite 
velocity bias parameter :math:`\\alpha_c` sets the ratio between the satellite
peculiar velocity to the particle peculiar velocity. The default for these two
parameters are 1. 

We additionaly introduce a set of satellite profile parameters 
:math:`(s, s_v, s_p, s_r)` that allow for flexibilities in how satellite 
galaxies are distributed within a halo. They respecctively allow the galaxy
weight per particle to depend on radial position (:math:`s`), peculair velocity
(:math:`s_v`), perihelion distance of the particle orbit (:math:`s_p`), and
the radial velocity (:math:`s_v`). The default values for these parameters are
0. A detailed description of these parameters are available in 
`Yuan et al. 2018 <https://arxiv.org/abs/1802.10115>`_, and more 
recently in `Yuan et al. 2020b <https://arxiv.org/abs/2010.04182>`_ . 


Some brief examples and technical details about the module
layout are presented below, followed by the full module API.


Short Example
=============

The first step is to create the configuration file such as ``config/abacus_hod.yaml``,
which provides the full customizability of the HOD code. By default, it lives in your 
current work directory under a subdirectory ``./config``. A template with 
default settings are provided under ``abacusutils/scripts/config``.

With the first use, you should define which simulation box, which redshift,
the path to simulation data, the path to output datasets, the various HOD 
flags and an initial set of HOD parameters. Other decisions that need to be 
made initially (you can always re-do this but it would take some time) include:
do you only want LRGs or do you want other tracers as well? 
Do you want to enable satellite profile flexibilities (the :math:`s, s_v, s_p, s_r`
parameters)? If so, you need to turn on ``want_ranks`` flag in the config file. 
If you want to enable secondary bias based on local environment, what scale 
radius do you want the environment do be defined in, this is set by the 
``density_sigma`` flag in Mpc/h. The default value is 3. Related, the ``Ndim``
parameter sets the grid size used to compute local density, and it should be set
to be larger than Lbox/sigma_density. 

Now you need to run the ``prepare_sim`` script, this extracts the simulation outputs
and organizes them into formats that are suited for the HOD code. This code can take 
approximately an hour depending on your configuration settings and system capabilities. 
We recommend setting the ``Nthread_load`` parameter to ``min(sys_core_count, memoryGB_divided_by_20)``.
You can run ``load_sims`` on command line with ::
    python -m abacusnbody.hod.prepare_sim --path2config PATH2CONFIG

Within Python, you can run the same script with

>>> from abacusnbody.hod import prepare_sim
>>> prepare_sim.main(/path/to/config.yaml)

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
there. Here we show the scripts within ``run_hod.py`` for reference.

>>> import os
>>> import glob
>>> import time
>>> 
>>> import yaml
>>> import numpy as np
>>> import argparse
>>> 
>>> from abacusnbody.hod.abacus_hod import AbacusHOD
>>> 
>>> path2config = 'config/abacus_hod.yaml' # path to config file
>>> 
>>> # load the config file and parse in relevant parameters
>>> config = yaml.load(open(path2config))
>>> sim_params = config['sim_params']
>>> HOD_params = config['HOD_params']
>>> clustering_params = config['clustering_params']
>>> 
>>> # additional parameter choices
>>> want_rsd = HOD_params['want_rsd']
>>> write_to_disk = HOD_params['write_to_disk']
>>> 
>>> # create a new AbacusHOD object
>>> newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
>>>     
>>> # first hod run, slow due to compiling jit, write to disk
>>> mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = 16)
>>> 
>>> # run the 10 different HODs for timing
>>> for i in range(10):
>>>     newBall.tracers['LRG']['alpha'] += 0.01
>>>     print("alpha = ",newBall.tracers['LRG']['alpha'])
>>>     start = time.time()
>>>     mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False, Nthread = 64)
>>>     print("Done iteration ", i, "took time ", time.time() - start)

The class also provides fast 2PCF calculators. For example to compute the 
redshift-space 2PCF (:math:`\\xi(r_p, \\pi)`):

>>> # load the rp pi binning from the config file
>>> bin_params = clustering_params['bin_params']
>>> rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
>>> pimax = clustering_params['pimax']
>>> pi_bin_size = clustering_params['pi_bin_size']    # the pi binning is configrured by pi_max and bin size
>>> 
>>> mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk)
>>> xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size)

"""
import os
import glob
import time
import timeit
from pathlib import Path

import numpy as np
import h5py
import asdf
import argparse
import multiprocessing
from multiprocessing import Pool
from astropy.io import ascii


from .GRAND_HOD import gen_gal_cat
from .tpcf_corrfunc import calc_xirppi_fast, calc_wp_fast
# TODO B.H.: staging can be shorter and prettier; perhaps asdf for h5 and ecsv?

class AbacusHOD:
    """
    A highly efficient multi-tracer HOD code for the AbacusSummmit simulations.
    """
    def __init__(self, sim_params, HOD_params, clustering_params):
        """
        Loads simulation. The ``sim_params`` dictionary specifies which simulation
        volume to load. The ``HOD_params`` specifies the HOD parameters and tracer
        configurations. The ``clustering_params`` specifies the summary statistics 
        configurations. The ``HOD_params`` and ``clustering_params`` can be set to their
        default values in the ``config/abacus_hod.yaml`` file and changed later. 
        The ``sim_params`` cannot be changed once the ``AbacusHOD`` object is created. 

        Parameters
        ----------
        sim_params: dict
            Dictionary of simulation parameters. Load from ``config/abacus_hod.yaml``. The dictionary should contain the following keys:
                * ``sim_name``: str, name of the simulation volume, e.g. 'AbacusSummit_base_c000_ph006'. 
                * ``sim_dir``: str, the directory that the simulation lives in, e.g. '/path/to/AbacusSummit/'.                                 
                * ``scratch_dir``: str, the diretory to save galaxy to, e.g. '/my/output/galalxies'. 
                * ``subsample_dir``: str, where to save halo+particle subsample, e.g. '/my/output/subsamples/'. 
                * ``z_mock``: float, which redshift slice, e.g. 0.5.    
                * ``Nthread_load``: int, how many threads to use to load the simulation data, default 7. 

        HOD_params: dict 
            HOD parameters and tracer configurations. Load from ``config/abacus_hod.yaml``. It contains the following keys:
                * ``tracer_flags``: dict, which tracers is enabled: 
                    * ``LRG``: bool, default ``True``. 
                    * ``ELG``: bool, default ``False``. 
                    * ``QSO``: bool, default ``False``. 
                * ``want_ranks``: bool, enable satellite profile flexibilities. If ``False``, satellite profile follows the DM, default ``True``. 
                * ``want_rsd``: bool, enable RSD? default ``True``.                 # want RSD? 
                * ``Ndim``: int, grid density for computing local environment, default 1024.
                * ``density_sigma``: float, scale radius in Mpc / h for local density definition, default 3.
                * ``write_to_disk``: bool, output to disk? default ``False``. Setting to ``True`` decreases performance. 
                * ``LRG_params``: dict, HOD parameter values for LRGs. Default values are given in config file. 
                * ``ELG_params``: dict, HOD parameter values for ELGs. Default values are given in config file. 
                * ``QSO_params``: dict, HOD parameter values for QSOs. Default values are given in config file. 

        clustering_params: dict
            Sumamry statistics configuration parameters. Load from ``config/abacus_hod.yaml``. It contains the following keys:
                * ``clustering_type``: str, which summary statistic to compute. Options: ``wp``, ``xirppi``, default: ``xirppi``.
                * ``bin_params``: dict, transverse scale binning. 
                    * ``logmin``: float, :math:`\\log_{10}r_{\\mathrm{min}} in Mpc/h.
                    * ``logmax``: float, :math:`\\log_{10}r_{\\mathrm{max}} in Mpc/h.
                    * ``nbins``: int, number of bins.
                * ``pimax``: int, :math:`\\pi_{\\mathrm{max}}`. 
                * ``pi_bin_size``: int, size of bins along of the line of sight. Need to be divisor of ``pimax``.

        """
        # simulation details
        self.sim_name = sim_params['sim_name']
        self.sim_dir = sim_params['sim_dir']
        self.subsample_dir = sim_params['subsample_dir']
        self.z_mock = sim_params['z_mock']
        self.scratch_dir = sim_params['scratch_dir']
        
        # tracers
        tracer_flags = HOD_params['tracer_flags']
        tracers = {}
        for key in tracer_flags.keys():
            if tracer_flags[key]:
                tracers[key] = HOD_params[key+'_params']
        self.tracers = tracers

        # HOD parameter choices
        self.want_ranks = HOD_params['want_ranks']
        self.want_rsd = HOD_params['want_rsd']
        
        # clusteringparameters
        self.pimax = clustering_params['pimax']
        self.pi_bin_size = clustering_params['pi_bin_size']
        bin_params = clustering_params['bin_params']
        self.rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
        self.clustering_type = clustering_params['clustering_type']
                
        # load the subsample particles
        self.halo_data, self.particle_data, self.params, self.mock_dir = self.staging()


    def staging(self):
        """
        Constructor call this function to load the halo+particle subsamples onto memory. 
        """
        # all paths relevant for mock generation
        scratch_dir = Path(self.scratch_dir)
        simname = Path(self.sim_name)
        sim_dir = Path(self.sim_dir)
        mock_dir = scratch_dir / simname / ('z%4.3f'%self.z_mock)
        # create mock_dir if not created
        mock_dir.mkdir(parents = True, exist_ok = True)
        subsample_dir = \
            Path(self.subsample_dir) / simname / ('z%4.3f'%self.z_mock)

        # load header to read parameters
        halo_info_fns = \
            list((sim_dir / simname / 'halos' / ('z%4.3f'%self.z_mock) / 'halo_info').glob('*.asdf'))
        f = asdf.open(halo_info_fns[0], lazy_load=True, copy_arrays=False)
        header = f['header']

        # constants
        params = {}
        params['z'] = self.z_mock
        params['h'] = header['H0']/100.
        params['Lbox'] = header['BoxSize'] # Mpc / h, box size
        params['Mpart'] = header['ParticleMassHMsun']  # Msun / h, mass of each particle
        params['velz2kms'] = header['VelZSpace_to_kms']/params['Lbox']
        params['numslabs'] = len(halo_info_fns)
        self.lbox = header['BoxSize']
        
        # # figure out number of halos and particles
        # if 'ELG' not in self.tracers.keys() and 'QSO' not in self.tracers.keys():
        #     numfile_name = subsample_dir / ("num_halos_parts.txt")
        # else:
        #     numfile_name = subsample_dir / ("num_halos_parts_MT.txt")
        # fnum = np.loadtxt(numfile_name)
        # fnum = np.array(sorted(fnum, key=lambda x : x[0])).astype(int)
        # ichunks = fnum[:, 0]
        # Nhalos = fnum[:, 1]
        # Nparts = fnum[:, 2]
        # Nhalos_tot = np.sum(Nhalos)
        # Nparts_tot = np.sum(Nparts)

        # count ther number of halos and particles
        Nhalos = np.empty(params['numslabs'])
        Nparts = np.empty(params['numslabs'])        
        for eslab in range(params['numslabs']):
            
            if 'ELG' not in self.tracers.keys() and 'QSO' not in self.tracers.keys():
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod'%eslab)
            else:
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod_MT'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod_MT'%eslab)            

            if self.want_ranks:
                particlefilename = str(particlefilename) + '_withranks'
            halofilename = str(halofilename) + '_new.h5'
            particlefilename = str(particlefilename) + '_new.h5'

            newfile = h5py.File(halofilename, 'r')
            newpart = h5py.File(particlefilename, 'r')
            Nhalos[eslab] = len(newfile['halos'])
            Nparts[eslab] = len(newpart['particles'])
        Nhalos = Nhalos.astype(int)
        Nparts = Nparts.astype(int)
        Nhalos_tot = int(np.sum(Nhalos))
        Nparts_tot = int(np.sum(Nparts))

        # list holding individual slabs
        hpos = np.empty((Nhalos_tot, 3))
        hvel = np.empty((Nhalos_tot, 3))
        hmass = np.empty([Nhalos_tot])
        hid = np.empty([Nhalos_tot], dtype = int)
        hmultis = np.empty([Nhalos_tot])
        hrandoms = np.empty([Nhalos_tot])
        hveldev = np.empty([Nhalos_tot])
        hdeltac = np.empty([Nhalos_tot])
        hfenv = np.empty([Nhalos_tot])

        ppos = np.empty((Nparts_tot, 3))
        pvel = np.empty((Nparts_tot, 3))
        phvel = np.empty((Nparts_tot, 3))
        phmass = np.empty([Nparts_tot])
        phid = np.empty([Nparts_tot], dtype = int)
        pNp = np.empty([Nparts_tot])
        psubsampling = np.empty([Nparts_tot])
        prandoms = np.empty([Nparts_tot])
        pdeltac = np.empty([Nparts_tot])
        pfenv = np.empty([Nparts_tot])

        # ranks
        if self.want_ranks:
            p_ranks = np.empty([Nparts_tot])
            p_ranksv = np.empty([Nparts_tot])
            p_ranksp = np.empty([Nparts_tot])
            p_ranksr = np.empty([Nparts_tot])

        # B.H. make into ASDF
        # load all the halo and particle data we need
        halo_ticker = 0
        parts_ticker = 0
        for eslab in range(params['numslabs']):
            print("Loading simulation by slab, ", eslab)
            
            if 'ELG' not in self.tracers.keys() and 'QSO' not in self.tracers.keys():
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod'%eslab)
            else:
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod_MT'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod_MT'%eslab)            

            if self.want_ranks:
                particlefilename = str(particlefilename) + '_withranks'
            halofilename = str(halofilename) + '_new.h5'
            particlefilename = str(particlefilename) + '_new.h5'

            start = time.time()
            newfile = h5py.File(halofilename, 'r')
            maskedhalos = newfile['halos']

            # extracting the halo properties that we need
            halo_ids = maskedhalos["id"].astype(int) # halo IDs
            halo_pos = maskedhalos["x_L2com"] # halo positions, Mpc / h
            halo_vels = maskedhalos['v_L2com'] # halo velocities, km/s
            halo_vel_dev = maskedhalos["randoms_gaus_vrms"] # halo velocity dispersions, km/s
            halo_mass = maskedhalos['N']*params['Mpart'] # halo mass, Msun / h, 200b
            halo_deltac = maskedhalos['deltac_rank'] # halo concentration
            halo_fenv = maskedhalos['fenv_rank'] # halo velocities, km/s
            halo_pstart = maskedhalos['npstartA'].astype(int) # starting index of particles
            halo_pnum = maskedhalos['npoutA'].astype(int) # number of particles 
            halo_multi = maskedhalos['multi_halos']
            halo_submask = maskedhalos['mask_subsample'].astype(bool)
            halo_randoms = maskedhalos['randoms']
            print("loading halo slab took ", time.time() - start)

            hpos[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_pos
            hvel[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_vels
            hmass[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_mass
            hid[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_ids
            hmultis[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_multi
            hrandoms[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_randoms
            hveldev[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_vel_dev
            hdeltac[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_deltac
            hfenv[halo_ticker: halo_ticker + Nhalos[eslab]] = halo_fenv
            halo_ticker += Nhalos[eslab]

            # extract particle data that we need
            newpart = h5py.File(particlefilename, 'r')
            subsample = newpart['particles']
            part_pos = subsample['pos']
            part_vel = subsample['vel']
            part_hvel = subsample['halo_vel']
            part_halomass = subsample['halo_mass'] # msun / h
            part_haloid = subsample['halo_id'].astype(int)
            part_Np = subsample['Np'] # number of particles that end up in the halo
            part_subsample = subsample['downsample_halo']
            part_randoms = subsample['randoms']
            part_deltac = subsample['halo_deltac']
            part_fenv = subsample['halo_fenv']

            if self.want_ranks:
                part_ranks = subsample['ranks']
                part_ranksv = subsample['ranksv']
                part_ranksp = subsample['ranksp']
                part_ranksr = subsample['ranksr']
                p_ranks[parts_ticker: parts_ticker + Nparts[eslab]] = part_ranks
                p_ranksv[parts_ticker: parts_ticker + Nparts[eslab]] = part_ranksv
                p_ranksp[parts_ticker: parts_ticker + Nparts[eslab]] = part_ranksp
                p_ranksr[parts_ticker: parts_ticker + Nparts[eslab]] = part_ranksr

            # #     part_data_slab += [part_ranks, part_ranksv, part_ranksp, part_ranksr]
            # particle_data = vstack([particle_data, new_part_table])
            ppos[parts_ticker: parts_ticker + Nparts[eslab]] = part_pos
            pvel[parts_ticker: parts_ticker + Nparts[eslab]] = part_vel
            phvel[parts_ticker: parts_ticker + Nparts[eslab]] =  part_hvel
            phmass[parts_ticker: parts_ticker + Nparts[eslab]] = part_halomass
            phid[parts_ticker: parts_ticker + Nparts[eslab]] =  part_haloid
            pNp[parts_ticker: parts_ticker + Nparts[eslab]] = part_Np
            psubsampling[parts_ticker: parts_ticker + Nparts[eslab]] = part_subsample
            prandoms[parts_ticker: parts_ticker + Nparts[eslab]] = part_randoms
            pdeltac[parts_ticker: parts_ticker + Nparts[eslab]] = part_deltac
            pfenv[parts_ticker: parts_ticker + Nparts[eslab]] = part_fenv
            parts_ticker += Nparts[eslab]

        halo_data = {"hpos": hpos, 
                     "hvel": hvel, 
                     "hmass": hmass, 
                     "hid": hid, 
                     "hmultis": hmultis, 
                     "hrandoms": hrandoms, 
                     "hveldev": hveldev, 
                     "hdeltac": hdeltac, 
                     "hfenv": hfenv}
        pweights = 1/pNp/psubsampling
        particle_data = {"ppos": ppos, 
                         "pvel": pvel, 
                         "phvel": phvel, 
                         "phmass": phmass, 
                         "phid": phid, 
                         "pweights": pweights, 
                         "prandoms": prandoms, 
                         "pdeltac": pdeltac, 
                         "pfenv": pfenv}
        if self.want_ranks:
            particle_data['pranks'] = p_ranks
            particle_data['pranksv'] = p_ranksv
            particle_data['pranksp'] = p_ranksp
            particle_data['pranksr'] = p_ranksr
        else:
            particle_data['pranks'] = np.ones(Nparts_tot)
            particle_data['pranksv'] =  np.ones(Nparts_tot)
            particle_data['pranksp'] =  np.ones(Nparts_tot)
            particle_data['pranksr'] =  np.ones(Nparts_tot)
            
        return halo_data, particle_data, params, mock_dir

    
    def run_hod(self, tracers, want_rsd = True, write_to_disk = False, Nthread = 16):
        """
        Runs a custom HOD.

        Parameters
        ----------
        ``tracers``: dict
            dictionary of multi-tracer HOD. ``tracers['LRG']`` is the dictionary of LRG HOD parameters,
            overwrites the ``LRG_params`` argument in the constructor.
            Same for keys ``'ELG'`` and ``'QSO'``.
        
        ``want_rsd``: bool 
            enable RSD? default ``True``.

        ``write_to_disk``: bool 
            output to disk? default ``False``. Setting to ``True`` decreases performance. 

        ``Nthread``: int
            Number of threads in the HOD run. Default 16. 

        Returns
        -------
        mock_dict: dict
            dictionary of galaxy outputs. Contains keys ``'LRG'``, ``'ELG'``, and ``'QSO'``. Each
            tracer key corresponds to a sub-dictionary that contains the galaxy properties with keys 
            ``'x'``, ``'y'``, ``'z'``, ``'vx'``, ``'vy'``, ``'vz'``, ``'mass'``, ``'id'``, ``Ncent'``.
            The coordinates are in Mpc/h, and the velocities are in km/s.
            The ``'mass'`` refers to host halo mass and is in units of Msun/h.
            The ``'id'`` refers to halo id, and the ``'Ncent'`` key refers to number of
            central galaxies for that tracer. The first ``'Ncent'`` galaxies 
            in the catalog are always centrals and the rest are satellites. 

        """
        mock_dict = gen_gal_cat(self.halo_data, self.particle_data, self.tracers, self.params, Nthread,
            enable_ranks = self.want_ranks, rsd = want_rsd, write_to_disk = write_to_disk, savedir = self.mock_dir)

        return mock_dict

    def compute_clustering(self, mock_dict, *args, **kwargs):
        """
        Computes summary statistics, currently enabling ``wp`` and ``xirppi``.

        Parameters
        ----------
        ``mock_dict``: dict
            dictionary of tracer positions. Output of ``run_hod``. 

        ``Ntread``: int
            number of threads in the HOD run. Default 16. 

        ``rpbins``: np.array
            array of transverse bins in Mpc/h.

        ``pimax``: int
            maximum bin edge along the line of sight direction, in Mpc/h. 

        ``pi_bin_size``: int
            size of bin along the line of sight. Currently, we only support linear binning along the line of sight. 

        Returns
        -------
        clustering: dict
            dicionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be 
            accessed with keys such as ``'LRG_ELG'``. 
        """
        if self.clustering_type == 'xirppi':
            clustering = self.compute_xirppi(mock_dict, *args, **kwargs)
        elif self.clustering_type == 'wp':
            clustering = self.compute_wp(mock_dict, *args, **kwargs)
        return clustering
    
    def compute_xirppi(self, mock_dict, rpbins, pimax, pi_bin_size, Nthread = 8):
        """
        Computes :math:`\\xi(r_p, \\pi)`.

        Parameters
        ----------
        ``mock_dict``: dict
            dictionary of tracer positions. Output of ``run_hod``. 

        ``Ntread``: int
            number of threads in the HOD run. Default 16. 

        ``rpbins``: np.array
            array of transverse bins in Mpc/h.

        ``pimax``: int
            maximum bin edge along the line of sight direction, in Mpc/h. 

        ``pi_bin_size``: int
            size of bin along the line of sight. Currently, we only support linear binning along the line of sight. 

        Returns
        -------
        clustering: dict
            dicionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be 
            accessed with keys such as ``'LRG_ELG'``. 
        """
        clustering = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2: continue # cross-correlations are symmetric
                x2 = mock_dict[tr2]['x']
                y2 = mock_dict[tr2]['y']
                z2 = mock_dict[tr2]['z']
                clustering[tr1+'_'+tr2] = calc_xirppi_fast(x1, y1, z1, rpbins, pimax, pi_bin_size, 
                    self.lbox, Nthread, x2 = x2, y2 = y2, z2 = z2)
                if i1 != i2: clustering[tr2+'_'+tr1] = clustering[tr1+'_'+tr2]
        return clustering

    def compute_wp(self, mock_dict, rpbins, pimax, pi_bin_size, Nthread = 8):
        """
        Computes :math:`w_p`.

        Parameters
        ----------
        ``mock_dict``: dict
            dictionary of tracer positions. Output of ``run_hod``. 

        ``Ntread``: int
            number of threads in the HOD run. Default 16. 

        ``rpbins``: np.array
            array of transverse bins in Mpc/h.

        ``pimax``: int
            maximum bin edge along the line of sight direction, in Mpc/h. 

        ``pi_bin_size``: int
            size of bin along the line of sight. Currently, we only support linear binning along the line of sight. 

        Returns
        -------
        clustering: dict
            dicionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be 
            accessed with keys such as ``'LRG_ELG'``. 
        """
        clustering = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2: continue # cross-correlations are symmetric
                x2 = mock_dict[tr2]['x']
                y2 = mock_dict[tr2]['y']
                z2 = mock_dict[tr2]['z']
                clustering[tr1+'_'+tr2] = calc_wp_fast(x1, y1, z1, rpbins, pimax, self.lbox, Nthread, x2 = x2, y2 = y2, z2 = z2)
                if i1 != i2: clustering[tr2+'_'+tr1] = clustering[tr1+'_'+tr2]
        return clustering

    def gal_reader(self, scratch_dir = None, simname = None, 
        sim_dir = None, z_mock = None, want_rsd = None, tracers = None):
        """
        Loads galaxy data given directory and return a ``mock_dict`` dictionary.  

        Parameters
        ----------
        ``sim_name``: str
            name of the simulation volume, e.g. 'AbacusSummit_base_c000_ph006'. 

        ``sim_dir``: str
            the directory that the simulation lives in, e.g. '/path/to/AbacusSummit/'.                                 

        ``scratch_dir``: str
            the diretory to save galaxy to, e.g. '/my/output/galalxies'. 

        ``z_mock``: floa
             which redshift slice, e.g. 0.5.    

        ``want_rsd``: bool
            RSD?

        ``tracers``: dict
            dictionary of tracer types to load, e.g. `{'LRG', 'ELG'}`.

        Returns
        -------
        ``mock_dict``: dict
            dictionary of tracer positions. Output of ``run_hod``. 

        """

        if scratch_dir == None:
            scratch_dir = Path(self.scratch_dir)
        if simname == None:
            simname = Path(self.sim_name)
        if sim_dir == None:
            sim_dir = Path(self.sim_dir)
        if z_mock == None:
            z_mock = self.z_mock
        if want_rsd == None:
            want_rsd = self.want_rsd
        if tracers == None:
            tracers = self.tracers.keys()
        mock_dir = scratch_dir / simname / ('z%4.3f'%self.z_mock)

        if want_rsd:
            rsd_string = "_rsd"
        else:
            rsd_string = ""

        outdir = (self.mock_dir) / ("galaxies"+rsd_string)

        mockdict = {}
        for tracer in tracers:
            mockdict[tracer] = ascii.read(outdir/(tracer+'s.dat'))
        return mockdict


