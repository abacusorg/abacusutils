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
HOD chains. A code template is given in ``abacusnbody/hod/run_hod.py`` for 
running a few example HODs and ``abacusnbody/hod/run_emcee.py`` for integrating 
with the ``emcee`` sampler. Here we provide a code snippet

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
>>> power_params = config['power_params']
>>> 
>>> # additional parameter choices
>>> want_rsd = HOD_params['want_rsd']
>>> write_to_disk = HOD_params['write_to_disk']
>>> 
>>> # create a new AbacusHOD object
>>> newBall = AbacusHOD(sim_params, HOD_params, power_params)
>>>     
>>> # first hod run, slow due to compiling jit, write to disk
>>> mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk)
>>> 
>>> # run the 10 different HODs for timing
>>> for i in range(10):
>>>     newBall.tracers['LRG']['alpha'] += 0.01
>>>     print("alpha = ",newBall.tracers['LRG']['alpha'])
>>>     start = time.time()
>>>     mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False)
>>>     print("Done iteration ", i, "took time ", time.time() - start)

The class also provides fast 2PCF calculators. For example to compute the 
redshift-space 2PCF (:math:`\\xi(r_p, \\pi)`):

>>> # load the rp pi binning from the config file
>>> bin_params = power_params['bin_params']
>>> rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
>>> pimax = power_params['pimax']
>>> pi_bin_size = power_params['pi_bin_size']    # the pi binning is configrured by pi_max and bin size
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


from .GRAND_HOD import gen_gal_cat
from .tpcf_corrfunc import calc_xirppi_fast, calc_wp_fast
# TODO B.H.: staging can be shorter and prettier; perhaps asdf for h5 and ecsv?

class AbacusHOD:
    """
    A highly efficient multi-tracer HOD code for the AbacusSummmit simulations.
    """
    def __init__(self, sim_params, HOD_params, power_params):
        """
        Loads simulation. The ``sim_params`` dictionary specifies which simulation
        volume to load. The ``HOD_params`` specifies the HOD parameters and tracer
        configurations. The ``power_params`` specifies the summary statistics 
        configurations. The ``HOD_params`` and ``power_params`` can be set to their
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

        power_params: dict
            Sumamry statistics configuration parameters. Load from ``config/abacus_hod.yaml``. It contains the following keys:
                * ``power_type``: str, which summary statistic to compute. Options: ``wp``, ``xirppi``, default: ``xirppi``.
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
        
        # power spectrum parameters
        self.pimax = power_params['pimax']
        self.pi_bin_size = power_params['pi_bin_size']
        bin_params = power_params['bin_params']
        self.rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
        self.power_type = power_params['power_type']
                
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
        
        # list holding individual slabs
        hpos = np.empty((1, 3))
        hvel = np.empty((1, 3))
        hmass = np.array([])
        hid = np.array([])
        hmultis = np.array([])
        hrandoms = np.array([])
        hveldev = np.array([])
        hdeltac = np.array([])
        hfenv = np.array([])

        ppos = np.empty((1, 3))
        pvel = np.empty((1, 3))
        phvel = np.empty((1, 3))
        phmass = np.array([])
        phid = np.array([])
        pNp = np.array([])
        psubsampling = np.array([])
        prandoms = np.array([])
        pdeltac = np.array([])
        pfenv = np.array([])

        # ranks
        if self.want_ranks:
            p_ranks = np.array([])
            p_ranksv = np.array([])
            p_ranksp = np.array([])
            p_ranksr = np.array([])

        # B.H. make into ASDF
        # load all the halo and particle data we need
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

            newfile = h5py.File(halofilename, 'r')
            allhalos = newfile['halos']
            mask = np.array(allhalos['mask_subsample'], dtype = bool)
            maskedhalos = allhalos[mask]

            # extracting the halo properties that we need
            halo_ids = np.array(maskedhalos["id"], dtype = int) # halo IDs
            halo_pos = maskedhalos["x_L2com"] # halo positions, Mpc / h
            halo_vels = maskedhalos['v_L2com'] # halo velocities, km/s
            halo_vel_dev = maskedhalos["randoms_gaus_vrms"] # halo velocity dispersions, km/s
            halo_mass = maskedhalos['N']*params['Mpart'] # halo mass, Msun / h, 200b
            halo_deltac = maskedhalos['deltac_rank'] # halo concentration
            halo_fenv = maskedhalos['fenv_rank'] # halo velocities, km/s
            halo_pstart = np.array(maskedhalos['npstartA'], dtype = int) # starting index of particles
            halo_pnum = np.array(maskedhalos['npoutA'], dtype = int) # number of particles 
            halo_multi = maskedhalos['multi_halos']
            halo_submask = np.array(maskedhalos['mask_subsample'], dtype = bool)
            halo_randoms = maskedhalos['randoms']
    
            hpos = np.concatenate((hpos, halo_pos))
            hvel = np.concatenate((hvel, halo_vels))
            hmass = np.concatenate((hmass, halo_mass))
            hid = np.concatenate((hid, halo_ids))
            hmultis = np.concatenate((hmultis, halo_multi))
            hrandoms = np.concatenate((hrandoms, halo_randoms))
            hveldev = np.concatenate((hveldev, halo_vel_dev))
            hdeltac = np.concatenate((hdeltac, halo_deltac))
            hfenv = np.concatenate((hfenv, halo_fenv))

            # extract particle data that we need
            newpart = h5py.File(particlefilename, 'r')
            subsample = newpart['particles']
            part_pos = subsample['pos']
            part_vel = subsample['vel']
            part_hvel = subsample['halo_vel']
            part_halomass = subsample['halo_mass'] # msun / h
            part_haloid = np.array(subsample['halo_id'], dtype = int)
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
                p_ranks = np.concatenate((p_ranks, part_ranks))
                p_ranksv = np.concatenate((p_ranksv, part_ranksv))
                p_ranksp = np.concatenate((p_ranksp, part_ranksp))
                p_ranksr = np.concatenate((p_ranksr, part_ranksr))

            # #     part_data_slab += [part_ranks, part_ranksv, part_ranksp, part_ranksr]
            # particle_data = vstack([particle_data, new_part_table])
            ppos = np.concatenate((ppos, part_pos))
            pvel = np.concatenate((pvel, part_vel))
            phvel = np.concatenate((phvel, part_hvel))
            phmass = np.concatenate((phmass, part_halomass))
            phid = np.concatenate((phid, part_haloid))
            pNp = np.concatenate((pNp, part_Np))
            psubsampling = np.concatenate((psubsampling, part_subsample))
            prandoms = np.concatenate((prandoms, part_randoms))
            pdeltac = np.concatenate((pdeltac, part_deltac))
            pfenv = np.concatenate((pfenv, part_fenv))

        hpos = hpos[1:]
        hvel = hvel[1:]
        ppos = ppos[1:]
        pvel = pvel[1:]

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
            particle_data['pranks'] = np.ones(len(phmass))
            particle_data['pranksv'] =  np.ones(len(phmass))
            particle_data['pranksp'] =  np.ones(len(phmass))
            particle_data['pranksr'] =  np.ones(len(phmass))
            
        return halo_data, particle_data, params, mock_dir

    
    def run_hod(self, tracers, want_rsd, write_to_disk = False, Nthread = 16):
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

        ``Ntread``: int
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

    def compute_power(self, mock_dict, *args, **kwargs):
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
        power: dict
            dicionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be 
            accessed with keys such as ``'LRG_ELG'``. 
        """
        if self.power_type == 'xirppi':
            power = self.compute_xirppi(mock_dict, *args, **kwargs)
        elif self.power_type == 'wp':
            power = self.compute_wp(mock_dict, *args, **kwargs)
        return power
    
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
        power: dict
            dicionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be 
            accessed with keys such as ``'LRG_ELG'``. 
        """
        power_spectra = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2: continue # cross-correlations are symmetric
                x2 = mock_dict[tr2]['x']
                y2 = mock_dict[tr2]['y']
                z2 = mock_dict[tr2]['z']
                power_spectra[tr1+'_'+tr2] = calc_xirppi_fast(x1, y1, z1, rpbins, pimax, pi_bin_size, 
                    self.lbox, Nthread, x2 = x2, y2 = y2, z2 = z2)
                if i1 != i2: power_spectra[tr2+'_'+tr1] = power_spectra[tr1+'_'+tr2]
        return power_spectra

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
        power: dict
            dicionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be 
            accessed with keys such as ``'LRG_ELG'``. 
        """
        power_spectra = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2: continue # cross-correlations are symmetric
                x2 = mock_dict[tr2]['x']
                y2 = mock_dict[tr2]['y']
                z2 = mock_dict[tr2]['z']
                power_spectra[tr1+'_'+tr2] = calc_wp_fast(x1, y1, z1, rpbins, pimax, self.lbox, Nthread, x2 = x2, y2 = y2, z2 = z2)
                if i1 != i2: power_spectra[tr2+'_'+tr1] = power_spectra[tr1+'_'+tr2]
        return power_spectra
