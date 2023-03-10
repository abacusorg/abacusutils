'''
'''


# The AbacusHOD module generates HOD tracers from Abacus simulations.
# A high-level overview of this module can be found in
# https://abacusutils.readthedocs.io/en/latest/hod.html
# or docs/hod.rst.


import gc
import time
from pathlib import Path

import asdf
import h5py
import numba
import numpy as np
from astropy.io import ascii
from numba import njit
from parallel_numpy_rng import MTGenerator

from ..analysis.power_spectrum import calc_power
from ..analysis.tpcf_corrfunc import (
    calc_multipole_fast,
    calc_wp_fast,
    calc_xirppi_fast,
)
from .GRAND_HOD import (
    gen_gal_cat,
    n_cen_LRG,
    n_sat_LRG_modified,
    N_cen_ELG_v1,
    N_sat_elg,
    N_cen_QSO,
    N_sat_generic,
)

# TODO B.H.: staging can be shorter and prettier; perhaps asdf for h5 and ecsv?

class AbacusHOD:
    """
    A highly efficient multi-tracer HOD code for the AbacusSummmit simulations.
    """
    def __init__(self, sim_params, HOD_params, clustering_params = None, chunk=-1, n_chunks=1):
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
                * ``output_dir``: str, the diretory to save galaxy to, e.g. '/my/output/galalxies'.
                * ``subsample_dir``: str, where to save halo+particle subsample, e.g. '/my/output/subsamples/'.
                * ``z_mock``: float, which redshift slice, e.g. 0.5.

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

        clustering_params: dict, optional
            Summary statistics configuration parameters. Load from ``config/abacus_hod.yaml``. It contains the following keys:
                * ``clustering_type``: str, which summary statistic to compute. Options: ``wp``, ``xirppi``, default: ``xirppi``.
                * ``bin_params``: dict, transverse scale binning.
                    * ``logmin``: float, :math:`\\log_{10}r_{\\mathrm{min}}` in Mpc/h.
                    * ``logmax``: float, :math:`\\log_{10}r_{\\mathrm{max}}` in Mpc/h.
                    * ``nbins``: int, number of bins.
                * ``pimax``: int, :math:`\\pi_{\\mathrm{max}}`.
                * ``pi_bin_size``: int, size of bins along of the line of sight. Need to be divisor of ``pimax``.
        chunk: int, optional
            Index of current chunk. Must be between ``0`` and ``n_chunks-1``. Files associated with this chunk are written out as ``{tracer}s_{chunk}.dat``. Default is -1 (no chunking).
        n_chunks: int, optional
            Number of chunks to split the input from the halo+particle subsample and number of output files in which to write out the galaxy catalogs following the format ``{tracer}s_{chunk}.dat``.
        """
        # simulation details
        self.sim_name = sim_params['sim_name']
        self.sim_dir = sim_params['sim_dir']
        self.subsample_dir = sim_params['subsample_dir']
        self.z_mock = sim_params['z_mock']
        self.output_dir = sim_params.get('output_dir', './')
        self.halo_lc = sim_params.get('halo_lc', False)

        # tracers
        tracer_flags = HOD_params['tracer_flags']
        tracers = {}
        for key in tracer_flags.keys():
            if tracer_flags[key]:
                tracers[key] = HOD_params[key+'_params']
        self.tracers = tracers

        # HOD parameter choices
        self.want_ranks = HOD_params.get('want_ranks', False)
        self.want_rsd = HOD_params['want_rsd']

        if clustering_params is not None:
            # clusteringparameters
            self.pimax = clustering_params.get('pimax', None)
            self.pi_bin_size = clustering_params.get('pi_bin_size', None)
            bin_params = clustering_params['bin_params']
            self.rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
            self.clustering_type = clustering_params.get('clustering_type', None)

        # setting up chunking
        self.chunk = chunk
        self.n_chunks = n_chunks
        assert self.chunk < self.n_chunks, "Total number of chunks needs to be larger than current chunk index"

        # load the subsample particles
        self.halo_data, self.particle_data, self.params, self.mock_dir = self.staging()


        # determine the halo mass function
        self.logMbins = np.linspace(
            np.log10(np.min(self.halo_data['hmass'])),
            np.log10(np.max(self.halo_data['hmass'])), 101)
        self.deltacbins = np.linspace(-0.5, 0.5, 101)
        self.fenvbins = np.linspace(-0.5, 0.5, 101)

        self.halo_mass_func, edges = np.histogramdd(
            np.vstack((np.log10(self.halo_data['hmass']), self.halo_data['hdeltac'], self.halo_data['hfenv'])).T,
            bins = [self.logMbins, self.deltacbins, self.fenvbins],
            weights = self.halo_data['hmultis'])

    def staging(self):
        """
        Constructor call this function to load the halo+particle subsamples onto memory.
        """
        # all paths relevant for mock generation
        output_dir = Path(self.output_dir)
        simname = Path(self.sim_name)
        sim_dir = Path(self.sim_dir)
        mock_dir = output_dir / simname / ('z%4.3f'%self.z_mock)
        # create mock_dir if not created
        mock_dir.mkdir(parents = True, exist_ok = True)
        subsample_dir = \
            Path(self.subsample_dir) / simname / ('z%4.3f'%self.z_mock)

        # load header to read parameters
        if self.halo_lc:
            halo_info_fns = [str(sim_dir / simname / ('z%4.3f'%self.z_mock) / 'lc_halo_info.asdf')]
        else:
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
        if self.halo_lc:
            params['origin'] = np.array(header['LightConeOrigins']).reshape(-1,3)[0]
        else:
            params['origin'] = None # observer at infinity in the -z direction

        # settitng up chunking
        n_chunks = self.n_chunks
        params['chunk'] = self.chunk
        if self.chunk == -1:
            chunk = 0
        else:
            chunk = self.chunk
        n_jump = int(np.ceil(len(halo_info_fns)/n_chunks))
        start = ((chunk)*n_jump)
        end = ((chunk+1)*n_jump)
        if end > len(halo_info_fns):
            end = len(halo_info_fns)
        params['numslabs'] = end-start
        self.lbox = header['BoxSize']

        # count ther number of halos and particles
        Nhalos = np.empty(params['numslabs'])
        Nparts = np.empty(params['numslabs'])
        for eslab in range(start, end):
            if 'ELG' not in self.tracers.keys() and 'QSO' not in self.tracers.keys():
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod_oldfenv'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod_oldfenv'%eslab)
            else:
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod_oldfenv_MT'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod_oldfenv_MT'%eslab)

            if self.want_ranks:
                particlefilename = str(particlefilename) + '_withranks'
            halofilename = str(halofilename) + '_new.h5'
            particlefilename = str(particlefilename) + '_new.h5'

            newfile = h5py.File(halofilename, 'r')
            newpart = h5py.File(particlefilename, 'r')

            Nhalos[eslab-start] = len(newfile['halos'])
            Nparts[eslab-start] = len(newpart['particles'])

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
        hsigma3d = np.empty([Nhalos_tot])
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
            p_ranksc = np.empty([Nparts_tot])

        # B.H. make into ASDF
        # load all the halo and particle data we need
        halo_ticker = 0
        parts_ticker = 0
        for eslab in range(start, end):

            print("Loading simulation by slab, ", eslab)
            if 'ELG' not in self.tracers.keys() and 'QSO' not in self.tracers.keys():
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod_oldfenv'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod_oldfenv'%eslab)
            else:
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod_oldfenv_MT'%eslab)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod_oldfenv_MT'%eslab)

            if self.want_ranks:
                particlefilename = str(particlefilename) + '_withranks'
            halofilename = str(halofilename) + '_new.h5'
            particlefilename = str(particlefilename) + '_new.h5'

            newfile = h5py.File(halofilename, 'r')
            maskedhalos = newfile['halos']

            # extracting the halo properties that we need
            halo_ids = maskedhalos["id"].astype(int) # halo IDs
            halo_pos = maskedhalos["x_L2com"] # halo positions, Mpc / h
            halo_vels = maskedhalos['v_L2com'] # halo velocities, km/s
            halo_vel_dev = maskedhalos["randoms_gaus_vrms"] # halo velocity dispersions, km/s
            halo_sigma3d = maskedhalos["sigmav3d_L2com"] # 3d velocity dispersion
            halo_mass = maskedhalos['N']*params['Mpart'] # halo mass, Msun / h, 200b
            halo_deltac = maskedhalos['deltac_rank'] # halo concentration
            halo_fenv = maskedhalos['fenv_rank'] # halo velocities, km/s
            # halo_pstart = maskedhalos['npstartA'].astype(int) # starting index of particles
            # halo_pnum = maskedhalos['npoutA'].astype(int) # number of particles
            halo_multi = maskedhalos['multi_halos']
            # halo_submask = maskedhalos['mask_subsample'].astype(bool)
            halo_randoms = maskedhalos['randoms']

            hpos[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_pos
            hvel[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_vels
            hmass[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_mass
            hid[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_ids
            hmultis[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_multi
            hrandoms[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_randoms
            hveldev[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_vel_dev
            hsigma3d[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_sigma3d
            hdeltac[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_deltac
            hfenv[halo_ticker: halo_ticker + Nhalos[eslab-start]] = halo_fenv
            halo_ticker += Nhalos[eslab-start]

            # extract particle data that we need
            newpart = h5py.File(particlefilename, 'r')
            subsample = newpart['particles']
            part_fields = subsample.dtype.fields.keys()
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
                assert 'ranks' in part_fields
                assert 'ranksv' in part_fields
                part_ranks = subsample['ranks']
                part_ranksv = subsample['ranksv']

                if 'ranksp' in part_fields:
                    part_ranksp = subsample['ranksp']
                else:
                    part_ranksp = np.zeros(len(subsample))

                if 'ranksr' in part_fields:
                    part_ranksr = subsample['ranksr']
                else:
                    part_ranksr = np.zeros(len(subsample))

                if 'ranksc' in part_fields:
                    part_ranksc = subsample['ranksc']
                else:
                    part_ranksc = np.zeros(len(subsample))

                p_ranks[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_ranks
                p_ranksv[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_ranksv
                p_ranksp[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_ranksp
                p_ranksr[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_ranksr
                p_ranksc[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_ranksc

            # #     part_data_slab += [part_ranks, part_ranksv, part_ranksp, part_ranksr]
            # particle_data = vstack([particle_data, new_part_table])
            ppos[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_pos
            pvel[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_vel
            phvel[parts_ticker: parts_ticker + Nparts[eslab-start]] =  part_hvel
            phmass[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_halomass
            phid[parts_ticker: parts_ticker + Nparts[eslab-start]] =  part_haloid
            pNp[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_Np
            psubsampling[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_subsample
            prandoms[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_randoms
            pdeltac[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_deltac
            pfenv[parts_ticker: parts_ticker + Nparts[eslab-start]] = part_fenv
            parts_ticker += Nparts[eslab-start]

        # sort halos by hid, important for conformity
        if not np.all(hid[:-1] <= hid[1:]):
            print("sorting halos for conformity calculation")
            sortind = np.argsort(hid)
            hpos = hpos[sortind]
            hvel = hvel[sortind]
            hmass = hmass[sortind]
            hid = hid[sortind]
            hmultis = hmultis[sortind]
            hrandoms = hrandoms[sortind]
            hveldev = hveldev[sortind]
            hsigma3d = hsigma3d[sortind]
            hdeltac = hdeltac[sortind]
            hfenv = hfenv[sortind]
        assert np.all(hid[:-1] <= hid[1:])

        halo_data = {"hpos": hpos,
                     "hvel": hvel,
                     "hmass": hmass,
                     "hid": hid,
                     "hmultis": hmultis,
                     "hrandoms": hrandoms,
                     "hveldev": hveldev,
                     "hsigma3d": hsigma3d,
                     "hdeltac": hdeltac,
                     "hfenv": hfenv}
        pweights = 1/pNp/psubsampling
        pinds = _searchsorted_parallel(hid, phid)
        particle_data = {"ppos": ppos,
                         "pvel": pvel,
                         "phvel": phvel,
                         "phmass": phmass,
                         "phid": phid,
                         "pweights": pweights,
                         "prandoms": prandoms,
                         "pdeltac": pdeltac,
                         "pfenv": pfenv,
                         "pinds": pinds}
        if self.want_ranks:
            particle_data['pranks'] = p_ranks
            particle_data['pranksv'] = p_ranksv
            particle_data['pranksp'] = p_ranksp
            particle_data['pranksr'] = p_ranksr
            particle_data['pranksc'] = p_ranksc
        else:
            particle_data['pranks'] = np.ones(Nparts_tot)
            particle_data['pranksv'] =  np.ones(Nparts_tot)
            particle_data['pranksp'] =  np.ones(Nparts_tot)
            particle_data['pranksr'] =  np.ones(Nparts_tot)
            particle_data['pranksc'] =  np.ones(Nparts_tot)

        return halo_data, particle_data, params, mock_dir

    def run_hod(self, tracers = None, want_rsd = True, reseed = None, write_to_disk = False,
        Nthread = 16, verbose = False, fn_ext = None):
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

        ``reseed``: int
            re-generate random numbers? supply random number seed. This overwrites the pre-generated random numbers, at a performance cost.
            Default ``None``.

        ``write_to_disk``: bool
            output to disk? default ``False``. Setting to ``True`` decreases performance.

        ``Nthread``: int
            number of threads in the HOD run. Default 16.

        ``verbose``: bool,
            detailed stdout? default ``False``.

        ``fn_ext``: str
            filename extension for saved files. Only relevant when ``write_to_disk = True``.

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
        if tracers is None:
            tracers = self.tracers
        if reseed:
            start = time.time()
            # np.random.seed(reseed)
            mtg = MTGenerator(np.random.PCG64(reseed))
            r1 = mtg.random(size=len(self.halo_data['hrandoms']), nthread=Nthread, dtype=np.float32)
            r2 = mtg.standard_normal(size=len(self.halo_data['hveldev']), nthread=Nthread, dtype=np.float32)
            r3 = mtg.random(size=len(self.particle_data['prandoms']), nthread=Nthread, dtype=np.float32)
            self.halo_data['hrandoms'] = r1
            self.halo_data['hveldev'] = r2*self.halo_data['hsigma3d']/np.sqrt(3)
            self.particle_data['prandoms'] = r3

            print("gen randoms took, ", time.time() - start)

        start = time.time()
        mock_dict = gen_gal_cat(self.halo_data, self.particle_data, tracers, self.params, Nthread,
            enable_ranks = self.want_ranks,
            rsd = want_rsd,
            write_to_disk = write_to_disk,
            savedir = self.mock_dir,
            verbose = verbose,
            fn_ext = fn_ext)
        print("gen mocks", time.time() - start)

        return mock_dict

    def compute_ngal(self, tracers = None, Nthread = 16):
        """
        Computes the number of each tracer generated by the HOD

        Parameters
        ----------
        ``tracers``: dict
            dictionary of multi-tracer HOD. ``tracers['LRG']`` is the dictionary of LRG HOD parameters,
            overwrites the ``LRG_params`` argument in the constructor.
            Same for keys ``'ELG'`` and ``'QSO'``.

        ``Nthread``: int
            Number of threads in the HOD run. Default 16.

        Returns
        -------
        ngal_dict: dict
        dictionary of number of each tracer.

        fsat_dict: dict
        dictionary of satellite fraction of each tracer.

        """
        if tracers is None:
            tracers = self.tracers

        ngal_dict = {}
        fsat_dict = {}
        for etracer in tracers.keys():
            tracer_hod = tracers[etracer]

            # used in z-evolving HOD
            Delta_a = 1./(1+self.z_mock) - 1./(1+tracer_hod.get('z_pivot', self.z_mock))
            if etracer == 'LRG':
                newngal = AbacusHOD._compute_ngal_lrg(
                    self.logMbins, self.deltacbins, self.fenvbins, self.halo_mass_func, tracer_hod['logM_cut'], tracer_hod['logM1'], tracer_hod['sigma'],
                    tracer_hod['alpha'], tracer_hod['kappa'], tracer_hod.get('logM_cut_pr', 0), tracer_hod.get('logM1_pr', 0), tracer_hod.get('Acent', 0),
                    tracer_hod.get('Asat', 0), tracer_hod.get('Bcent', 0), tracer_hod.get('Bsat', 0), tracer_hod.get('ic', 1), Delta_a, Nthread)
                ngal_dict[etracer] = newngal[0] + newngal[1]
                fsat_dict[etracer] = newngal[1] / (newngal[0] + newngal[1])
            elif etracer == 'ELG':
                newngal = AbacusHOD._compute_ngal_elg(
                    self.logMbins, self.deltacbins, self.fenvbins, self.halo_mass_func,
                    tracer_hod['p_max'], tracer_hod['Q'], tracer_hod['logM_cut'],
                    tracer_hod['kappa'], tracer_hod['sigma'], tracer_hod['logM1'],
                    tracer_hod['alpha'], tracer_hod['gamma'], tracer_hod.get('logM_cut_pr', 0),
                    tracer_hod.get('logM1_pr', 0), tracer_hod.get('A_s', 1),
                    tracer_hod.get('Acent', 0), tracer_hod.get('Asat', 0),
                    tracer_hod.get('Bcent', 0), tracer_hod.get('Bsat', 0),
                    tracer_hod.get('delta_M1', 0), tracer_hod.get('delta_alpha', 0),
                    tracer_hod.get('alpha1', 0), tracer_hod.get('beta', 0),
                    tracer_hod.get('ic', 1), Delta_a, Nthread)
                ngal_dict[etracer] = newngal[0] + newngal[1]
                fsat_dict[etracer] = newngal[1] / (newngal[0] + newngal[1])
            elif etracer == 'QSO':
                newngal = AbacusHOD._compute_ngal_qso(
                    self.logMbins, self.deltacbins, self.fenvbins, self.halo_mass_func,
                    tracer_hod['logM_cut'], tracer_hod['kappa'], tracer_hod['sigma'], tracer_hod['logM1'],
                    tracer_hod['alpha'], tracer_hod.get('logM_cut_pr', 0), tracer_hod.get('logM1_pr', 0), tracer_hod.get('Acent', 0),
                    tracer_hod.get('Asat', 0), tracer_hod.get('Bcent', 0), tracer_hod.get('Bsat', 0), tracer_hod.get('ic', 1), Delta_a, Nthread)
                ngal_dict[etracer] = newngal[0] + newngal[1]
                fsat_dict[etracer] = newngal[1] / (newngal[0] + newngal[1])
        return ngal_dict, fsat_dict

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _compute_ngal_lrg(logMbins, deltacbins, fenvbins, halo_mass_func,
                          logM_cut, logM1, sigma, alpha, kappa, logM_cut_pr, logM1_pr, Acent, Asat, Bcent, Bsat, ic, Delta_a, Nthread):
        """
        internal helper to compute number of LRGs
        """
        numba.set_num_threads(Nthread)

        logMs = 0.5*(logMbins[1:] + logMbins[:-1])
        deltacs = 0.5*(deltacbins[1:] + deltacbins[:-1])
        fenvs = 0.5*(fenvbins[1:] + fenvbins[:-1])
        ngal_cent = 0
        ngal_sat = 0
        # z-evolving HOD
        logM_cut = logM_cut + logM_cut_pr*Delta_a
        logM1 = logM1 + logM1_pr*Delta_a
        for i in numba.prange(len(logMbins) - 1):
            for j in range(len(deltacbins) - 1):
                for k in range(len(fenvbins) - 1):
                    Mh_temp = 10**logMs[i]
                    logM_cut_temp = logM_cut + Acent * deltacs[j] + Bcent * fenvs[k]
                    M1_temp = 10**(logM1 + Asat * deltacs[j] + Bsat * fenvs[k])
                    ncent_temp = n_cen_LRG(Mh_temp, logM_cut_temp, sigma)
                    nsat_temp = n_sat_LRG_modified(Mh_temp, logM_cut_temp,
                        10**logM_cut_temp, M1_temp, sigma, alpha, kappa)
                    ngal_cent += halo_mass_func[i, j, k] * ncent_temp * ic
                    ngal_sat += halo_mass_func[i, j, k] * nsat_temp * ic
        return ngal_cent, ngal_sat

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _compute_ngal_elg(logMbins, deltacbins, fenvbins, halo_mass_func, p_max, Q,
                          logM_cut, kappa, sigma, logM1, alpha, gamma, logM_cut_pr, logM1_pr, As, Acent, Asat, Bcent, Bsat,
                          delta_M1, delta_alpha, alpha1, beta, ic, Delta_a, Nthread):
        """
        internal helper to compute number of LRGs
        """
        numba.set_num_threads(Nthread)

        logMs = 0.5*(logMbins[1:] + logMbins[:-1])
        deltacs = 0.5*(deltacbins[1:] + deltacbins[:-1])
        fenvs = 0.5*(fenvbins[1:] + fenvbins[:-1])
        ngal_cent = 0
        ngal_sat = 0
        # z-evolving HOD
        logM_cut = logM_cut + logM_cut_pr*Delta_a
        logM1 = logM1 + logM1_pr*Delta_a
        for i in numba.prange(len(logMbins) - 1):
            for j in range(len(deltacbins) - 1):
                for k in range(len(fenvbins) - 1):
                    Mh_temp = 10**logMs[i]
                    logM_cut_temp = logM_cut + Acent * deltacs[j] + Bcent * fenvs[k]
                    M1_temp = 10**(logM1 + Asat * deltacs[j] + Bsat * fenvs[k])
                    ncent_temp = N_cen_ELG_v1(Mh_temp, p_max, Q, logM_cut_temp, sigma, gamma) * ic
                    nsat_temp = N_sat_elg(Mh_temp, 10**logM_cut_temp, kappa, M1_temp, alpha, As) * ic
                    # conformity treatment
                    M1_conf = M1_temp*10**delta_M1
                    alpha_conf = alpha + delta_alpha
                    nsat_conf = N_sat_elg(Mh_temp, 10**logM_cut_temp, kappa, M1_conf, alpha_conf, As, alpha1, beta) * ic

                    ngal_cent += halo_mass_func[i, j, k] * ncent_temp
                    ngal_sat += halo_mass_func[i, j, k] * (nsat_temp * (1-ncent_temp) + nsat_conf * ncent_temp)
        return ngal_cent, ngal_sat

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _compute_ngal_qso(logMbins, deltacbins, fenvbins, halo_mass_func,
                          logM_cut, kappa, sigma, logM1, alpha, logM_cut_pr, logM1_pr, Acent, Asat, Bcent, Bsat, ic, Delta_a, Nthread):
        """
        internal helper to compute number of LRGs
        """
        numba.set_num_threads(Nthread)

        logMs = 0.5*(logMbins[1:] + logMbins[:-1])
        deltacs = 0.5*(deltacbins[1:] + deltacbins[:-1])
        fenvs = 0.5*(fenvbins[1:] + fenvbins[:-1])
        ngal_cent = 0
        ngal_sat = 0
        # z-evolving HOD
        logM_cut = logM_cut + logM_cut_pr*Delta_a
        logM1 = logM1 + logM1_pr*Delta_a
        for i in numba.prange(len(logMbins) - 1):
            for j in range(len(deltacbins) - 1):
                for k in range(len(fenvbins) - 1):
                    Mh_temp = 10**logMs[i]
                    logM_cut_temp = logM_cut + Acent * deltacs[j] + Bcent * fenvs[k]
                    M1_temp = 10**(logM1 + Asat * deltacs[j] + Bsat * fenvs[k])
                    ncent_temp = N_cen_QSO(Mh_temp, logM_cut_temp, sigma)
                    nsat_temp = N_sat_generic(Mh_temp, 10**logM_cut_temp, kappa, M1_temp, alpha)
                    ngal_cent += halo_mass_func[i, j, k] * ncent_temp * ic
                    ngal_sat += halo_mass_func[i, j, k] * nsat_temp * ic
        return ngal_cent, ngal_sat

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
            dictionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be
            accessed with keys such as ``'LRG_ELG'``.
        """
        if self.clustering_type == 'xirppi':
            clustering = self.compute_xirppi(mock_dict, *args, **kwargs)
        elif self.clustering_type == 'wp':
            clustering = self.compute_wp(mock_dict, *args, **kwargs)
        elif self.clustering_type == 'multipole':
            clustering = self.compute_multipole(mock_dict, *args, **kwargs)
        else:
            raise ValueError('clustering_type not implemented or not specified, use xirppi, wp, multipole')
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
            dictionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be
            accessed with keys such as ``'LRG_ELG'``.
        """
        clustering = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2:
                    continue # cross-correlations are symmetric
                if i1 == i2: # auto corr
                    clustering[tr1+'_'+tr2] = calc_xirppi_fast(x1, y1, z1, rpbins, pimax, pi_bin_size,
                        self.lbox, Nthread)
                else:
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    clustering[tr1+'_'+tr2] = calc_xirppi_fast(x1, y1, z1, rpbins, pimax, pi_bin_size,
                        self.lbox, Nthread, x2 = x2, y2 = y2, z2 = z2)
                    clustering[tr2+'_'+tr1] = clustering[tr1+'_'+tr2]
        return clustering

    def compute_multipole(self, mock_dict, rpbins, pimax, Nthread = 8):
        clustering = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2:
                    continue # cross-correlations are symmetric
                if i1 == i2: # auto corr
                    new_multi = calc_multipole_fast(x1, y1, z1, rpbins,
                        self.lbox, Nthread)
                    new_wp = calc_wp_fast(x1, y1, z1, rpbins, pimax, self.lbox, Nthread)
                    clustering[tr1+'_'+tr2] = np.concatenate((new_wp, new_multi))
                else:
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    new_multi = calc_multipole_fast(x1, y1, z1, rpbins,
                        self.lbox, Nthread, x2 = x2, y2 = y2, z2 = z2)
                    new_wp = calc_wp_fast(x1, y1, z1, rpbins, pimax, self.lbox, Nthread,
                        x2 = x2, y2 = y2, z2 = z2)
                    clustering[tr1+'_'+tr2] = np.concatenate((new_wp, new_multi))
                    clustering[tr2+'_'+tr1] = clustering[tr1+'_'+tr2]
        return clustering

    def compute_power(self, mock_dict, nbins_k, nbins_mu, k_hMpc_max, logk, poles = [], paste = 'TSC', num_cells = 550, compensated = False, interlaced = False):
        r"""
        Computes :math:`P(k, \mu)` and/or :math:`P_\ell(k)`.

        TODO: parallelize, document, include deconvolution and aliasing, cross-correlations

        Parameters
        ----------
        ``mock_dict``: dict
            dictionary of tracer positions. Output of ``run_hod``.

        ``nbins_k``: int
            number of k bin centers (same convention as other correlation functions).

        ``nbins_mu``: int
            number of mu bin centers (same convention as other correlation functions).

        ``k_hMpc_max``: float
            maximum wavemode k in units of [Mpc/h]^-1. Note that the minimum k mode is
            currently set as the fundamental mode of the box

        ``logk``: bool
            flag determining whether the k bins are defined in log space or in normal space.

        ``poles``: list
            list of integers determining which multipoles to compute. Default is [], i.e. none.

        ``paste``: str
            scheme for painting particles on a mesh. Can be one of ``TSC`` or ``CIC``.

        ``num_cells``: int
            number of cells per dimension adopted for the particle gridding.

        ``compensated``: bool
            flag determining whether to apply the TSC/CIC grid deconvolution.

        ``interlaced``: bool
            flag determining whether to apply interlacing (i.e., aliasing).

        Returns
        -------
        clustering: dict
            dictionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'`` and ``'LRG_LRG_ell'`` for the
            multipoles with number of modes per bin, ``'LRG_LRG[_ell]_modes'``.
            Cross-correlations/spectra can be accessed with keys such
            as ``'LRG_ELG'`` and ``'LRG_ELG_ell'`` for the multipoles
            with number of modes per bin, ``'LRG_LRG[_ell]_modes'``. Keys ``k_binc``
            and ``mu_binc`` contain the bin centers of k and mu, respectively.
            The power spectrum P(k, mu) has a shape (nbins_k, nbins_mu), whereas
            the multipole power spectrum has shape (len(poles), nbins_k). Cubic box only.
        """
        Lbox = self.lbox
        clustering = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            w1 = mock_dict[tr1].get('w', None)
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2:
                    continue # cross-correlations are symmetric
                if i1 == i2:
                    print(tr1+'_'+tr2)
                    k_binc, mu_binc, pk3d, N3d, binned_poles, Npoles = calc_power(x1, y1, z1, nbins_k, nbins_mu, k_hMpc_max, logk, Lbox, paste, num_cells, compensated, interlaced, w = w1, poles = poles)
                    clustering[tr1+'_'+tr2] = pk3d
                    clustering[tr1+'_'+tr2+'_modes'] = N3d
                    clustering[tr1+'_'+tr2+'_ell'] = binned_poles
                    clustering[tr1+'_'+tr2+'_ell_modes'] = Npoles
                else:
                    print(tr1+'_'+tr2)
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    w2 = mock_dict[tr2].get('w', None)
                    k_binc, mu_binc, pk3d, N3d, binned_poles, Npoles = calc_power(x1, y1, z1, nbins_k, nbins_mu, k_hMpc_max, logk, Lbox, paste, num_cells, compensated, interlaced,
                                                      w = w1, x2 = x2, y2 = y2, z2 = z2, w2 = w2, poles = poles)
                    clustering[tr1+'_'+tr2] = pk3d
                    clustering[tr1+'_'+tr2+'_modes'] = N3d
                    clustering[tr1+'_'+tr2+'_ell'] = binned_poles
                    clustering[tr1+'_'+tr2+'_ell_modes'] = Npoles
                    clustering[tr2+'_'+tr1] = clustering[tr1+'_'+tr2]
                    clustering[tr2+'_'+tr1+'_modes'] = clustering[tr1+'_'+tr2+'_modes']
                    clustering[tr2+'_'+tr1+'_ell'] = clustering[tr1+'_'+tr2+'_ell']
                    clustering[tr2+'_'+tr1+'_ell_modes'] = clustering[tr1+'_'+tr2+'_ell_modes']
        clustering['k_binc'] = k_binc
        clustering['mu_binc'] = mu_binc
        return clustering

    def apply_zcv(self, mock_dict, config, load_presaved=False):
        """
        Apply control variates reduction of the variance to a power spectrum observable.
        """

        # ZCV module has optional dependencies, don't import unless necessary
        from .zcv.tools_jdr import run_zcv
        from .zcv.tracer_power import get_tracer_power

        # compute real space and redshift space
        #assert config['HOD_params']['want_rsd'], "Currently want_rsd=False not implemented"
        assert len(mock_dict.keys()) == 1, "Currently implemented only a single tracer" # should make a dict of dicts, but need cross
        assert len(config['power_params']['poles']) == 3, "Currently implemented only multipoles 0, 2, 4; need to change ZeNBu"

        # create save directory
        save_dir = Path(config['zcv_params']['zcv_dir']) / config['sim_params']['sim_name']
        save_z_dir = save_dir / f"z{config['sim_params']['z_mock']:.3f}"
        rsd_str = "_rsd" if config['HOD_params']['want_rsd'] else ""

        if load_presaved:
            pk_rsd_tr_dict = asdf.open(save_z_dir / f"power{rsd_str}_tr_nmesh{config['zcv_params']['nmesh']}.asdf")['data']
            pk_rsd_ij_dict = asdf.open(save_z_dir / f"power{rsd_str}_ij_nmesh{config['zcv_params']['nmesh']}.asdf")['data']
            if config['HOD_params']['want_rsd']:
                pk_tr_dict = asdf.open(save_z_dir / f"power_tr_nmesh{config['zcv_params']['nmesh']}.asdf")['data']
                pk_ij_dict = asdf.open(save_z_dir / f"power_ij_nmesh{config['zcv_params']['nmesh']}.asdf")['data']
            else:
                pk_tr_dict, pk_ij_dict = None, None

        else:
            # run version with rsd or without rsd
            for tr in mock_dict.keys():
                # obtain the positions
                tracer_pos = (np.vstack((mock_dict[tr]['x'], mock_dict[tr]['y'], mock_dict[tr]['z'])).T).astype(np.float32)
                del mock_dict
                gc.collect()

                # get power spectra for this tracer
                pk_rsd_tr_dict = get_tracer_power(tracer_pos, config['HOD_params']['want_rsd'], config)
                pk_rsd_ij_dict = asdf.open(save_z_dir / f"power{rsd_str}_ij_nmesh{config['zcv_params']['nmesh']}.asdf")['data']

            # run version without rsd if rsd was requested
            if config['HOD_params']['want_rsd']:
                mock_dict = self.run_hod(self.tracers, want_rsd=False, reseed=None, write_to_disk=False,
                                         Nthread=16, verbose=False, fn_ext=None)
                for tr in mock_dict.keys():
                    # obtain the positions
                    tracer_pos = (np.vstack((mock_dict[tr]['x'], mock_dict[tr]['y'], mock_dict[tr]['z'])).T).astype(np.float32)
                    del mock_dict
                    gc.collect()

                    # get power spectra for this tracer
                    pk_tr_dict = get_tracer_power(tracer_pos, want_rsd=False, config=config)
                    pk_ij_dict = asdf.open(save_z_dir / f"power_ij_nmesh{config['zcv_params']['nmesh']}.asdf")['data']
            else:
                pk_tr_dict, pk_ij_dict = None, None

        # run the final part and save
        zcv_dict = run_zcv(pk_rsd_tr_dict, pk_rsd_ij_dict, pk_tr_dict, pk_ij_dict, config)
        return zcv_dict

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
            dictionary of summary statistics. Auto-correlations/spectra can be
            accessed with keys such as ``'LRG_LRG'``. Cross-correlations/spectra can be
            accessed with keys such as ``'LRG_ELG'``.
        """
        clustering = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2:
                    continue # cross-correlations are symmetric
                if i1 == i2:
                    print(tr1+'_'+tr2)
                    clustering[tr1+'_'+tr2] = calc_wp_fast(x1, y1, z1, rpbins, pimax, self.lbox, Nthread)
                else:
                    print(tr1+'_'+tr2)
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    clustering[tr1+'_'+tr2] = calc_wp_fast(x1, y1, z1, rpbins, pimax, self.lbox, Nthread,
                        x2 = x2, y2 = y2, z2 = z2)
                    clustering[tr2+'_'+tr1] = clustering[tr1+'_'+tr2]
        return clustering


    def gal_reader(self, output_dir = None, simname = None,
        sim_dir = None, z_mock = None, want_rsd = None, tracers = None):
        """
        Loads galaxy data given directory and return a ``mock_dict`` dictionary.

        Parameters
        ----------
        ``sim_name``: str
            name of the simulation volume, e.g. 'AbacusSummit_base_c000_ph006'.

        ``sim_dir``: str
            the directory that the simulation lives in, e.g. '/path/to/AbacusSummit/'.

        ``output_dir``: str
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

        if output_dir is None:
            output_dir = Path(self.output_dir)
        if simname is None:
            simname = Path(self.sim_name)
        if sim_dir is None:
            sim_dir = Path(self.sim_dir)
        if z_mock is None:
            z_mock = self.z_mock
        if want_rsd is None:
            want_rsd = self.want_rsd
        if tracers is None:
            tracers = self.tracers.keys()
        # mock_dir = output_dir / simname / ('z%4.3f'%self.z_mock)

        if want_rsd:
            rsd_string = "_rsd"
        else:
            rsd_string = ""

        outdir = (self.mock_dir) / ("galaxies"+rsd_string)

        mockdict = {}
        for tracer in tracers:
            mockdict[tracer] = ascii.read(outdir/(tracer+'s.dat'))
        return mockdict

@njit(parallel=True)
def _searchsorted_parallel(a, b):
    res = np.empty(len(b), dtype = np.int64)
    for i in numba.prange(len(b)):
        res[i] = np.searchsorted(a, b[i])
    return res
