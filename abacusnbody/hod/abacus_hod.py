""" """


# The AbacusHOD module generates HOD tracers from Abacus simulations.
# A high-level overview of this module can be found in
# https://abacusutils.readthedocs.io/en/latest/hod.html
# or docs/hod.rst.

import gc
import time
from pathlib import Path
import logging

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

    def __init__(
        self,
        sim_params,
        HOD_params,
        clustering_params=None,
        chunk=-1,
        n_chunks=1,
        skip_staging=False,
    ):
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
        self.logger = logging.getLogger('AbacusHOD')
        # simulation details
        self.sim_name = sim_params['sim_name']
        self.sim_dir = sim_params['sim_dir']
        self.subsample_dir = sim_params['subsample_dir']
        self.z_mock = sim_params['z_mock']
        self.output_dir = sim_params.get('output_dir', './')
        self.halo_lc = sim_params.get('halo_lc', False)
        self.force_mt = sim_params.get('force_mt', False)  # use MT subsamples for LRG?

        ztype = None
        if self.halo_lc:
            ztype = 'lightcone'
        elif self.z_mock in [
            3.0,
            2.5,
            2.0,
            1.7,
            1.4,
            1.1,
            0.8,
            0.5,
            0.4,
            0.3,
            0.2,
            0.1,
            0.0,
        ]:
            ztype = 'primary'
        elif self.z_mock in [
            0.15,
            0.25,
            0.35,
            0.45,
            0.575,
            0.65,
            0.725,
            0.875,
            0.95,
            1.025,
            1.175,
            1.25,
            1.325,
            1.475,
            1.55,
            1.625,
            1.85,
            2.25,
            2.75,
            3.0,
            5.0,
            8.0,
        ]:
            ztype = 'secondary'
        else:
            raise Exception('illegal redshift')
        self.z_type = ztype

        # tracers
        tracer_flags = HOD_params['tracer_flags']
        tracers = {}
        for key in tracer_flags.keys():
            if tracer_flags[key]:
                tracers[key] = HOD_params[key + '_params']
        self.tracers = tracers

        # HOD parameter choices
        self.want_ranks = HOD_params.get('want_ranks', False)
        self.want_AB = HOD_params.get('want_AB', False)
        self.want_shear = HOD_params.get('want_shear', False)
        self.want_expvel = HOD_params.get('want_expvel', False)
        self.want_rsd = HOD_params['want_rsd']

        if clustering_params is not None:
            # clusteringparameters
            self.pimax = clustering_params.get('pimax', None)
            self.pi_bin_size = clustering_params.get('pi_bin_size', None)
            bin_params = clustering_params['bin_params']
            self.rpbins = np.logspace(
                bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1
            )
            self.clustering_type = clustering_params.get('clustering_type', None)

        # setting up chunking
        self.chunk = chunk
        self.n_chunks = n_chunks
        assert self.chunk < self.n_chunks, (
            'Total number of chunks needs to be larger than current chunk index'
        )

        if not skip_staging:
            # load the subsample particles
            self.halo_data, self.particle_data, self.params, self.mock_dir = (
                self.staging()
            )

            # determine the halo mass function
            self.logMbins = np.linspace(
                np.log10(np.min(self.halo_data['hmass'])),
                np.log10(np.max(self.halo_data['hmass'])),
                101,
            )
            self.deltacbins = np.linspace(-0.5, 0.5, 101)
            self.fenvbins = np.linspace(-0.5, 0.5, 101)
            self.shearbins = np.linspace(-0.5, 0.5, 101)

            self.halo_mass_func, edges = np.histogramdd(
                np.vstack(
                    (
                        np.log10(self.halo_data['hmass']),
                        self.halo_data.get(
                            'hdeltac', np.zeros(len(self.halo_data['hmass']))
                        ),
                        self.halo_data.get(
                            'hfenv', np.zeros(len(self.halo_data['hmass']))
                        ),
                    )
                ).T,
                bins=[self.logMbins, self.deltacbins, self.fenvbins],
                weights=self.halo_data['hmultis'],
            )
        else:
            from abacusnbody.metadata import get_meta

            meta = get_meta(self.sim_name, redshift=0.1)
            self.lbox = meta['BoxSize']

        if self.want_AB:
            assert 'hfenv' in self.halo_data.keys()
            assert 'hdeltac' in self.halo_data.keys()
        if self.want_shear:
            assert 'hshear' in self.halo_data.keys()

        self.halo_mass_func_wshear, edges = np.histogramdd(
            np.vstack(
                (
                    np.log10(self.halo_data['hmass']),
                    self.halo_data.get(
                        'hdeltac', np.zeros(len(self.halo_data['hmass']))
                    ),
                    self.halo_data.get('hfenv', np.zeros(len(self.halo_data['hmass']))),
                    self.halo_data.get(
                        'hshear', np.zeros(len(self.halo_data['hmass']))
                    ),
                )
            ).T,
            bins=[self.logMbins, self.deltacbins, self.fenvbins, self.shearbins],
            weights=self.halo_data['hmultis'],
        )

    def staging(self):
        """
        Constructor call this function to load the halo+particle subsamples onto memory.
        """
        # all paths relevant for mock generation
        output_dir = Path(self.output_dir)
        simname = Path(self.sim_name)
        sim_dir = Path(self.sim_dir)
        mock_dir = output_dir / simname / ('z%4.3f' % self.z_mock)
        # create mock_dir if not created
        mock_dir.mkdir(parents=True, exist_ok=True)
        subsample_dir = Path(self.subsample_dir) / simname / ('z%4.3f' % self.z_mock)

        # load header to read parameters
        if self.halo_lc:
            halo_info_fns = [
                str(sim_dir / simname / ('z%4.3f' % self.z_mock) / 'lc_halo_info.asdf')
            ]
        else:
            halo_info_fns = list(
                (
                    sim_dir / simname / 'halos' / ('z%4.3f' % self.z_mock) / 'halo_info'
                ).glob('*.asdf')
            )
        f = asdf.open(halo_info_fns[0], lazy_load=True)
        header = f['header']

        # constants
        params = {}
        params['z'] = self.z_mock
        params['h'] = header['H0'] / 100.0
        params['Lbox'] = header['BoxSize']  # Mpc / h, box size
        params['Mpart'] = header['ParticleMassHMsun']  # Msun / h, mass of each particle
        params['velz2kms'] = header['VelZSpace_to_kms'] / params['Lbox']
        if self.halo_lc:
            params['origin'] = np.array(header['LightConeOrigins']).reshape(-1, 3)[0]
        else:
            params['origin'] = None  # observer at infinity in the -z direction

        # settitng up chunking
        n_chunks = self.n_chunks
        params['chunk'] = self.chunk
        if self.chunk == -1:
            chunk = 0
        else:
            chunk = self.chunk
        n_jump = int(np.ceil(len(halo_info_fns) / n_chunks))
        start = (chunk) * n_jump
        end = (chunk + 1) * n_jump
        if end > len(halo_info_fns):
            end = len(halo_info_fns)
        params['numslabs'] = end - start
        self.lbox = header['BoxSize']

        # count ther number of halos and particles
        Nhalos = np.zeros(params['numslabs'])
        Nparts = np.zeros(params['numslabs'])
        for eslab in range(start, end):
            if (
                ('ELG' not in self.tracers.keys())
                and ('QSO' not in self.tracers.keys())
                and (not self.force_mt)
            ):
                halofilename = subsample_dir / (
                    'halos_xcom_%d_seed600_abacushod_oldfenv' % eslab
                )
                particlefilename = subsample_dir / (
                    'particles_xcom_%d_seed600_abacushod_oldfenv' % eslab
                )
            else:
                halofilename = subsample_dir / (
                    'halos_xcom_%d_seed600_abacushod_oldfenv_MT' % eslab
                )
                particlefilename = subsample_dir / (
                    'particles_xcom_%d_seed600_abacushod_oldfenv_MT' % eslab
                )

            if self.want_ranks:
                particlefilename = str(particlefilename) + '_withranks'
            halofilename = str(halofilename) + '_new.h5'
            particlefilename = str(particlefilename) + '_new.h5'

            newfile = h5py.File(halofilename, 'r')
            Nhalos[eslab - start] = len(newfile['halos'])
            if self.z_type == 'primary' or self.z_type == 'lightcone':
                newpart = h5py.File(particlefilename, 'r')
                Nparts[eslab - start] = len(newpart['particles'])

        Nhalos = Nhalos.astype(int)
        Nparts = Nparts.astype(int)
        Nhalos_tot = int(np.sum(Nhalos))
        Nparts_tot = int(np.sum(Nparts))

        # list holding individual slabs
        hpos = np.empty((Nhalos_tot, 3))
        hvel = np.empty((Nhalos_tot, 3))
        hmass = np.empty([Nhalos_tot])
        hid = np.empty([Nhalos_tot], dtype=int)
        hmultis = np.empty([Nhalos_tot])
        hrandoms = np.empty([Nhalos_tot])
        hveldev = np.empty((Nhalos_tot, 3))
        hsigma3d = np.empty([Nhalos_tot])
        hc = np.empty([Nhalos_tot])
        hrvir = np.empty([Nhalos_tot])
        if self.want_AB:
            hdeltac = np.empty([Nhalos_tot])
            hfenv = np.empty([Nhalos_tot])
        if self.want_shear:
            hshear = np.empty([Nhalos_tot])

        ppos = np.empty((Nparts_tot, 3))
        pvel = np.empty((Nparts_tot, 3))
        phvel = np.empty((Nparts_tot, 3))
        phmass = np.empty([Nparts_tot])
        phid = np.empty([Nparts_tot], dtype=int)
        pNp = np.empty([Nparts_tot])
        psubsampling = np.empty([Nparts_tot])
        prandoms = np.empty([Nparts_tot])
        if self.want_AB:
            pdeltac = np.empty([Nparts_tot])
            pfenv = np.empty([Nparts_tot])
        if self.want_shear:
            pshear = np.empty([Nparts_tot])

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
            self.logger.info(f'Loading simulation slab {eslab}')
            if (
                ('ELG' not in self.tracers.keys())
                and ('QSO' not in self.tracers.keys())
                and (not self.force_mt)
            ):
                halofilename = subsample_dir / (
                    'halos_xcom_%d_seed600_abacushod_oldfenv' % eslab
                )
                particlefilename = subsample_dir / (
                    'particles_xcom_%d_seed600_abacushod_oldfenv' % eslab
                )
            else:
                halofilename = subsample_dir / (
                    'halos_xcom_%d_seed600_abacushod_oldfenv_MT' % eslab
                )
                particlefilename = subsample_dir / (
                    'particles_xcom_%d_seed600_abacushod_oldfenv_MT' % eslab
                )

            if self.want_ranks:
                particlefilename = str(particlefilename) + '_withranks'
            halofilename = str(halofilename) + '_new.h5'
            particlefilename = str(particlefilename) + '_new.h5'

            newfile = h5py.File(halofilename, 'r')
            maskedhalos = newfile['halos']

            # extracting the halo properties that we need
            halo_ids = maskedhalos['id'].astype(int)  # halo IDs
            halo_pos = maskedhalos['x_L2com']  # halo positions, Mpc / h
            halo_vels = maskedhalos['v_L2com']  # halo velocities, km/s
            if self.want_expvel:
                halo_vel_dev = maskedhalos[
                    'randoms_exp'
                ]  # halo velocity dispersions, km/s
            else:
                halo_vel_dev = maskedhalos[
                    'randoms_gaus_vrms'
                ]  # halo velocity dispersions, km/s

            if len(halo_vel_dev.shape) == 1:
                self.logger.warning(
                    'Warning: galaxy x, y velocity bias randoms not set, using z randoms instead. x, y velocities may be unreliable.'
                )
                halo_vel_dev = np.concatenate(
                    (halo_vel_dev, halo_vel_dev, halo_vel_dev)
                ).reshape(-1, 3)
            halo_sigma3d = maskedhalos['sigmav3d_L2com']  # 3d velocity dispersion
            halo_c = (
                maskedhalos['r98_L2com'] / maskedhalos['r25_L2com']
            )  # concentration
            halo_rvir = maskedhalos['r98_L2com']  # rvir but using r98
            halo_mass = maskedhalos['N'] * params['Mpart']  # halo mass, Msun / h, 200b

            halo_deltac = maskedhalos['deltac_rank']  # halo concentration
            halo_fenv = maskedhalos['fenv_rank']  # halo velocities, km/s
            # halo_pstart = maskedhalos['npstartA'].astype(int) # starting index of particles
            # halo_pnum = maskedhalos['npoutA'].astype(int) # number of particles
            halo_multi = maskedhalos['multi_halos']
            # halo_submask = maskedhalos['mask_subsample'].astype(bool)
            halo_randoms = maskedhalos['randoms']

            hpos[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_pos
            hvel[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_vels
            hmass[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_mass
            hid[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_ids
            hmultis[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_multi
            hrandoms[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_randoms
            hveldev[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_vel_dev
            hsigma3d[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_sigma3d
            hc[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_c
            hrvir[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_rvir
            if self.want_AB:
                halo_deltac = maskedhalos['deltac_rank']  # halo concentration
                halo_fenv = maskedhalos['fenv_rank']  # halo velocities, km/s
                hdeltac[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_deltac
                hfenv[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_fenv
            if self.want_shear:
                halo_shear = maskedhalos['shear_rank']  # halo velocities, km/s
                hshear[halo_ticker : halo_ticker + Nhalos[eslab - start]] = halo_shear
            halo_ticker += Nhalos[eslab - start]

            if self.z_type == 'primary' or self.z_type == 'lightcone':
                # extract particle data that we need
                newpart = h5py.File(particlefilename, 'r')
                subsample = newpart['particles']
                part_fields = subsample.dtype.fields.keys()
                part_pos = subsample['pos']
                part_vel = subsample['vel']
                part_hvel = subsample['halo_vel']
                part_halomass = subsample['halo_mass']  # msun / h
                part_haloid = subsample['halo_id'].astype(int)
                part_Np = subsample['Np']  # number of particles that end up in the halo
                part_subsample = subsample['downsample_halo']
                part_randoms = subsample['randoms']
                if self.want_AB:
                    part_deltac = subsample['halo_deltac']
                    part_fenv = subsample['halo_fenv']
                if self.want_shear:
                    part_shear = subsample['halo_shear']

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

                    p_ranks[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_ranks
                    )
                    p_ranksv[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_ranksv
                    )
                    p_ranksp[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_ranksp
                    )
                    p_ranksr[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_ranksr
                    )
                    p_ranksc[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_ranksc
                    )

                # #     part_data_slab += [part_ranks, part_ranksv, part_ranksp, part_ranksr]
                # particle_data = vstack([particle_data, new_part_table])
                ppos[parts_ticker : parts_ticker + Nparts[eslab - start]] = part_pos
                pvel[parts_ticker : parts_ticker + Nparts[eslab - start]] = part_vel
                phvel[parts_ticker : parts_ticker + Nparts[eslab - start]] = part_hvel
                phmass[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                    part_halomass
                )
                phid[parts_ticker : parts_ticker + Nparts[eslab - start]] = part_haloid
                pNp[parts_ticker : parts_ticker + Nparts[eslab - start]] = part_Np
                psubsampling[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                    part_subsample
                )
                prandoms[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                    part_randoms
                )
                if self.want_AB:
                    pdeltac[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_deltac
                    )
                    pfenv[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_fenv
                    )
                if self.want_shear:
                    pshear[parts_ticker : parts_ticker + Nparts[eslab - start]] = (
                        part_shear
                    )
                parts_ticker += Nparts[eslab - start]

        # sort halos by hid, important for conformity
        if not np.all(hid[:-1] <= hid[1:]):
            self.logger.info('Sorting halos for conformity calculation.')
            sortind = np.argsort(hid)
            hpos = hpos[sortind]
            hvel = hvel[sortind]
            hmass = hmass[sortind]
            hid = hid[sortind]
            hmultis = hmultis[sortind]
            hrandoms = hrandoms[sortind]
            hveldev = hveldev[sortind]
            hsigma3d = hsigma3d[sortind]
            if self.want_AB:
                hdeltac = hdeltac[sortind]
                hfenv = hfenv[sortind]
            if self.want_shear:
                hshear = hshear[sortind]
        assert np.all(hid[:-1] <= hid[1:])

        halo_data = {
            'hpos': hpos,
            'hvel': hvel,
            'hmass': hmass,
            'hid': hid,
            'hmultis': hmultis,
            'hrandoms': hrandoms,
            'hveldev': hveldev,
            'hsigma3d': hsigma3d,
            'hc': hc,
            'hrvir': hrvir,
        }

        pweights = 1 / pNp / psubsampling
        pinds = _searchsorted_parallel(hid, phid)
        particle_data = {
            'ppos': ppos,
            'pvel': pvel,
            'phvel': phvel,
            'phmass': phmass,
            'phid': phid,
            'pweights': pweights,
            'prandoms': prandoms,
            'pinds': pinds,
        }
        if self.want_AB:
            halo_data['hdeltac'] = hdeltac
            halo_data['hfenv'] = hfenv
            particle_data['pdeltac'] = pdeltac
            particle_data['pfenv'] = pfenv
        if self.want_shear:
            halo_data['hshear'] = hshear
            particle_data['pshear'] = pshear

        if self.want_ranks:
            particle_data['pranks'] = p_ranks
            particle_data['pranksv'] = p_ranksv
            particle_data['pranksp'] = p_ranksp
            particle_data['pranksr'] = p_ranksr
            particle_data['pranksc'] = p_ranksc
        else:
            particle_data['pranks'] = np.ones(Nparts_tot)
            particle_data['pranksv'] = np.ones(Nparts_tot)
            particle_data['pranksp'] = np.ones(Nparts_tot)
            particle_data['pranksr'] = np.ones(Nparts_tot)
            particle_data['pranksc'] = np.ones(Nparts_tot)

        return halo_data, particle_data, params, mock_dir

    def run_hod(
        self,
        tracers=None,
        want_rsd=True,
        want_nfw=False,
        NFW_draw=None,
        reseed=None,
        write_to_disk=False,
        Nthread=16,
        verbose=False,
        fn_ext=None,
    ):
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

        ``want_nfw``: bool
            Distribute satellites on NFW instead of particles? default ``False``.
            Needs to feed in a long array of random numbers drawn from an NFW profile.
            !!! NFW profile is unoptimized. It has different velocity bias. It does not support lightcone. !!!

        ``NFW_draw``: np.array
            A long array of random numbers drawn from an NFW profile. P(x) = 1./(x*(1+x)**2)*x**2. default ``None``.
            Only needed if ``want_nfw == True``.

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
        if self.z_type == 'secondary':
            assert want_nfw
        if reseed:
            start = time.time()
            # np.random.seed(reseed)
            mtg = MTGenerator(np.random.PCG64(reseed))
            r1 = mtg.random(
                size=len(self.halo_data['hrandoms']), nthread=Nthread, dtype=np.float32
            )
            if self.want_expvel:
                rt0 = mtg.random(
                    size=len(self.halo_data['hrandoms']),
                    nthread=Nthread,
                    dtype=np.float32,
                )
                rt1 = mtg.random(
                    size=len(self.halo_data['hrandoms']),
                    nthread=Nthread,
                    dtype=np.float32,
                )
                rt2 = mtg.random(
                    size=len(self.halo_data['hrandoms']),
                    nthread=Nthread,
                    dtype=np.float32,
                )
                rt = np.vstack((rt0, rt1, rt2)).T
                r2 = np.zeros((len(rt), 3), dtype=np.float32)
                r2[rt >= 0.5] = -np.log(2 * (1 - rt[rt >= 0.5]))
                r2[rt < 0.5] = np.log(2 * rt[rt < 0.5])
            else:
                r20 = mtg.standard_normal(
                    size=len(self.halo_data['hveldev']),
                    nthread=Nthread,
                    dtype=np.float32,
                )
                r21 = mtg.standard_normal(
                    size=len(self.halo_data['hveldev']),
                    nthread=Nthread,
                    dtype=np.float32,
                )
                r22 = mtg.standard_normal(
                    size=len(self.halo_data['hveldev']),
                    nthread=Nthread,
                    dtype=np.float32,
                )
                r2 = np.vstack((r20, r21, r22)).T
            r3 = mtg.random(
                size=len(self.particle_data['prandoms']),
                nthread=Nthread,
                dtype=np.float32,
            )
            self.halo_data['hrandoms'] = r1
            if len(self.halo_data['hveldev'].shape) == 1:
                self.halo_data['hveldev'] = (
                    r20 * self.halo_data['hsigma3d'] / np.sqrt(3)
                )
            else:
                self.halo_data['hveldev'] = (
                    r2
                    * np.repeat(self.halo_data['hsigma3d'], 3).reshape((-1, 3))
                    / np.sqrt(3)
                )
            self.particle_data['prandoms'] = r3

            self.logger.info(
                f'Randoms generated in elapsed time {time.time() - start:.2f} s.'
            )

        start = time.time()
        mock_dict = gen_gal_cat(
            self.halo_data,
            self.particle_data,
            tracers,
            self.params,
            Nthread,
            enable_ranks=self.want_ranks,
            rsd=want_rsd,
            nfw=want_nfw,
            NFW_draw=NFW_draw,
            write_to_disk=write_to_disk,
            savedir=self.mock_dir,
            verbose=verbose,
            fn_ext=fn_ext,
        )
        self.logger.info(f'HOD generated in elapsed time {time.time() - start:.2f} s.')

        return mock_dict

    def compute_ngal(self, tracers=None, Nthread=16):
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
            Delta_a = 1.0 / (1 + self.z_mock) - 1.0 / (
                1 + tracer_hod.get('z_pivot', self.z_mock)
            )
            if etracer == 'LRG':
                newngal = AbacusHOD._compute_ngal_lrg(
                    self.logMbins,
                    self.deltacbins,
                    self.fenvbins,
                    self.halo_mass_func,
                    tracer_hod['logM_cut'],
                    tracer_hod['logM1'],
                    tracer_hod['sigma'],
                    tracer_hod['alpha'],
                    tracer_hod['kappa'],
                    tracer_hod.get('logM_cut_pr', 0),
                    tracer_hod.get('logM1_pr', 0),
                    tracer_hod.get('Acent', 0),
                    tracer_hod.get('Asat', 0),
                    tracer_hod.get('Bcent', 0),
                    tracer_hod.get('Bsat', 0),
                    tracer_hod.get('ic', 1),
                    Delta_a,
                    Nthread,
                )
                ngal_dict[etracer] = newngal[0] + newngal[1]
                fsat_dict[etracer] = newngal[1] / (newngal[0] + newngal[1])
            elif etracer == 'ELG':
                newngal = AbacusHOD._compute_ngal_elg(
                    self.logMbins,
                    self.deltacbins,
                    self.fenvbins,
                    self.shearbins,
                    self.halo_mass_func_wshear,
                    tracer_hod['p_max'],
                    tracer_hod['Q'],
                    tracer_hod['logM_cut'],
                    tracer_hod['kappa'],
                    tracer_hod['sigma'],
                    tracer_hod['logM1'],
                    tracer_hod['alpha'],
                    tracer_hod['gamma'],
                    tracer_hod.get('logM_cut_pr', 0),
                    tracer_hod.get('logM1_pr', 0),
                    tracer_hod.get('A_s', 1),
                    tracer_hod.get('Acent', 0),
                    tracer_hod.get('Asat', 0),
                    tracer_hod.get('Bcent', 0),
                    tracer_hod.get('Bsat', 0),
                    tracer_hod.get('Ccent', 0),
                    tracer_hod.get('Csat', 0),
                    tracer_hod.get('logM1_EE', tracer_hod['logM1']),
                    tracer_hod.get('alpha_EE', tracer_hod['alpha']),
                    tracer_hod.get('logM1_EL', tracer_hod['logM1']),
                    tracer_hod.get('alpha_EL', tracer_hod['alpha']),
                    tracer_hod.get('ic', 1),
                    Delta_a,
                    Nthread,
                )
                print('newngal', newngal)

                ngal_dict[etracer] = newngal[0] + newngal[1]
                fsat_dict[etracer] = newngal[1] / (newngal[0] + newngal[1])
            elif etracer == 'QSO':
                newngal = AbacusHOD._compute_ngal_qso(
                    self.logMbins,
                    self.deltacbins,
                    self.fenvbins,
                    self.halo_mass_func,
                    tracer_hod['logM_cut'],
                    tracer_hod['kappa'],
                    tracer_hod['sigma'],
                    tracer_hod['logM1'],
                    tracer_hod['alpha'],
                    tracer_hod.get('logM_cut_pr', 0),
                    tracer_hod.get('logM1_pr', 0),
                    tracer_hod.get('Acent', 0),
                    tracer_hod.get('Asat', 0),
                    tracer_hod.get('Bcent', 0),
                    tracer_hod.get('Bsat', 0),
                    tracer_hod.get('ic', 1),
                    Delta_a,
                    Nthread,
                )
                ngal_dict[etracer] = newngal[0] + newngal[1]
                fsat_dict[etracer] = newngal[1] / (newngal[0] + newngal[1])
        return ngal_dict, fsat_dict

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _compute_ngal_lrg(
        logMbins,
        deltacbins,
        fenvbins,
        halo_mass_func,
        logM_cut,
        logM1,
        sigma,
        alpha,
        kappa,
        logM_cut_pr,
        logM1_pr,
        Acent,
        Asat,
        Bcent,
        Bsat,
        ic,
        Delta_a,
        Nthread,
    ):
        """
        internal helper to compute number of LRGs
        """
        numba.set_num_threads(Nthread)

        logMs = 0.5 * (logMbins[1:] + logMbins[:-1])
        deltacs = 0.5 * (deltacbins[1:] + deltacbins[:-1])
        fenvs = 0.5 * (fenvbins[1:] + fenvbins[:-1])
        ngal_cent = 0
        ngal_sat = 0
        # z-evolving HOD
        logM_cut = logM_cut + logM_cut_pr * Delta_a
        logM1 = logM1 + logM1_pr * Delta_a
        for i in numba.prange(len(logMbins) - 1):
            for j in range(len(deltacbins) - 1):
                for k in range(len(fenvbins) - 1):
                    Mh_temp = 10 ** logMs[i]
                    logM_cut_temp = logM_cut + Acent * deltacs[j] + Bcent * fenvs[k]
                    M1_temp = 10 ** (logM1 + Asat * deltacs[j] + Bsat * fenvs[k])
                    ncent_temp = n_cen_LRG(Mh_temp, logM_cut_temp, sigma)
                    nsat_temp = n_sat_LRG_modified(
                        Mh_temp,
                        logM_cut_temp,
                        10**logM_cut_temp,
                        M1_temp,
                        sigma,
                        alpha,
                        kappa,
                    )
                    ngal_cent += halo_mass_func[i, j, k] * ncent_temp * ic
                    ngal_sat += halo_mass_func[i, j, k] * nsat_temp * ic
        return ngal_cent, ngal_sat

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _compute_ngal_elg(
        logMbins,
        deltacbins,
        fenvbins,
        shearbins,
        halo_mass_func,
        p_max,
        Q,
        logM_cut,
        kappa,
        sigma,
        logM1,
        alpha,
        gamma,
        logM_cut_pr,
        logM1_pr,
        As,
        Acent,
        Asat,
        Bcent,
        Bsat,
        Ccent,
        Csat,
        logM1_EE,
        alpha_EE,
        logM1_EL,
        alpha_EL,
        ic,
        Delta_a,
        Nthread,
    ):
        """
        internal helper to compute number of LRGs
        """
        numba.set_num_threads(Nthread)

        logMs = 0.5 * (logMbins[1:] + logMbins[:-1])
        deltacs = 0.5 * (deltacbins[1:] + deltacbins[:-1])
        fenvs = 0.5 * (fenvbins[1:] + fenvbins[:-1])
        shears = 0.5 * (shearbins[1:] + shearbins[:-1])
        ngal_cent = 0
        ngal_sat = 0
        # z-evolving HOD
        logM_cut = logM_cut + logM_cut_pr * Delta_a
        logM1 = logM1 + logM1_pr * Delta_a
        for i in numba.prange(len(logMbins) - 1):
            for j in range(len(deltacbins) - 1):
                for k in range(len(fenvbins) - 1):
                    for el in range(len(shearbins) - 1):
                        Mh_temp = 10 ** logMs[i]
                        logM_cut_temp = (
                            logM_cut
                            + Acent * deltacs[j]
                            + Bcent * fenvs[k]
                            + Ccent * shears[el]
                        )
                        M1_temp = 10 ** (
                            logM1
                            + Asat * deltacs[j]
                            + Bsat * fenvs[k]
                            + Csat * shears[el]
                        )
                        ncent_temp = (
                            N_cen_ELG_v1(Mh_temp, p_max, Q, logM_cut_temp, sigma, gamma)
                            * ic
                        )
                        nsat_temp = (
                            N_sat_elg(
                                Mh_temp, 10**logM_cut_temp, kappa, M1_temp, alpha, As
                            )
                            * ic
                        )
                        # conformity treatment

                        M1_conf = 10 ** (
                            logM1_EE
                            + Asat * deltacs[j]
                            + Bsat * fenvs[k]
                            + Csat * shears[el]
                        )
                        nsat_conf = (
                            N_sat_elg(
                                Mh_temp, 10**logM_cut_temp, kappa, M1_conf, alpha_EE, As
                            )
                            * ic
                        )
                        # we cannot calculate the number of EL conformal satellites with this approach, so we ignore it for now.

                        ngal_cent += halo_mass_func[i, j, k, el] * ncent_temp
                        ngal_sat += halo_mass_func[i, j, k, el] * (
                            nsat_temp * (1 - ncent_temp) + nsat_conf * ncent_temp
                        )
                        # print(Mh_temp, 10**logM_cut_temp, kappa, M1_temp, alpha, As)
        return ngal_cent, ngal_sat

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _compute_ngal_qso(
        logMbins,
        deltacbins,
        fenvbins,
        halo_mass_func,
        logM_cut,
        kappa,
        sigma,
        logM1,
        alpha,
        logM_cut_pr,
        logM1_pr,
        Acent,
        Asat,
        Bcent,
        Bsat,
        ic,
        Delta_a,
        Nthread,
    ):
        """
        internal helper to compute number of LRGs
        """
        numba.set_num_threads(Nthread)

        logMs = 0.5 * (logMbins[1:] + logMbins[:-1])
        deltacs = 0.5 * (deltacbins[1:] + deltacbins[:-1])
        fenvs = 0.5 * (fenvbins[1:] + fenvbins[:-1])
        ngal_cent = 0
        ngal_sat = 0
        # z-evolving HOD
        logM_cut = logM_cut + logM_cut_pr * Delta_a
        logM1 = logM1 + logM1_pr * Delta_a
        for i in numba.prange(len(logMbins) - 1):
            for j in range(len(deltacbins) - 1):
                for k in range(len(fenvbins) - 1):
                    Mh_temp = 10 ** logMs[i]
                    logM_cut_temp = logM_cut + Acent * deltacs[j] + Bcent * fenvs[k]
                    M1_temp = 10 ** (logM1 + Asat * deltacs[j] + Bsat * fenvs[k])
                    ncent_temp = N_cen_QSO(Mh_temp, logM_cut_temp, sigma)
                    nsat_temp = N_sat_generic(
                        Mh_temp, 10**logM_cut_temp, kappa, M1_temp, alpha
                    )
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
            raise ValueError(
                'clustering_type not implemented or not specified, use xirppi, wp, multipole'
            )
        return clustering

    def compute_xirppi(self, mock_dict, rpbins, pimax, pi_bin_size, Nthread=8):
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
                    continue  # cross-correlations are symmetric
                if i1 == i2:  # auto corr
                    clustering[tr1 + '_' + tr2] = calc_xirppi_fast(
                        x1, y1, z1, rpbins, pimax, pi_bin_size, self.lbox, Nthread
                    )
                else:
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    clustering[tr1 + '_' + tr2] = calc_xirppi_fast(
                        x1,
                        y1,
                        z1,
                        rpbins,
                        pimax,
                        pi_bin_size,
                        self.lbox,
                        Nthread,
                        x2=x2,
                        y2=y2,
                        z2=z2,
                    )
                    clustering[tr2 + '_' + tr1] = clustering[tr1 + '_' + tr2]
        return clustering

    def compute_multipole(
        self, mock_dict, rpbins, pimax, sbins, nbins_mu, orders=[0, 2], Nthread=8
    ):
        clustering = {}
        for i1, tr1 in enumerate(mock_dict.keys()):
            x1 = mock_dict[tr1]['x']
            y1 = mock_dict[tr1]['y']
            z1 = mock_dict[tr1]['z']
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2:
                    continue  # cross-correlations are symmetric
                if i1 == i2:  # auto corr
                    new_multi = calc_multipole_fast(
                        x1,
                        y1,
                        z1,
                        sbins,
                        self.lbox,
                        Nthread,
                        nbins_mu=nbins_mu,
                        orders=orders,
                    )
                    new_wp = calc_wp_fast(x1, y1, z1, rpbins, pimax, self.lbox, Nthread)
                    clustering[tr1 + '_' + tr2] = np.concatenate((new_wp, new_multi))
                else:
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    new_multi = calc_multipole_fast(
                        x1,
                        y1,
                        z1,
                        rpbins,
                        self.lbox,
                        Nthread,
                        x2=x2,
                        y2=y2,
                        z2=z2,
                        nbins_mu=nbins_mu,
                        orders=orders,
                    )
                    new_wp = calc_wp_fast(
                        x1,
                        y1,
                        z1,
                        rpbins,
                        pimax,
                        self.lbox,
                        Nthread,
                        x2=x2,
                        y2=y2,
                        z2=z2,
                    )
                    clustering[tr1 + '_' + tr2] = np.concatenate((new_wp, new_multi))
                    clustering[tr2 + '_' + tr1] = clustering[tr1 + '_' + tr2]
        return clustering

    def compute_power(
        self,
        mock_dict,
        nbins_k,
        nbins_mu,
        k_hMpc_max,
        logk,
        poles=[],
        paste='TSC',
        num_cells=550,
        compensated=False,
        interlaced=False,
    ):
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
            pos1 = np.stack((x1, y1, z1), axis=1)
            w1 = mock_dict[tr1].get('w', None)
            for i2, tr2 in enumerate(mock_dict.keys()):
                if i1 > i2:
                    continue  # cross-correlations are symmetric
                if i1 == i2:
                    print(tr1 + '_' + tr2)
                    power = calc_power(
                        pos1,
                        Lbox,
                        nbins_k,
                        nbins_mu,
                        k_hMpc_max,
                        logk,
                        paste,
                        num_cells,
                        compensated,
                        interlaced,
                        w=w1,
                        poles=poles,
                    )
                    clustering[tr1 + '_' + tr2] = power['power']
                    clustering[tr1 + '_' + tr2 + '_modes'] = power['N_mode']
                    clustering[tr1 + '_' + tr2 + '_ell'] = power['poles']
                    clustering[tr1 + '_' + tr2 + '_ell_modes'] = power['N_mode_poles']
                else:
                    print(tr1 + '_' + tr2)
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    pos2 = np.stack((x2, y2, z2), axis=1)
                    w2 = mock_dict[tr2].get('w', None)
                    power = calc_power(
                        pos1,
                        Lbox,
                        nbins_k,
                        nbins_mu,
                        k_hMpc_max,
                        logk,
                        paste,
                        num_cells,
                        compensated,
                        interlaced,
                        w=w1,
                        pos2=pos2,
                        w2=w2,
                        poles=poles,
                    )
                    clustering[tr1 + '_' + tr2] = power['power']
                    clustering[tr1 + '_' + tr2 + '_modes'] = power['N_mode']
                    clustering[tr1 + '_' + tr2 + '_ell'] = power['poles']
                    clustering[tr1 + '_' + tr2 + '_ell_modes'] = power['N_mode_poles']
                    clustering[tr2 + '_' + tr1] = clustering[tr1 + '_' + tr2]
                    clustering[tr2 + '_' + tr1 + '_modes'] = clustering[
                        tr1 + '_' + tr2 + '_modes'
                    ]
                    clustering[tr2 + '_' + tr1 + '_ell'] = clustering[
                        tr1 + '_' + tr2 + '_ell'
                    ]
                    clustering[tr2 + '_' + tr1 + '_ell_modes'] = clustering[
                        tr1 + '_' + tr2 + '_ell_modes'
                    ]
        clustering['k_binc'] = power['k_mid']
        clustering['mu_binc'] = power['mu_mid'][0]
        return clustering

    def apply_zcv(self, mock_dict, config, load_presaved=False):
        """
        Apply control variates reduction of the variance to a power spectrum observable.
        """

        # ZCV module has optional dependencies, don't import unless necessary
        from .zcv.tools_cv import run_zcv
        from .zcv.tracer_power import get_tracer_power
        from ..analysis.power_spectrum import get_k_mu_edges

        # compute real space and redshift space
        # assert config['HOD_params']['want_rsd'], "Currently want_rsd=False not implemented"
        assert len(mock_dict.keys()) == 1, (
            'Currently implemented only a single tracer'
        )  # should make a dict of dicts, but need cross
        assert len(config['power_params']['poles']) <= 3, (
            'Currently implemented only multipoles 0, 2, 4; need to change ZeNBu'
        )
        assert config['power_params']['nbins_mu'] == 1, (
            'Currently wedges are not implemented; need to change ZeNBu'
        )
        if 'nmesh' not in config['power_params'].keys():
            config['power_params']['nmesh'] = config['zcv_params']['nmesh']
        assert config['zcv_params']['nmesh'] == config['power_params']['nmesh'], (
            '`nmesh` in `power_params` and `zcv_params` should match.'
        )

        # create save directory
        save_dir = (
            Path(config['zcv_params']['zcv_dir']) / config['sim_params']['sim_name']
        )
        save_z_dir = save_dir / f'z{config["sim_params"]["z_mock"]:.3f}'
        rsd_str = '_rsd' if config['HOD_params']['want_rsd'] else ''

        # define bins
        Lbox = self.lbox
        k_bin_edges, mu_bin_edges = get_k_mu_edges(
            Lbox,
            config['power_params']['k_hMpc_max'],
            config['power_params']['nbins_k'],
            config['power_params']['nbins_mu'],
            config['power_params']['logk'],
        )
        k_binc = 0.5 * (k_bin_edges[1:] + k_bin_edges[:-1])
        mu_binc = 0.5 * (mu_bin_edges[1:] + mu_bin_edges[:-1])

        # get file names
        if not config['power_params']['logk']:
            dk = k_bin_edges[1] - k_bin_edges[0]
        else:
            dk = np.log(k_bin_edges[1] / k_bin_edges[0])
        if config['power_params']['nbins_k'] == config['zcv_params']['nmesh'] // 2:
            power_rsd_tr_fn = (
                save_z_dir
                / f'power{rsd_str}_tr_nmesh{config["zcv_params"]["nmesh"]}.asdf'
            )
            power_rsd_ij_fn = (
                save_z_dir
                / f'power{rsd_str}_ij_nmesh{config["zcv_params"]["nmesh"]}.asdf'
            )
            power_tr_fn = (
                save_z_dir / f'power_tr_nmesh{config["zcv_params"]["nmesh"]}.asdf'
            )
            power_ij_fn = (
                save_z_dir / f'power_ij_nmesh{config["zcv_params"]["nmesh"]}.asdf'
            )
        else:
            power_rsd_tr_fn = (
                save_z_dir
                / f'power{rsd_str}_tr_nmesh{config["zcv_params"]["nmesh"]}_dk{dk:.3f}.asdf'
            )
            power_rsd_ij_fn = (
                save_z_dir
                / f'power{rsd_str}_ij_nmesh{config["zcv_params"]["nmesh"]}_dk{dk:.3f}.asdf'
            )
            power_tr_fn = (
                save_z_dir
                / f'power_tr_nmesh{config["zcv_params"]["nmesh"]}_dk{dk:.3f}.asdf'
            )
            power_ij_fn = (
                save_z_dir
                / f'power_ij_nmesh{config["zcv_params"]["nmesh"]}_dk{dk:.3f}.asdf'
            )
        pk_fns = [power_rsd_tr_fn, power_rsd_ij_fn, power_tr_fn, power_ij_fn]
        for fn in pk_fns:
            try:
                assert np.isclose(
                    asdf.open(fn)['header']['kcut'], config['zcv_params']['kcut']
                ), f'Mismatching file: {str(fn)}'
            except FileNotFoundError:
                pass

        if load_presaved:
            pk_rsd_tr_dict = asdf.open(power_rsd_tr_fn)['data']
            pk_rsd_ij_dict = asdf.open(power_rsd_ij_fn)['data']
            assert np.allclose(k_binc, pk_rsd_tr_dict['k_binc']), (
                f'Mismatching file: {str(power_rsd_tr_fn)}'
            )
            assert np.allclose(k_binc, pk_rsd_ij_dict['k_binc']), (
                f'Mismatching file: {str(power_rsd_ij_fn)}'
            )
            assert np.allclose(mu_binc, pk_rsd_tr_dict['mu_binc']), (
                f'Mismatching file: {str(power_rsd_tr_fn)}'
            )
            assert np.allclose(mu_binc, pk_rsd_ij_dict['mu_binc']), (
                f'Mismatching file: {str(power_rsd_ij_fn)}'
            )
            if config['HOD_params']['want_rsd']:
                pk_tr_dict = asdf.open(power_tr_fn)['data']
                pk_ij_dict = asdf.open(power_ij_fn)['data']
                assert np.allclose(k_binc, pk_tr_dict['k_binc']), (
                    f'Mismatching file: {str(power_tr_fn)}'
                )
                assert np.allclose(k_binc, pk_ij_dict['k_binc']), (
                    f'Mismatching file: {str(power_ij_fn)}'
                )
                assert np.allclose(mu_binc, pk_tr_dict['mu_binc']), (
                    f'Mismatching file: {str(power_tr_fn)}'
                )
                assert np.allclose(mu_binc, pk_ij_dict['mu_binc']), (
                    f'Mismatching file: {str(power_ij_fn)}'
                )
            else:
                pk_tr_dict, pk_ij_dict = None, None

        else:
            # run version with rsd or without rsd
            for tr in mock_dict.keys():
                # obtain the positions
                tracer_pos = (
                    np.vstack(
                        (mock_dict[tr]['x'], mock_dict[tr]['y'], mock_dict[tr]['z'])
                    ).T
                ).astype(np.float32)
                del mock_dict
                gc.collect()

                # get power spectra for this tracer
                pk_rsd_tr_dict = get_tracer_power(
                    tracer_pos, config['HOD_params']['want_rsd'], config
                )
                pk_rsd_ij_dict = asdf.open(power_rsd_ij_fn)['data']
                assert np.allclose(k_binc, pk_rsd_ij_dict['k_binc']), (
                    f'Mismatching file: {str(power_rsd_ij_fn)}'
                )
                assert np.allclose(mu_binc, pk_rsd_ij_dict['mu_binc']), (
                    f'Mismatching file: {str(power_rsd_ij_fn)}'
                )
            # run version without rsd if rsd was requested
            if config['HOD_params']['want_rsd']:
                mock_dict = self.run_hod(
                    self.tracers,
                    want_rsd=False,
                    reseed=None,
                    write_to_disk=False,
                    Nthread=16,
                    verbose=False,
                    fn_ext=None,
                )
                for tr in mock_dict.keys():
                    # obtain the positions
                    tracer_pos = (
                        np.vstack(
                            (mock_dict[tr]['x'], mock_dict[tr]['y'], mock_dict[tr]['z'])
                        ).T
                    ).astype(np.float32)
                    del mock_dict
                    gc.collect()

                    # get power spectra for this tracer
                    pk_tr_dict = get_tracer_power(
                        tracer_pos, want_rsd=False, config=config
                    )
                    pk_ij_dict = asdf.open(power_ij_fn)['data']
                    assert np.allclose(k_binc, pk_ij_dict['k_binc']), (
                        f'Mismatching file: {str(power_ij_fn)}'
                    )
                    assert np.allclose(mu_binc, pk_ij_dict['mu_binc']), (
                        f'Mismatching file: {str(power_ij_fn)}'
                    )
            else:
                pk_tr_dict, pk_ij_dict = None, None

        # run the final part and save
        zcv_dict = run_zcv(
            pk_rsd_tr_dict, pk_rsd_ij_dict, pk_tr_dict, pk_ij_dict, config
        )
        return zcv_dict

    def apply_zcv_xi(self, mock_dict, config, load_presaved=False):
        """
        Apply control variates reduction of the variance to a power spectrum observable.
        """

        # ZCV module has optional dependencies, don't import unless necessary
        from .zcv.tools_cv import run_zcv_field
        from .zcv.tracer_power import get_tracer_power
        from ..analysis.power_spectrum import pk_to_xi

        # compute real space and redshift space
        assert config['HOD_params']['want_rsd'], (
            'Currently want_rsd=False not implemented'
        )
        assert len(mock_dict.keys()) == 1, (
            'Currently implemented only a single tracer'
        )  # should make a dict of dicts, but need cross
        assert len(config['power_params']['poles']) <= 3, (
            'Currently implemented only multipoles 0, 2, 4; need to change ZeNBu'
        )
        assert config['power_params']['nbins_mu'] == 1, (
            'Currently wedges are not implemented; need to change ZeNBu'
        )
        if 'nmesh' not in config['power_params'].keys():
            config['power_params']['nmesh'] = config['zcv_params']['nmesh']
        assert config['zcv_params']['nmesh'] == config['power_params']['nmesh'], (
            '`nmesh` in `power_params` and `zcv_params` should match.'
        )

        # create save directory
        save_dir = (
            Path(config['zcv_params']['zcv_dir']) / config['sim_params']['sim_name']
        )
        save_z_dir = save_dir / f'z{config["sim_params"]["z_mock"]:.3f}'
        rsd_str = '_rsd' if config['HOD_params']['want_rsd'] else ''

        # construct names of files based on fields
        keynames = config['zcv_params']['fields']

        # tracer and field file names
        pk_rsd_tr_fns = []
        pk_tr_fns = []
        pk_rsd_ij_fns = []
        pk_ij_fns = []
        pk_rsd_tr_fns.append(
            save_z_dir
            / f'power{rsd_str}_tr_tr_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
        )
        pk_tr_fns.append(
            save_z_dir / f'power_tr_tr_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
        )
        for i in range(len(keynames)):
            pk_rsd_tr_fns.append(
                save_z_dir
                / f'power{rsd_str}_{keynames[i]}_tr_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
            )
            pk_tr_fns.append(
                save_z_dir
                / f'power_{keynames[i]}_tr_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
            )
            for j in range(len(keynames)):
                if i < j:
                    continue
                pk_rsd_ij_fns.append(
                    save_z_dir
                    / f'power{rsd_str}_{keynames[i]}_{keynames[j]}_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
                )
                pk_ij_fns.append(
                    save_z_dir
                    / f'power_{keynames[i]}_{keynames[j]}_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
                )

        if not load_presaved:
            # run version with rsd or without rsd
            for tr in mock_dict.keys():
                # obtain the positions
                tracer_pos = (
                    np.vstack(
                        (mock_dict[tr]['x'], mock_dict[tr]['y'], mock_dict[tr]['z'])
                    ).T
                ).astype(np.float32)
                del mock_dict
                gc.collect()

                pk_rsd_tr_fns = get_tracer_power(
                    tracer_pos,
                    config['HOD_params']['want_rsd'],
                    config,
                    save_3D_power=True,
                )
                del tracer_pos
                gc.collect()

            # run version without rsd if rsd was requested
            if config['HOD_params']['want_rsd']:
                mock_dict = self.run_hod(
                    self.tracers,
                    want_rsd=False,
                    reseed=None,
                    write_to_disk=False,
                    Nthread=16,
                    verbose=False,
                    fn_ext=None,
                )  # TODO: reseed
                for tr in mock_dict.keys():
                    # obtain the positions
                    tracer_pos = (
                        np.vstack(
                            (mock_dict[tr]['x'], mock_dict[tr]['y'], mock_dict[tr]['z'])
                        ).T
                    ).astype(np.float32)
                    del mock_dict
                    gc.collect()

                    pk_tr_fns = get_tracer_power(
                        tracer_pos, False, config, save_3D_power=True
                    )
                    del tracer_pos
                    gc.collect()
            else:
                pk_tr_fns, pk_ij_fns = None, None  # TODO: unsure

        # pass field names as a list to run_zcv
        pks = [pk_rsd_tr_fns, pk_rsd_ij_fns, pk_tr_fns, pk_ij_fns]
        for pk_fns in pks:
            if pk_fns is not None:
                for fn in pk_fns:
                    assert np.isclose(
                        asdf.open(fn)['header']['kcut'], config['zcv_params']['kcut']
                    ), f'Mismatching file: {str(fn)}'
        zcv_dict = run_zcv_field(
            pk_rsd_tr_fns, pk_rsd_ij_fns, pk_tr_fns, pk_ij_fns, config
        )

        # convert 3d power spectrum to correlation function multipoles
        r_bins = np.linspace(0.0, 200.0, 201)
        pk_rsd_tr_fns = [
            save_z_dir
            / f'power{rsd_str}_tr_tr_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
        ]  # TODO: same as other (could check that we have this if presaved)
        power_cv_tr_fn = (
            save_z_dir
            / f'power{rsd_str}_ZCV_tr_nmesh{config["zcv_params"]["nmesh"]:d}.asdf'
        )  # TODO: should be an output (could check that we have this if presaved; run_zcv too)
        r_binc, binned_poles_zcv, Npoles = pk_to_xi(
            asdf.open(power_cv_tr_fn)['data']['P_k3D_tr_tr_zcv'],
            self.lbox,
            r_bins,
            poles=config['power_params']['poles'],
        )
        r_binc, binned_poles, Npoles = pk_to_xi(
            asdf.open(pk_rsd_tr_fns[0])['data']['P_k3D_tr_tr'],
            self.lbox,
            r_bins,
            poles=config['power_params']['poles'],
        )
        zcv_dict['Xi_tr_tr_ell_zcv'] = binned_poles_zcv
        zcv_dict['Xi_tr_tr_ell'] = binned_poles
        zcv_dict['Np_tr_tr_ell'] = Npoles
        zcv_dict['r_binc'] = r_binc

        return zcv_dict

    def compute_wp(self, mock_dict, rpbins, pimax, pi_bin_size, Nthread=8):
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
                    continue  # cross-correlations are symmetric
                if i1 == i2:
                    print(tr1 + '_' + tr2)
                    clustering[tr1 + '_' + tr2] = calc_wp_fast(
                        x1, y1, z1, rpbins, pimax, self.lbox, Nthread
                    )
                else:
                    print(tr1 + '_' + tr2)
                    x2 = mock_dict[tr2]['x']
                    y2 = mock_dict[tr2]['y']
                    z2 = mock_dict[tr2]['z']
                    clustering[tr1 + '_' + tr2] = calc_wp_fast(
                        x1,
                        y1,
                        z1,
                        rpbins,
                        pimax,
                        self.lbox,
                        Nthread,
                        x2=x2,
                        y2=y2,
                        z2=z2,
                    )
                    clustering[tr2 + '_' + tr1] = clustering[tr1 + '_' + tr2]
        return clustering

    def gal_reader(
        self,
        output_dir=None,
        simname=None,
        sim_dir=None,
        z_mock=None,
        want_rsd=None,
        tracers=None,
    ):
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
            rsd_string = '_rsd'
        else:
            rsd_string = ''

        outdir = (self.mock_dir) / ('galaxies' + rsd_string)

        mockdict = {}
        for tracer in tracers:
            mockdict[tracer] = ascii.read(outdir / (tracer + 's.dat'))
        return mockdict


@njit(parallel=True)
def _searchsorted_parallel(a, b):
    res = np.empty(len(b), dtype=np.int64)
    for i in numba.prange(len(b)):
        res[i] = np.searchsorted(a, b[i])
    return res
