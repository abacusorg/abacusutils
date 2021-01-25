"""
The AbacusHOD module loads halo catalogs from the AbacusSummit 
simulations and outputs multi-tracer mock galaxy catalogs.  
The module defines one class, ``AbacusHOD``, hose constructor 
takes the path to the simulation volume, and a set of HOD 
parameters, and runs the ``staging`` function to compile the 
simulation halo catalog as a set of arrays that are saved on
memory. The ``run_hod`` function can then be called to 
generate galaxy catalogs. 

The output 
The galaxies can be written to disk by setting the 
``write_to_disk`` flag to ``True`` in the argument of 
``run_hod``. However, the I/O is slow and the``write_to_disk`` 
flag defaults to ``False``.

Beyond just loading the halo catalog files into memory, this
module performs a few other manipulations.  Many of the halo
catalog columns are stored in bit-packed formats (e.g.
floats are scaled to a ratio from 0 to 1 then packed in 16-bit
ints), so these columns are unpacked as they are loaded.

Furthermore, the halo catalogs for big simulations are divided
across a few dozen files.  These files are transparently loaded
into one monolithic Astropy table if one passes a directory
to ``CompaSOHaloCatalog``; to save memory by loading only one file,
pass just that file as the argument to ``CompaSOHaloCatalog``.

Importantly, because ASDF and Astropy tables are both column-
oriented, it can be much faster to load only the subset of
halo catalog columns that one needs, rather than all 60-odd
columns.  Use the ``fields`` argument to the ``CompaSOHaloCatalog``
constructor to specify a subset of fields to load.  Similarly, the
particles can be quite large, and one can use the ``load_subsamples``
argument to restrict the particles to the subset one needs.

Some brief examples and technical details about the halo catalog
layout are presented below, followed by the full module API.
Examples of using this module to work with AbacusSummit data can
be found on the AbacusSummit website here:
https://abacussummit.readthedocs.io


Short Example
=============
>>> from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
>>> # Load the RVs and PIDs for particle subsample A
>>> cat = CompaSOHaloCatalog('/storage/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.100', load_subsamples='A_all')
>>> print(cat.halos[:5])  # cat.halos is an Astropy Table, print the first 5 rows
   id    npstartA npstartB ... sigmavrad_L2com sigmavtan_L2com rvcirc_max_L2com
-------- -------- -------- ... --------------- --------------- ----------------
25000000        0        2 ...       0.9473971      0.96568024      0.042019103
25000001       11       12 ...      0.86480814       0.8435805      0.046611086
48000000       18       15 ...      0.66734606      0.68342227      0.033434115
58000000       31       18 ...      0.52170926       0.5387341      0.042292822
58001000       38       23 ...       0.4689916      0.40759262      0.034498636
>>> print(cat.halos['N','x_com'][:5])  # print the position and mass of the first 5 halos
  N         x_com [3]        
--- ------------------------
278 -998.88525 .. -972.95404
 45  -998.9751 .. -972.88416
101   -999.7485 .. -947.8377
 82    -998.904 .. -937.6313
 43   -999.3252 .. -937.5813
>>> # Examine the particle subsamples associated with the 5th halo
>>> h5 = cat.halos[4]
>>> print(cat.subsamples['pos'][h5['npstartA']:h5['npstartA'] + h5['npoutA']])
        pos [3]         
------------------------
  -999.3019 .. -937.5229
 -999.33435 .. -937.5515
-999.38965 .. -937.58777
>>> # At a glance, the pos fields match that of the 5th halo above, so it appears we have indexed correctly!


Catalog Structure
=================
The catalogs are stored in a directory structure that looks like:

.. code-block:: none

    - SimulationName/
        - halos/
            - z0.100/
                - halo_info/
                    halo_info_000.asdf
                    halo_info_001.asdf
                    ...
                - halo_rv_A/
                    halo_rv_A_000.asdf
                    halo_rv_A_001.asdf
                    ...
                - <field & halo, rv & PID, subsample A & B directories>
            - <other redshift directories, some with particle subsamples, others without>

The file numbering roughly corresponds to a planar chunk of the simulation
(all y and z for some range of x).  The matching of the halo_info file numbering
to the particle file numbering is important; halo_info files index into the
corresponding particle files.

The halo catalogs are stored on disk in ASDF files (https://asdf.readthedocs.io/).
The ASDF files start with a human-readable header that describes
the data present in the file and metadata about the simulation
from which it came (redshift, cosmology, etc).  The rest of
the file is binary blobs, each representing a column (i.e.
a halo property).

Internally, the ASDF binary portions are usually compressed.  This should
be transparent to users, although you may be prompted to install the
blosc package if it is not present.  Decompression should be fast,
at least 500 MB/s per core.


Particle Subsamples
===================
We define two disjoint sets of "subsample" particles, called "subsample A" and
"subsample B".  Subsample A is a few percent of all particles, with subsample B
a few times larger than that.  Particles membership in each group is a function
of PID and is thus consistent across redshift.

At most redshifts, we only output halo catalogs and halo subsample particle PIDs.
This aids with construction of merger trees.  At a few redshifts, we provide
subsample particle positions as well as PIDs, for both halo particles and
non-halo particles, called "field" particles.


Halo File Types
===============
Each file type (for halos, particles, etc) is grouped into a subdirectory.
These subdirectories are:

- ``halo_info/``
    The primary halo catalog files.  Contains stats like
    CoM positions and velocities and moments of the particles.
    Also indicates the index and count of subsampled particles in the
    ``halo_pid_A/B`` and ``halo_rv_A/B`` files.

- ``halo_pid_A/`` and ``halo_pid_B/``
    The 64-bit particle IDs of particle subsamples A and B.  The PIDs
    contain information about the Lagrangian position of the particles,
    whether they are tagged, and their local density.

The following subdirectories are only present for the redshifts for which
we output particle subsamples and not just halo catalogs:
    
- ``halo_rv_A/`` and ``halo_rv_B/``
    The positions and velocities of the halo subsample particles, in "RVint"
    format. The halo associations are recoverable with the indices in the
    ``halo_info`` files.

- ``field_rv_A/`` and ``field_rv_B/``
    Same as ``halo_rv_<A|B>/``, but only for the field (non-halo) particles.

- ``field_pid_A/`` and ``field_pid_B/``
    Same as ``halo_pid_<A|B>/``, but only for the field (non-halo) particles.


Bit-packed Formats
==================
The "RVint" format packs six fields (x,y,z, and vx,vy,vz) into three ints (12 bytes).
Positions are stored to 20 bits (global), and velocities 12 bits (max 6000 km/s).

The PIDs are 8 bytes and encode a local density estimate, tag bits for merger trees,
and a unique particle id, the last of which encodes the Lagrangian particle coordinate.

These are described in more detail on the :doc:`AbacusSummit Data Model page <summit:data-products>`.

Use the ``unpack_bits`` argument of the ``CompaSOHaloCatalog`` constructor to specify
which PID bit fields you want unpacked.  Be aware that some of them might use a lot of
memory; e.g. the Lagrangian positions are three 4-byte floats per subsample particle.
Also be aware that some of the returned arrays use narrow int dtypes to save memory,
such as the ``lagr_idx`` field using ``int16``.  It is easy to silently overflow such
narrow int types; make sure your operations stay within the type width and cast
if necessary.

Field Subset Loading
====================
Because the ASDF files are column-oriented, it is possible to load just one or a few
columns (halo catalog fields) rather than the whole file.  This can save huge amounts
of IO, memory, and CPU time (due to the decompression).  Use the ``fields`` argument
to the ``CompaSOHaloCatalog`` constructor to specify the list of columns you want.

In detail, some columns are stored as ratios to other columns.  For example, ``r90``
is stored as a ratio relative to ``r100``.  So to properly unpack
``r90``, the ``r100`` column must also be read.  ``CompaSOHaloCatalog`` knows about
these dependencies and will load the minimum set necessary to return the requested
columns to the user.  However, this may result in more IO than expected.  The ``verbose``
constructor flag or the ``dependency_info`` field of the ``CompaSOHaloCatalog``
object may be useful for diagnosing exactly what data is being loaded.

Despite the potential extra IO and CPU time, the extra memory usage is granular
at the level of individual files.  In other words, when loading multiple files,
the concatenated array will never be constructed for columns that only exist for
dependency purposes.

Multi-threaded Decompression
============================
The Blosc compression we use inside the ASDF files supports multi-threaded
decompression.  We have packed AbacusSummit files with 4 Blosc blocks (each ~few MB)
per ASDF block, so 4 Blosc threads is probably the optimal value.  This is the
default value, unless fewer cores are available (as determined by the process
affinity mask).

.. note::

    Loading a CompaSOHaloCatalog will use 4 decompression threads by default.

You can control the number of decompression threads with:

.. code-block:: python

    import asdf.compression
    asdf.compression.set_decompression_options(nthreads=N)
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
    def __init__(self, sim_params, HOD_params, power_params):

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

        # all paths relevant for mock generation
        scratch_dir = Path(self.scratch_dir)
        simname = Path(self.sim_name)
        sim_dir = Path(self.sim_dir)
        mock_dir = scratch_dir / simname / ('z%4.3f'%self.z_mock)
        # create mock_dir if not created
        if not mock_dir.exists():
            mock_dir.mkdir(parents = True)
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
        params['numchunks'] = len(halo_info_fns)
        self.lbox = header['BoxSize']
        
        # list holding individual chunks
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
        for echunk in range(params['numchunks']):
            print("Loading simulation by chunk, ", echunk)
            
            if 'ELG' not in self.tracers.keys() and 'QSO' not in self.tracers.keys():
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod'%echunk)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod'%echunk)
            else:
                halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod_MT'%echunk)
                particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod_MT'%echunk)            

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
            halo_pos = maskedhalos["x_com"] # halo positions, Mpc / h
            halo_vels = maskedhalos['v_com'] # halo velocities, km/s
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

            # #     part_data_chunk += [part_ranks, part_ranksv, part_ranksp, part_ranksr]
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

    
    def run_hod(self, tracers, want_rsd, write_to_disk = False):
        
        mock_dict = gen_gal_cat(self.halo_data, self.particle_data, self.tracers, self.params, enable_ranks = self.want_ranks, rsd = want_rsd, write_to_disk = write_to_disk, savedir = self.mock_dir)

        return mock_dict

    def compute_power(self, mock_dict, *args, **kwargs):
        if self.power_type == 'xirppi':
            power = self.compute_xirppi(mock_dict, *args, **kwargs)
        elif self.power_type == 'wp':
            power = self.compute_wp(mock_dict, *args, **kwargs)
        return power
    
    def compute_xirppi(self, mock_dict, rpbins, pimax, pi_bin_size, Nthread = 8):

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
                power_spectra[tr1+'_'+tr2] = calc_xirppi_fast(x1, y1, z1, rpbins, pimax, pi_bin_size, self.lbox, Nthread, x2 = x2, y2 = y2, z2 = z2)
                if i1 != i2: power_spectra[tr1+'_'+tr2] = power_spectra[tr2+'_'+tr1]
        return power_spectra

    def compute_wp(self, mock_dict, rpbins, pimax, pi_bin_size, Nthread = 8):

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
                if i1 != i2: power_spectra[tr1+'_'+tr2] = power_spectra[tr2+'_'+tr1]
        return power_spectra
