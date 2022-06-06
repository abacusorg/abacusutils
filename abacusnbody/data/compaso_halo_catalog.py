"""
The ``compaso_halo_catalog`` module loads halo catalogs from CompaSO, Abacus's
on-the-fly halo finder.  The module defines one class, ``CompaSOHaloCatalog``,
whose constructor takes the path to a halo catalog as an argument.
Users should use this class as the primary interface to load
and manipulate halo catalogs.

The halo catalogs and particle subsamples are stored on disk in
ASDF files and are loaded into memory as Astropy tables.  Each
column of an Astropy table is essentially a Numpy array and can
be accessed with familiar Numpy-like syntax.  More on Astropy
tables here: http://docs.astropy.org/en/stable/table/

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
particles can be quite large, and one can use the ``subsamples``
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
>>> cat = CompaSOHaloCatalog('/storage/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.100', subsamples=dict(A=True,pos=True))
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
non-halo particles, called "field" particles.  Only halo particles (specifically,
:doc:`L1 particles<summit:compaso>`) may be loaded through this module; field
particles and L0 halo particles can be loaded by reading the particle files
directly with :ref:`abacusnbody.data:read_abacus module`.

Use the ``subsamples`` argument to the constructor to specify loading
subsample A and/or B, and which fields---pos, vel, pid---to load.  Note that
if only one of pos & vel is specified, the IO amount is the same, because
the pos & vel are packed together in :ref:`RVint format<compaso:Bit-packed Formats>`.
But the memory usage and time to unpack will be lower.


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
of IO, memory, and CPU time (the latter due to the decompression).  Use the ``fields`` argument
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


Superslab (Chunk) Processing
============================
The halo catalogs are divided across multiple files, called "superslabs", which are
typically planar chunks of the simulation volume (all y,z for some range of x, with
a bit of overlap at the boundaries).  Applications that can process the volume
superslab-by-superslab can save a substantial amount of memory compared to loading
the full volume.  To load a single superslab, pass the corresponding ``halo_info_XXX.asdf``
file as the ``path`` argument:

.. code-block:: python

    cat = CompaSOHaloCatalog('AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_000.asdf')

If your application needs one slab of padding, you can pass a list of files and proceed in a
rolling fashion:

.. code-block:: python

    cat = CompaSOHaloCatalog(['AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_033.asdf',
                              'AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_000.asdf',
                              'AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_001.asdf'])


Superslab Filtering
===================
Another way to save memory is to use the ``filter_func`` argument.  This function
will be called for each superslab, and must return a mask representing the rows
to keep.  For example, to drop all halos with less than 100 particles, use:

.. code-block:: python
    
    cat = CompaSOHaloCatalog(..., filter_func=lambda h: h['N'] >= 100)
    
Because this mask is applied on each superslab immediately after loading,
the full, unfiltered table is never constructed, thus saving memory.

The filtering adds some CPU time, but in many cases loading catalogs is
IO limited, so this won't add much overhead.


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

    import abacusnbody.data.asdf
    abacusnbody.data.asdf.set_nthreads(N)
"""

from glob import glob
import os
import os.path
from pathlib import PurePath
from os.path import join as pjoin, dirname, basename, isdir, isfile, normpath, abspath, samefile
import re
import gc
import warnings
from time import perf_counter as timer

from collections import defaultdict

# Stop astropy from trying to download time data; nodes on some clusters are not allowed to access the internet directly
from astropy.utils import iers
iers.conf.auto_download = False

import numpy as np
import numba as nb
import astropy.table
from astropy.table import Table
import asdf

import asdf.compression
try:
    asdf.compression.validate('blsc')
except Exception as e:
    raise Exception("Abacus ASDF extension not properly loaded! Try reinstalling abacusutils, or updating ASDF: `pip install asdf>=2.8`") from e

from . import bitpacked

# Default to 4 decompression threads, or fewer if fewer cores are available
DEFAULT_BLOSC_THREADS = 4
DEFAULT_BLOSC_THREADS = max(1, min(len(os.sched_getaffinity(0)), DEFAULT_BLOSC_THREADS))
from . import asdf as _asdf
_asdf.set_nthreads(DEFAULT_BLOSC_THREADS)

class CompaSOHaloCatalog:
    """
    A halo catalog from Abacus's on-the-fly group finder.
    """

    # TODO: optional progress meter for loading files
    # TODO: generator mode over superslabs

    def __init__(self, path, cleaned=True, subsamples=False, convert_units=True, unpack_bits=False,
                 fields='DEFAULT_FIELDS', verbose=False, cleandir=None,
                 filter_func=None, halo_lc=None,
                 **kwargs):
        """
        Loads halos.  The ``halos`` field of this object will contain
        the halo records; and the ``subsamples`` field will contain
        the corresponding halo/field subsample positions and velocities and their
        ids (if requested via ``subsamples``).  The ``header`` field contains
        metadata about the simulation.

        Whether a particle is tagged or not is returned when loading the
        halo and field pids, as it is encoded for each in the 64-bit PID.
        The local density of the particle is also encoded in the PIDs
        and returned upon loading those.

        Parameters
        ----------
        path: path-like or list of path-like
            The halo catalog directory, like ``MySimulation/halos/z1.000/``.
            Or a single halo info file, or a list of halo info files.
            Will accept ``halo_info`` dirs or "redshift" dirs
            (e.g. ``z1.000/halo_info/`` or ``z1.000/``).
	    
	    .. note::
	    
                To load cleaned catalogs, you do *not* need to pass a different
		argument to the ``path`` directory.  Use ``cleaned=True`` instead
		and the path to the cleaning info will be detected automatically
		(or see ``cleandir``).

        cleaned: bool, optional
            Loads the "cleaned" version of the halo catalogues. Always recommended.
            Assumes there is a directory called ``cleaning/`` at the same level
            as the top-level simulation directory (or see ``cleandir``).
            Default: True.
            False returns the out-of-the-box CompaSO halos. May be useful for specific
            applications.

        subsamples: bool or dict, optional
            Load halo particle subsamples.  True or False may be specified
            to load all particles or none, or a dict to specify whether to
            load subsample A and/or B, with pos, vel, and/or pid fields:

            .. code-block:: none

                subsamples=dict(A=True, B=True, pos=True, vel=True, pid=True)

            The ``rv`` key may be used as shorthand to set both ``pos`` and ``vel``.
            False (the default) loads nothing.

        convert_units: bool, optional
            Convert positions from unit-box units to BoxSize-box units,
            velocities already come in km/s.  Default: True.

        unpack_bits: bool, or list of str, optional
            Extract information from the PID field of each subsample particle
            info about its Lagrangian position, whether it is tagged, and its
            current local density.  If False, only the particle ID part will
            be extracted.  Note that this per-particle information can be large.
            Can be a list of str, in which case only those fields will be unpacked.
            Field names are: ('pid', 'lagr_pos', 'tagged', 'density', 'lagr_idx').
            Default: False.

        fields: str or list of str, optional
            A list of field names/halo properties to load.  Selecting a small
            subset of fields can be substantially faster than loading all fields
            because the file IO will be limited to the desired fields.
            See ``compaso_halo_catalog.user_dt`` or the :doc:`AbacusSummit Data Model page <summit:data-products>`
            for a list of available fields. See ``compaso_halo_catalog.clean_dt`` for the list
            of cleaned halo fields that will be loaded. 'all' will also load main progenitor
            information, which could be slow.
            Default: 'DEFAULT_FIELDS'

        verbose: bool, optional
            Print informational messages. Default: False
            
        cleandir: str, optional
            Where the halo catalog cleaning files are located (usually called ``cleaning/``).
            Default of None will try to detect it automatically.  Only has any effect if
            using ``cleaned=True``.
            
        filter_func: function, optional
            A mask function to be applied to each superslab as it is loaded.  The function
            must take one argument (a halo table) and return a boolean array or similar
            mask on the rows. Simple lambda expressions are particularly useful here;
            for example, to load all halos with 100 particles or more, use:
            
            .. code-block:: python
            
                filter_func=lambda h: h['N'] >= 100
                
        halo_lc: bool or None, optional
            Whether the catalog is a halo light cone catalog, i.e. an output of the CompaSO
            halo light cone pipeline. Default of None means to detect based on the catalog path.
            
        """
        # Internally, we will use `load_subsamples` as the name of the `subsamples` arg to distinguish it from the `self.subsamples` table
        load_subsamples = subsamples
        del subsamples
        if 'load_subsamples' in kwargs:
            load_subsamples = kwargs.pop('load_subsamples')
            warnings.warn('`load_subsamples` argument is deprecated; use `subsamples`', FutureWarning)
        if 'cleaned_halos' in kwargs:
            cleaned = kwargs.pop('cleaned_halos')
            warnings.warn('`cleaned_halos` argument is deprecated; use `cleaned`', FutureWarning)
            
        # `cleaned` and `self.cleaned` mean slightly different things.
        # `cleaned` (local var) means to load the cleaning info files,
        # `self.cleaned` means the catalog incorporates cleaning info, either because the user
        # said `cleaned=True` or because this is a halo light cone catalog, which is already cleaned
        self.cleaned = cleaned
        
        if halo_lc == None:
            halo_lc = self.is_path_halo_lc(path)
            if verbose and halo_lc:
                print('Detected halo light cone catalog.')
        self.halo_lc = halo_lc

        # If loading halo light cones, turn off cleaning and bit unpacking because done already
        if halo_lc:
            if not self.cleaned:
                warnings.warn('`cleaned=False` was specified but halo light cones always incorporate cleaning')
            cleaned = False
            unpack_bits = False
            self.cleaned = True
            
        # Check no unknown args!
        if kwargs:
            raise ValueError(f'Unknown arguments to CompaSOHaloCatalog constructor: {list(kwargs)}')

        # Parse `path` to determine what files to read
        (self.groupdir,
         self.cleandir,
         self.superslab_inds,
         self.halo_fns,
         self.cleaned_halo_fns) = self._setup_file_paths(path, cleaned=cleaned, cleandir=cleandir, halo_lc=halo_lc)

        # Figure out what subsamples the user is asking us to loads
        (self.load_AB,
         self.load_pidrv,
         self.unpack_subsamples) = self._setup_load_subsamples(load_subsamples)
        del load_subsamples  # use the parsed values

        # If using halo light cones, only have subsample A available
        if halo_lc:
            if self.unpack_subsamples:
                self.load_AB = ['A']
        
        # Parse `fields` and `cleaned_fields` to determine halo catalog fields to read
        (self.fields,
         self.cleaned_fields) = self._setup_fields(fields, cleaned=cleaned, load_AB=self.load_AB, halo_lc=halo_lc)
        del fields  # use self.fields

        self.data_key = 'data'
        self.convert_units = convert_units  # let's save, user might want to check later
        self.verbose = verbose
        self.filter_func = filter_func

        unpack_bits = self._setup_unpack_bits(unpack_bits)

        # End parameter parsing, begin opening files

        # Open the first file, just to grab the header
        with asdf.open(self.halo_fns[0], lazy_load=True, copy_arrays=False) as af:
            # will also be available as self.halos.meta
            self.header = af['header']
            # For any applications that propagate the header, record whether they used cleaned halos
            self.header['cleaned_halos'] = self.cleaned

        # If we are using cleaned haloes, want to also grab header information regarding number of preceding timesteps
        if cleaned:
            with asdf.open(self.cleaned_halo_fns[0], lazy_load=True, copy_arrays=False) as af:
                self.header['TimeSliceRedshiftsPrev'] = af['header']['TimeSliceRedshiftsPrev']
                self.header['NumTimeSliceRedshiftsPrev'] = len(af['header']['TimeSliceRedshiftsPrev'])

        # Read and unpack the catalog into self.halos
        self._setup_halo_field_loaders()
        N_halo_per_file = self._read_halo_info(self.halo_fns, self.fields,
                                               cleaned_fns=self.cleaned_halo_fns, cleaned_fields=self.cleaned_fields,
                                               halo_lc=halo_lc,
                                              )

        self.subsamples = Table()  # empty table, to be filled with PIDs and RVs in the loading functions below

        self.numhalos = N_halo_per_file

        # reindex subsamples if this is an L1 redshift
        # halo subsamples have not yet been reindexed
        self._reindexed = {'A': False, 'B': False}

        self._reindexed_merge = {'A': False, 'B': False}

        if cleaned:
            self._updated_indices = {'A': False, 'B': False}

        # Updating subsamples indices needs to be done only once, and only right
        # at the end (i.e. after all pid/pos/vel/rv have been read)
        # Define `self.subsamples_to_load` as a way to track what still needs to be loaded
        self.subsamples_to_load = self.load_pidrv.copy()

        # Loading the particle information
        # B.H. unpack_pid
        if "pid" in self.load_pidrv:
            self.subsamples_to_load.remove('pid')
            self._load_pids(unpack_bits, N_halo_per_file, cleaned=cleaned, halo_lc=halo_lc)

        # B.H. unpacked_pos and vel
        if 'pos' in self.load_pidrv or 'vel' in self.load_pidrv:
            for k in ['pos','vel']:
                if k in self.subsamples_to_load:
                    self.subsamples_to_load.remove(k)

            # unpack_which = None will still read the RVs but will keep them in rvint
            unpack_which = self.load_pidrv if self.unpack_subsamples else None
            self._load_RVs(N_halo_per_file, cleaned=cleaned, unpack_which=unpack_which, halo_lc=halo_lc)

        # Don't need this any more
        del self.subsamples_to_load

        # If we're reading in cleaned haloes, N should be updated
        if cleaned:
            self.halos.rename_column('N_total', 'N')

        if verbose:
            print('\n'+str(self))
            
        gc.collect()


    def _setup_file_paths(self, path, cleaned=True, cleandir=None, halo_lc=False):
        '''Figure out what files the user is asking for
        '''

        # For the moment, coerce pathlib to str
        if isinstance(path, PurePath):
            path = str(path)
        if type(path) is str:
            path = [path]  # dir or file
        else:
            # if list, must be all files
            for p in path:
                if os.path.exists(p) and not isfile(p):
                    raise ValueError(f'If passing a list of paths, all paths must be files, not dirs. Path "{p}" is not a file.')

        for p in path:
            if not os.path.exists(p):
                raise FileNotFoundError(f'Path "{p}" does not exist!')

        path = [abspath(p) for p in path]

        # Allow users to pass halo_info dirs, even though redshift dirs remain canoncial
        for i,p in enumerate(path):
            if basename(p) == 'halo_info':
                path[i] = abspath(pjoin(p,os.pardir))

        # Can't mix files from different catalogs!
        if isfile(path[0]):
            groupdir = dirname(dirname(path[0]))
            if halo_lc:
                groupdir = dirname(path[0])
            for p in path:
                if not samefile(groupdir, dirname(dirname(p))) and not halo_lc:
                    raise ValueError("Can't mix files from different catalogs!")
                halo_fns = path  # path is list of one or more files

            for i,p in enumerate(path):
                for j,q in enumerate(path[i+1:]):
                    if samefile(p,q):
                        raise ValueError(f'Cannot pass duplicate halo_info files! Found duplicate "{p}" and at indices {i} and {i+j+1}')

        else:
            groupdir = path[0]  # path is a singlet of one dir
            if halo_lc:  # naming convention differs for the light cone catalogs
                globpat = pjoin(groupdir, 'lc_halo_info*.asdf')
            else:
                globpat = pjoin(groupdir, 'halo_info', 'halo_info_*')
            halo_fns = sorted(glob(globpat))
            if len(halo_fns) == 0:
                raise FileNotFoundError(f'No halo_info files found! Search pattern was: "{globpat}"')

        
        if halo_lc:  # halo light cones files aggregate all superslabs into a single file
            superslab_inds = np.array([0])
        else:
            superslab_inds = np.array([int(hfn.split('_')[-1].strip('.asdf')) for hfn in halo_fns])

        if cleaned:
            pathsplit = groupdir.split(os.path.sep)
            del pathsplit[-2]  # remove halos/, leaving .../SimName/z0.000
            s = -2
            if not cleandir:
                cleandir = os.path.sep + pjoin(*pathsplit[:s], 'cleaning')
                if not isdir(cleandir) and 'small' in pathsplit[-2]:
                    s = -3
                    cleandir_small = os.path.sep + pjoin(*pathsplit[:s], 'cleaning')
                    if not isdir(cleandir_small):
                        raise FileNotFoundError(f'Could not find cleaning info dir. Tried:\n"{cleandir}"\n"{cleandir_small}"\nTo load the uncleaned catalog, use `cleaned=False`.')
                    cleandir = cleandir_small
            
            cleandir = pjoin(cleandir, *pathsplit[s:])  # TODO ugly
            
            cleaned_halo_fns = [pjoin(cleandir, 'cleaned_halo_info', 'cleaned_halo_info_%03d.asdf'%(ext)) for ext in superslab_inds]
            
            for fn in cleaned_halo_fns:
                if not isfile(fn):
                    raise FileNotFoundError(f'Cleaning info not found. File path was: "{fn}". To load the uncleaned catalog, use `cleaned=False`.')
            
        else:
            cleandir = None
            cleaned_halo_fns = []

        return groupdir, cleandir, superslab_inds, halo_fns, cleaned_halo_fns


    def _setup_unpack_bits(self, unpack_bits):
        # validate unpack_bits
        if type(unpack_bits) is str:
            unpack_bits = [unpack_bits]
        if unpack_bits not in (True,False):
            try:
                for _f in unpack_bits:
                    assert _f in bitpacked.PID_FIELDS
            except:
                raise ValueError(f'`unpack_bits` must be True, False, or one of: "{bitpacked.PID_FIELDS}"')
        return unpack_bits


    def _setup_load_subsamples(self, load_subsamples):
        '''
        Figure out if the user wants A, B, pid, pos, vel.
        Will be returned as lists of strings in `load_AB` and `load_pidrv`.
        `unpack_subsamples` is for pipelining, to keep things in rvint.
        '''
        if load_subsamples == False:
            # stub
            load_AB = []
            load_pidrv = []
            unpack_subsamples = True
        else:
            # If user has not specified which subsamples, then assume user wants to load everything
            if load_subsamples == True:
                load_subsamples = dict(A=True, B=True, rv=True, pid=True)

            if type(load_subsamples) == dict:
                load_AB = [k for k in 'AB' if load_subsamples.get(k)]  # ['A', 'B']

                # Check for conflicts between rv, pos, vel. Must be done before list-ifying to distinguish False and not given.
                if 'rv' in load_subsamples:
                    if 'pos' in load_subsamples or 'vel' in load_subsamples:
                        raise ValueError('Cannot pass `rv` and `pos` or `vel` in `load_subsamples`.')

                load_pidrv = [k for k in load_subsamples if k in ('pid','pos','vel','rv') and load_subsamples.get(k)]  # ['pid', 'pos', 'vel']

                unpack_subsamples = load_subsamples.pop('unpack',True)

                # set some intelligent defaults
                if load_pidrv and not load_AB:
                    warnings.warn(f'Loading of {load_pidrv} was requested but neither subsample A nor B was specified. Assuming subsample A. Can specify with `load_subsamples=dict(A=True)`.')
                    load_AB = ['A']
                elif not load_pidrv and load_AB:
                    if load_subsamples.get('pos') is not False:
                        load_pidrv += ['pos']
                    if load_subsamples.get('vel') is not False:
                        load_pidrv += ['vel']
                    if not load_pidrv:
                        warnings.warn(f'Loading of subsample {load_AB} was requested but none of `pos`, `vel`, `rv`, `pid` was specified. Assuming `rv`. Can specify with `load_subsamples=dict(rv=True)`.')
                        load_pidrv = ['rv']

                if load_subsamples.pop('field',False):
                    raise ValueError('Loading field particles through CompaSOHaloCatalog is not supported. Read the particle files directly with `abacusnbody.data.read_abacus.read_asdf()`.')

                # Pop all known keys, so if anything is left, that's an error!
                for k in ['A', 'B', 'rv', 'pid', 'pos', 'vel', 'unpack']:
                    load_subsamples.pop(k,None)

                if load_subsamples:
                    raise ValueError(f'Unrecognized keys in `load_subsamples`: {list(load_subsamples)}')

            elif type(load_subsamples) == str:
                # This section is deprecated, will remove in mid-2021
                warnings.warn('Passing a string to `load_subsamples` is deprecated; use a dict instead, like: `load_subsamples=dict(A=True, rv=True)`', FutureWarning)

                # Validate the user's `load_subsamples` option and figure out what subsamples we need to load
                subsamp_match = re.fullmatch(r'(?P<AB>(A|B|AB))(_(?P<hf>halo|field))?_(?P<pidrv>all|pid|rv)', load_subsamples)
                if not subsamp_match:
                    raise ValueError(f'Value "{load_subsamples}" for argument `load_subsamples` not understood')
                load_AB = subsamp_match.group('AB')
                load_halofield = subsamp_match.group('hf')
                load_halofield = [load_halofield] if load_halofield else ['halo','field']  # default is both
                load_pidrv = subsamp_match.group('pidrv')
                load_pidrv = subsamp_match.group('pidrv')
                if load_pidrv == 'all':
                    load_pidrv = ['pid','rv']
                if type(load_pidrv) == str:
                    # Turn this into a list so that the .remove() operation below doesn't complain
                    load_pidrv = [load_pidrv]
                if 'field' in load_halofield:
                    raise ValueError('Loading field particles through CompaSOHaloCatalog is not supported. Read the particle files directly with `abacusnbody.data.read_abacus.read_asdf()`.')
                unpack_subsamples = True

        if 'rv' in load_pidrv:
            load_pidrv.remove('rv')
            load_pidrv += ['pos', 'vel']

        return load_AB, load_pidrv, unpack_subsamples


    def _setup_fields(self, fields, cleaned=True, load_AB=[], halo_lc=False):
        '''Determine the halo catalog fields to load based on user input
        '''

        if fields == 'DEFAULT_FIELDS':
            fields = list(user_dt.names)
            if cleaned:
                fields += list(clean_dt.names)
            if halo_lc:
                fields += list(halo_lc_dt.names)
        if fields == 'all':
            fields = list(user_dt.names)
            if cleaned:
                fields += list(clean_dt_progen.names)
            if halo_lc:
                fields += list(halo_lc_dt.names)

        if type(fields) == str:
            fields = [fields]
        # Convert any other iter, like tuple
        fields = list(fields)

        # Minimum requirement for cleaned haloes
        if cleaned:
            # If we load cleaned, 'N' no longer has meaning
            if 'N' in fields:
                fields.remove('N')
            if 'N_total' not in fields:
                fields += ['N_total']

        # Let's split `fields` so that there is a separate set of `cleaned_fields`
        cleaned_fields = []
        if cleaned:
            for item in list(clean_dt_progen.names):
                if item in fields:
                    fields.remove(item)
                    cleaned_fields += [item]

        # B.H. Remove fields that are not recorded for the light cone catalogs
        if halo_lc:
            for item in list(fields):
                # TODO: this will silently drop misspellings
                if 'L2' not in item and item not in halo_lc_dt.names:
                    fields.remove(item)
        
        if cleaned:
            # If the user has not asked to load npstart{AB}_merge columns, we need to do so ourselves for indexing
            for AB in load_AB:
                if 'npstart'+AB not in fields:
                    fields += ['npstart'+AB]
                if 'npout'+AB not in fields:
                    fields += ['npout'+AB]
                if 'npstart'+AB+'_merge' not in cleaned_fields:
                    cleaned_fields += ['npstart'+AB+'_merge']
                if 'npout'+AB+'_merge' not in cleaned_fields:
                    cleaned_fields += ['npout'+AB+'_merge']

        return fields, cleaned_fields


    def _read_halo_info(self, halo_fns, fields, cleaned_fns=None, cleaned_fields=None, halo_lc=False):
        if not cleaned_fields:
            cleaned_fields = []
        if not cleaned_fns:
            cleaned_fns = []
        else:
            assert len(cleaned_fns) == len(halo_fns)
            
        # Open all the files, validate them, and count the halos
        # Lazy load, but don't use mmap
        afs = [asdf.open(hfn, lazy_load=True, copy_arrays=True) for hfn in halo_fns]
        cleaned_afs = [asdf.open(hfn, lazy_load=True, copy_arrays=True) for hfn in cleaned_fns]

        N_halo_per_file = np.array([len(af[self.data_key][list(af[self.data_key].keys())[0]]) for af in afs])
        for _N,caf in zip(N_halo_per_file,cleaned_afs):
            assert len(caf[self.data_key][next(iter(caf[self.data_key]))]) == _N  # check cleaned/regular file consistency

        N_halos = N_halo_per_file.sum()

        # Make an empty table for the concatenated, unpacked values
        # Note that np.empty is being smart here and creating 2D arrays when the dtype is a vector
        
        cols = {}
        for col in fields:
            if col in halo_lc_dt.names:
                cols[col] = np.empty(N_halos, dtype=halo_lc_dt[col])
            else:
                cols[col] = np.empty(N_halos, dtype=user_dt[col])
        for col in cleaned_fields:
            cols[col] = np.empty(N_halos, dtype=clean_dt_progen[col])
        all_fields = fields + cleaned_fields
        
        # Figure out what raw columns we need to read based on the fields the user requested
        # TODO: provide option to drop un-requested columns
        raw_dependencies, fields_with_deps, extra_fields = self._get_halo_fields_dependencies(all_fields)
        # save for informational purposes
        if not hasattr(self, 'dependency_info'):
            self.dependency_info = defaultdict(list)
        self.dependency_info['raw_dependencies'] += raw_dependencies
        self.dependency_info['fields_with_deps'] += fields_with_deps
        self.dependency_info['extra_fields'] += extra_fields

        if self.verbose:
            # TODO: going to be repeated in output
            print(f'{len(fields)} halo catalog fields ({len(cleaned_fields)} cleaned) requested. '
                f'Reading {len(raw_dependencies)} fields from disk. '
                f'Computing {len(extra_fields)} intermediate fields.')
            if self.halo_lc:
                print('\nFor more information on the halo light cone catalog fields, see https://abacussummit.readthedocs.io/en/latest/data-products.html#halo-light-cone-catalogs')

        self.halos = Table(cols, copy=False)
        self.halos.meta.update(self.header)

         # If we're loading main progenitor info, do this:

        # TODO: this shows the limits of querying the types from a numpy dtype, should query from a function
        r = re.compile('.*mainprog')
        prog_fields = list(filter(r.match, cleaned_fields))
        for fields in prog_fields:
            if fields in ['v_L2com_mainprog', 'haloindex_mainprog']:
                continue
            else:
                self.halos.replace_column(fields, np.empty(N_halos, dtype=(clean_dt_progen[fields], self.header['NumTimeSliceRedshiftsPrev'])))

        # Unpack the cats into the concatenated array
        # The writes would probably be more efficient if the outer loop was over column
        # and the inner was over cats, but wow that would be ugly
        N_written = 0
        for i,af in enumerate(afs):
            caf = cleaned_afs[i] if cleaned_afs else None
            
            # This is where the IO on the raw columns happens
            # There are some fields that we'd prefer to directly read into the concatenated table,
            # but ASDF doesn't presently support that, so this is the best we can do
            rawhalos = {}
            for field in raw_dependencies:
                src = caf if field in clean_dt_progen.names else af
                rawhalos[field] = src[self.data_key][field]
            rawhalos = Table(data=rawhalos, copy=False)
            af.close()
            if caf:
                caf.close()

            # `halos` will be a "pointer" to the next open space in the master table
            halos = self.halos[N_written:N_written+len(rawhalos)]

            # For temporary (extra) columns, only need to construct the per-file version
            for field in extra_fields:
                src = clean_dt_progen if field in clean_dt_progen.names else user_dt
                halos.add_column(np.empty(len(rawhalos), dtype=src[col]), name=field, copy=False)
                # halos[field][:] = np.nan  # for debugging

            loaded_fields = []
            for field in fields_with_deps:
                if field in loaded_fields:
                    continue
                loaded_fields += self._load_halo_field(halos, rawhalos, field)
            
            if self.filter_func:
                # N_total from the cleaning replaces N. For filtering purposes, allow the user to use 'N'
                halos.rename_column('N_total', 'N')
                
                mask = self.filter_func(halos)
                nmask = mask.sum()
                halos[:nmask] = halos[mask]
                del mask
                N_superslab = nmask
            else:
                N_superslab = len(halos)
            N_written += N_superslab
            N_halo_per_file[i] = N_superslab

            del halos, rawhalos
            del af, caf, src
            afs[i] = None
            if cleaned_afs:
                cleaned_afs[i] = None
            gc.collect()
            
        # Now the filtered length
        self.halos = self.halos[:N_written]
        if N_written < N_halos:
            # Release virtual memory if we didn't fill the whole allocation
            for col in cols:
                s = list(cols[col].shape)
                s[0] = N_written
                oldaddr = cols[col].ctypes.data
                cols[col].resize(s, refcheck=False)
                if cols[col].ctypes.data != oldaddr:
                    warnings.warn('Resize resulted in copy')
        N_halos = len(self.halos)

        return N_halo_per_file


    def _setup_halo_field_loaders(self):
        # Loaders is a dict of regex -> lambda
        # The lambda is responsible for unpacking the rawhalos field
        # The first regex that matches will be used, so they must be precise
        self.halo_field_loaders = {}

        if self.convert_units:
            box = self.header['BoxSize']
            # TODO: correct velocity units? There is an earlier comment claiming that velocities are already in km/s
            zspace_to_kms = self.header['VelZSpace_to_kms']
        else:
            box = 1.
            zspace_to_kms = 1.

        # The first argument to the following lambdas is the match object from re.match()
        # We will use m[0] to access the full match (i.e. the full field name)
        # Other indices, like m['com'], will access the sub-match with that group name

        # r10,r25,r33,r50,r67,r75,r90,r95,r98
        pat = re.compile(r'(?:r\d{1,2}|rvcirc_max)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]+'_i16']*raw['r100'+m['com']]/INT16SCALE*box

        # sigmavMin, sigmavMaj, sigmavrad, sigmavtan
        pat = re.compile(r'(?P<stem>sigmav(?:Min|Maj|rad|tan))(?P<com>_(?:L2)?com)')
        def _sigmav_loader(m,raw,halos):
            stem = m['stem'].replace('Maj','Max')
            return raw[stem+'_to_sigmav3d'+m['com']+'_i16']*raw['sigmav3d'+m['com']]/INT16SCALE*box
        self.halo_field_loaders[pat] = _sigmav_loader

        # sigmavMid
        pat = re.compile(r'sigmavMid(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: np.sqrt(raw['sigmav3d'+m['com']]*raw['sigmav3d'+m['com']]*box**2 \
                                                            - halos['sigmavMaj'+m['com']]**2 - halos['sigmavMin'+m['com']]**2)

        # sigmar
        pat = re.compile(r'sigmar(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]+'_i16']*raw['r100'+m['com']].reshape(-1,1)/INT16SCALE*box

        # sigman
        pat = re.compile(r'sigman(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]+'_i16']/INT16SCALE*box

        # x,r100 (box-scaled fields)
        pat = re.compile(r'(x|r100)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]*box

        # v,sigmav,sigmav3d,meanSpeed,sigmav3d_r50,meanSpeed_r50,vcirc_max (vel-scaled fields)
        pat = re.compile(r'(v|sigmav3d|meanSpeed|sigmav3d_r50|meanSpeed_r50|vcirc_max)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]*zspace_to_kms

        # id,npstartA,npstartB,npoutA,npoutB,ntaggedA,ntaggedB,N,L2_N,L0_N (raw/passthrough fields)
        # If ASDF could read into a user-provided array, could avoid these copies
        pat = re.compile(r'id|npstartA|npstartB|npoutA|npoutB|ntaggedA|ntaggedB|N|L2_N|L0_N|N_total|N_merge|npstartA_merge|npstartB_merge|npoutA_merge|npoutB_merge|npoutA_L0L1|npoutB_L0L1|is_merged_to|N_mainprog|vcirc_max_L2com_mainprog|sigmav3d_L2com_mainprog|haloindex|haloindex_mainprog|v_L2com_mainprog')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]

        # SO_central_particle,SO_radius (and _L2max) (box-scaled fields)
        pat = re.compile(r'SO(?:_L2max)?(?:_central_particle|_radius)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]*box

        # SO_central_density (and _L2max)
        pat = re.compile(r'SO(?:_L2max)?(?:_central_density)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]

        # loader for halo light cone catalog specific fields
        pat = re.compile(r'index_halo|pos_avg|vel_avg|redshift_interp|N_interp')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]

        # loader for halo light cone catalog field `origin`
        pat = re.compile(r'origin')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]%3

        # loader for halo light cone catalog fields: interpolated position and velocity        
        pat = re.compile(r'(?P<pv>pos|vel)_interp')
        def lc_interp_loader(m, raw, halos):
            columns = {}
            interped = (raw['origin'] // 3).astype(bool)
            if m[0] == 'pos_interp' or 'pos_interp' in halos.colnames:
                columns['pos_interp'] = np.where(interped[:, None], raw['pos_avg'], raw['pos_interp'])
            if m[0] == 'vel_interp' or 'vel_interp' in halos.colnames:
                columns['vel_interp'] = np.where(interped[:, None], raw['vel_avg'], raw['vel_interp'])
            return columns

        self.halo_field_loaders[pat] = lc_interp_loader
        
        # eigvecs loader
        pat = re.compile(r'(?P<rnv>sigma(?:r|n|v)_eigenvecs)(?P<which>Min|Mid|Maj)(?P<com>_(?:L2)?com)')
        def eigvecs_loader(m,raw,halos):
            minor,middle,major = unpack_euler16(raw[m['rnv']+m['com']+'_u16'])
            columns = {}

            minor_field = m['rnv'] + 'Min' + m['com']
            if minor_field in halos.colnames:
                columns[minor_field] = minor
            middle_field = m['rnv'] + 'Mid' + m['com']
            if middle_field in halos.colnames:
                columns[middle_field] = middle
            major_field = m['rnv'] + 'Maj' + m['com']
            if major_field in halos.colnames:
                columns[major_field] = major

            return columns

        self.halo_field_loaders[pat] = eigvecs_loader


    def _get_halo_fields_dependencies(self, fields):
        '''Each of the loaders accesses some raw columns on disk to
        produce the user-facing halo catalog columns. This function
        will determine which of those raw fields needs to be read
        by calling the loader functions with a dummy object that
        records field accesses.
        '''

        # TODO: define pre-set subsets of common fields

        class DepCapture:
            def __init__(self):
                self.keys = []
                self.colnames = []
            def __getitem__(self,key):
                self.keys += [key]
                return np.ones(1)  # a safe numeric value

        iter_fields = list(fields)  # make a copy

        raw_dependencies = []
        field_dependencies = []
        for field in iter_fields:
            have_match = False
            for pat in self.halo_field_loaders:
                match = pat.fullmatch(field)
                if match:
                    if have_match:
                        raise KeyError(f'Found more than one way to load field "{field}"')
                    capturer,raw_capturer = DepCapture(),DepCapture()
                    self.halo_field_loaders[pat](match,raw_capturer,capturer)
                    raw_dependencies += raw_capturer.keys

                    # these are fields of `halos`
                    for k in capturer.keys:
                        # Add fields regardless of whether they have already been encountered
                        iter_fields += [k]
                        if k not in fields:
                            field_dependencies += [k]
                    have_match = True
                    #break  # comment out for debugging
            else:
                if not have_match:
                    raise KeyError(f"Don't know how to load halo field \"{field}\"")

        raw_dependencies = list(set(raw_dependencies))  # make unique
        # unique, preserve order, but using last occurrence
        # because nested dependencies need to be loaded in reverse order
        fields_with_deps = list(dict.fromkeys(iter_fields[::-1]))
        field_deps = list(dict.fromkeys(field_dependencies[::-1]))

        # All raw dependencies for all user-requested fields
        return raw_dependencies, fields_with_deps, field_deps


    def _load_halo_field(self, halos, rawhalos, field):
        # TODO: attach units to all these?

        # We must use the halos['field'][:] syntax in order to do an in-place update
        # We will enable column replacement warnings to make sure we don't make a mistake
        # Remember that "halos" here is a view into the self.halos table
        _oldwarn = astropy.table.conf.replace_warnings
        astropy.table.conf.replace_warnings = ['always']

        # Look for the loader for this field, should only match one
        have_match = False
        loaded_fields = []
        for pat in self.halo_field_loaders:
            match = pat.fullmatch(field)
            if match:
                if have_match:
                    raise KeyError(f'Found more than one way to load field "{field}"')
                column = self.halo_field_loaders[pat](match,rawhalos,halos)

                # The loader is allowed to return a dict if it incidentally loaded multiple columns
                if type(column) == dict:
                    assert field in column
                    for k in column:
                        halos[k][:] = column[k]
                    loaded_fields += list(column)
                else:
                    halos[field][:] = column
                    loaded_fields += [field]

                have_match = True
                #break  # comment out for debugging
        else:
            if not have_match:
                raise KeyError(f"Don't know how to load halo field \"{field}\"")

        astropy.table.conf.replace_warnings = _oldwarn

        return loaded_fields


    def _reindex_subsamples(self, RVorPID, N_halo_per_file, cleaned=True, halo_lc=False):
        # TODO: this whole function probably goes away.
        # The algorithm ought to be: load concatenated L1 table using original indices, then fix indices

        if RVorPID == 'pid':
            asdf_col_name = 'packedpid'
            if halo_lc:
                asdf_col_name = 'pid' # because unpacked already
        elif RVorPID == 'rv':
            asdf_col_name = 'rvint'
            if halo_lc:
                asdf_col_name = 'pos'
        else:
            raise ValueError(RVorPID)

        particle_AB_afs = []  # ASDF file handles for A+B
        np_total = 0
        np_per_file = []

        particle_AB_merge_afs = []
        np_total_merge = 0
        np_per_file_merge = []
        key_to_read = []

        for AB in self.load_AB:
            # Open the ASDF file handles so we can query the size
            if halo_lc:
                particle_afs = [asdf.open(pjoin(self.groupdir, 'lc_pid_rv.asdf'), lazy_load=True, copy_arrays=True)]
            else:
                particle_afs = [asdf.open(pjoin(self.groupdir, f'halo_{RVorPID}_{AB}', f'halo_{RVorPID}_{AB}_{i:03d}.asdf'), lazy_load=True, copy_arrays=True)
                                    for i in self.superslab_inds]
            if cleaned:
                particle_merge_afs = [asdf.open(pjoin(self.cleandir,  'cleaned_rvpid', f'cleaned_rvpid_{i:03d}.asdf'), lazy_load=True, copy_arrays=True) for i in self.superslab_inds]

            # Should have same number of files (1st subsample; 2nd L1), but note that empty slabs don't get files
            # TODO: double-check this assert
            assert len(N_halo_per_file) <= len(particle_afs)

            if cleaned:
                assert len(N_halo_per_file) == len(particle_merge_afs)

            if not self._reindexed[AB] and 'npstart'+AB in self.halos.colnames:
                # Offset npstartB in case the user is loading both subsample A and B
                self.halos['npstart'+AB] += np_total
                _reindex_subsamples_from_asdf_size(self.halos['npstart'+AB],
                                                  [af[self.data_key][asdf_col_name] for af in particle_afs],
                                                  N_halo_per_file)
                self._reindexed[AB] = True

            if cleaned:
                if not self._reindexed_merge[AB] and 'npstart'+AB+'_merge' in self.halos.colnames:
                    mask = (self.halos['N_total'] == 0)
                    self.halos['npstart'+AB+'_merge'] += np_total_merge
                    _reindex_subsamples_from_asdf_size(self.halos['npstart'+AB+'_merge'],
                                                        [af[self.data_key][asdf_col_name+'_'+AB] for af in particle_merge_afs], N_halo_per_file)
                    self.halos['npstart'+AB+'_merge'][mask] = -999
                    self._reindexed_merge[AB] = True

            # total number of particles
            for af in particle_afs:
                np_per_file += [len(af[self.data_key][asdf_col_name])]
            np_total = np.sum(np_per_file)
            particle_AB_afs += particle_afs

            if cleaned:
                for af in particle_merge_afs:
                    np_per_file_merge += [len(af[self.data_key][asdf_col_name+'_'+AB])]
                    np_total_merge = np.sum(np_per_file_merge)
                    key_to_read += [asdf_col_name+'_'+AB]
                particle_AB_merge_afs += particle_merge_afs

        if cleaned:
            np_per_file_merge = np.array(np_per_file_merge)

        np_per_file = np.array(np_per_file)

        particle_dict = {'particle_AB_afs': particle_AB_afs,
	                 'np_per_file': np_per_file,
                         'particle_AB_merge_afs': particle_AB_merge_afs,
                         'np_per_file_merge': np_per_file_merge,
                         'key_to_read': key_to_read}

        return particle_dict


    def _load_pids(self, unpack_bits, N_halo_per_file, cleaned=True, check_pids=False, halo_lc=False):
        # Even if unpack_bits is False, return the PID-masked value, not the raw value.

        particle_dict = self._reindex_subsamples('pid', N_halo_per_file, cleaned=cleaned, halo_lc=halo_lc)
        
        pid_AB_afs = particle_dict['particle_AB_afs']
        np_per_file = particle_dict['np_per_file']
        pid_AB_merge_afs = particle_dict['particle_AB_merge_afs']
        np_per_file_merge = particle_dict['np_per_file_merge']
        key_to_read = particle_dict['key_to_read']

        # If loading light cones, can skip the rest
        if halo_lc:
            self.subsamples.add_column(pid_AB_afs[0][self.data_key]['pid'], name='pid', copy=False)
            return
        
        start = 0
        np_total = np.sum(np_per_file)
        pids_AB = np.empty(np_total, dtype=np.uint64)

        if cleaned:
            start_merge = 0
            np_total_merge = np.sum(np_per_file_merge)
            pids_AB_merge = np.empty(np_total_merge, dtype=np.uint64)

        for i,af in enumerate(pid_AB_afs):
            thisnp = np_per_file[i]
            if not unpack_bits:
                pids_AB[start:start+thisnp] = af[self.data_key]['packedpid'] & bitpacked.AUXPID
            else:
                pids_AB[start:start+thisnp] = af[self.data_key]['packedpid']
            start += thisnp

        if cleaned:
            for i,af in enumerate(pid_AB_merge_afs):
                thisnp = np_per_file_merge[i]
                key = key_to_read[i]
                if not unpack_bits:
                    pids_AB_merge[start_merge:start_merge+thisnp] = af[self.data_key][key] & bitpacked.AUXPID
                else:
                    pids_AB_merge[start_merge:start_merge+thisnp] = af[self.data_key][key]
                start_merge += thisnp


        # Could be expensive!  Off by default.  Probably faster ways to implement this.
        if check_pids:
            assert len(np.unique(pids_AB)) == len(pids_AB)
            if cleaned:
                assert len(np.unique(pids_AB_merge)) == len(pids_AB_merge)

        # Join subsample arrays here
        if cleaned:
            offset = 0
            total_particles = len(pids_AB) + len(pids_AB_merge)
            pids_AB_total = np.empty(total_particles, dtype=np.uint64)
            for AB in self.load_AB:

                start_indices = self.halos[f'npstart{AB}']
                start_indices_merge = self.halos[f'npstart{AB}_merge']
                nump_indices = self.halos[f'npout{AB}']
                nump_indices_merge = self.halos[f'npout{AB}_merge']
                pids_AB_total, npstart_updated, offset = join_arrays(offset, pids_AB, pids_AB_merge, pids_AB_total, start_indices, nump_indices, start_indices_merge, nump_indices_merge, self.halos['N_total'])

                if not self.subsamples_to_load and not self._updated_indices[AB]:
                    self.halos[f'npstart{AB}'] = npstart_updated
                    self.halos[f'npout{AB}'] += self.halos[f'npout{AB}_merge']
                    mask = (self.halos['N_total'] == 0)
                    self.halos[f'npout{AB}'][mask] = 0
                    self._updated_indices[AB] = True
                    self.halos.remove_column(f'npout{AB}_merge')
                    self.halos.remove_column(f'npstart{AB}_merge')

            pids_AB_total = pids_AB_total[:offset]

        else:

            pids_AB_total = pids_AB

        if unpack_bits:  # anything to unpack?
            # TODO: eventually, unpacking could be done on each file to save memory, as we do with the rvint
            if unpack_bits is True:
                unpack_which = {_f:True for _f in bitpacked.PID_FIELDS}
            else:  # truthy but not True, like a list
                unpack_which = {_f:True for _f in unpack_bits}

            # unpack_pids will do unit conversion if requested
            unpackbox = self.header['BoxSize'] if self.convert_units else 1.

            unpacked_arrays = bitpacked.unpack_pids(pids_AB_total, box=unpackbox, ppd=self.header['ppd'], **unpack_which)

            for name in unpacked_arrays:
                self.subsamples.add_column(unpacked_arrays[name], name=name, copy=False)
        else:
            self.subsamples.add_column(pids_AB_total, name='pid', copy=False)


    def _load_RVs(self, N_halo_per_file, cleaned=True, unpack_which=['pos','vel'], halo_lc=False):

        particle_dict = self._reindex_subsamples('rv', N_halo_per_file, cleaned=cleaned, halo_lc=halo_lc)

        particle_AB_afs = particle_dict['particle_AB_afs']
        np_per_file = particle_dict['np_per_file']
        particle_AB_merge_afs = particle_dict['particle_AB_merge_afs']
        np_per_file_merge = particle_dict['np_per_file_merge']
        key_to_read = particle_dict['key_to_read']

        # If loading light cones, can skip the rest
        if halo_lc:
            self.subsamples.add_column(particle_AB_afs[0][self.data_key]['pos'], name='pos', copy=False)
            self.subsamples.add_column(particle_AB_afs[0][self.data_key]['vel'], name='vel', copy=False)
            return
        
        start = 0
        np_total = np.sum(np_per_file)
        particles_AB = np.empty((np_total,3),dtype=np.int32)

        if cleaned:
            start_merge = 0
            np_total_merge = np.sum(np_per_file_merge)
            particles_AB_merge = np.empty((np_total_merge,3),dtype=np.int32)

        for i,af in enumerate(particle_AB_afs):
            thisnp = np_per_file[i]
            # TODO: don't concatenate here, then unpack. Unpack into the concatenated array below.
            particles_AB[start:start+thisnp] = af[self.data_key]['rvint']
            start += thisnp

        if cleaned:
            for i,af in enumerate(particle_AB_merge_afs):
                thisnp = np_per_file_merge[i]
                key = key_to_read[i]
                particles_AB_merge[start_merge:start_merge+thisnp] = af[self.data_key][key]
                start_merge += thisnp

        # Join subsample arrays here
        if cleaned:
            offset = 0
            total_particles = len(particles_AB) + len(particles_AB_merge)
            particles_AB_total = np.empty((total_particles, 3), dtype=np.int32)
            for AB in self.load_AB:

                start_indices = self.halos[f'npstart{AB}']
                start_indices_merge = self.halos[f'npstart{AB}_merge']
                nump_indices = self.halos[f'npout{AB}']
                nump_indices_merge = self.halos[f'npout{AB}_merge']
                particles_AB_total, npstart_updated, offset = join_arrays(offset, particles_AB, particles_AB_merge, particles_AB_total, start_indices, nump_indices, start_indices_merge, nump_indices_merge, self.halos['N_total'])

                if not self.subsamples_to_load and not self._updated_indices[AB] :
                    self.halos[f'npstart{AB}'] = npstart_updated
                    self.halos[f'npout{AB}'] += self.halos[f'npout{AB}_merge']
                    mask = (self.halos['N_total'] == 0)
                    self.halos[f'npout{AB}'][mask] = 0
                    self._updated_indices[AB] = True
                    self.halos.remove_column(f'npout{AB}_merge')
                    self.halos.remove_column(f'npstart{AB}_merge')
            particles_AB_total = particles_AB_total[:offset]

        else:
            particles_AB_total = particles_AB

        if unpack_which:
            unpackbox = self.header['BoxSize'] if self.convert_units else 1.

            _out = {}
            _out['posout'] = None if 'pos' in unpack_which else False
            _out['velout'] = None if 'vel' in unpack_which else False

            ppos_AB, pvel_AB = bitpacked.unpack_rvint(particles_AB_total, unpackbox, **_out)
            if _out['posout'] is not False:
                self.subsamples.add_column(ppos_AB, name='pos', copy=False)
            if _out['velout'] is not False:
                self.subsamples.add_column(pvel_AB, name='vel', copy=False)
        else:
            self.subsamples.add_column(particles_AB_total, name='rvint', copy=False)


    def nbytes(self, halos=True, subsamples=True):
        '''Return the memory usage of the big arrays: the halo catalog and the particle subsamples'''
        nbytes = 0
        which = []
        if halos:
            which += [self.halos]
        if subsamples:
            which += [self.subsamples]
        for cat in which:
            for col in cat.columns:
                nbytes += cat[col].nbytes
        return nbytes
    
    @staticmethod
    def is_path_halo_lc(path):
        path = str(path)
        return 'halo_light_cones' in path or bool(glob(pjoin(path, 'lc_*.asdf')))
        

    def __repr__(self):
        # TODO: there's probably some more helpful info we could put in here
        # Formally, this is supposed to be unambiguous, but mostly we just want it to look good in a notebook
        lines =   ['CompaSO Halo Catalog',
                   '====================',
                  f'{self.header["SimName"]} @ z={self.header["Redshift"]:.5g}',
                ]
        n_halo_field = len(self.halos.columns)
        n_subsamp_field = len(self.subsamples.columns)
        lines += ['-'*len(lines[-1]),
                 f'     Halos: {len(self.halos):8.3g} halos,     {n_halo_field:3d} {"fields" if n_halo_field != 1 else "field "}, {self.nbytes(halos=True,subsamples=False)/1e9:7.3g} GB',
                 f'Subsamples: {len(self.subsamples):8.3g} particles, {n_subsamp_field:3d} {"fields" if n_subsamp_field != 1 else "field "}, {self.nbytes(halos=False,subsamples=True)/1e9:7.3g} GB',
                 f'Cleaned halos: {self.cleaned}',
                 f'Halo light cone: {self.halo_lc}',
                 ]
        return '\n'.join(lines)



def _reindex_subsamples_from_asdf_size(subsamp_start, particle_arrays, N_halo_per_file):
    '''
    For subsample redshifts where we have L1s followed by L0s in the halo_pids files,
    we need to reindex using the total number of PIDs in the file, not the npout fields,
    which only have the L1s.
    '''

    nh = 0
    for k,p in enumerate(particle_arrays):
        nh += N_halo_per_file[k]
        np_thisfile = len(p)
        subsamp_start[nh:] += np_thisfile


####################################################################################################
# The following constants and functions relate to unpacking our compressed halo and particle formats
####################################################################################################

# Constants
EULER_ABIN = 45
EULER_TBIN = 11
EULER_NORM = 1.8477590650225735122 # 1/sqrt(1-1/sqrt(2))

INT16SCALE = 32000.

# function to combine halo subsample and merged particle subsample arrays
@nb.njit
def join_arrays(offset, array_original, array_merge, array_joined, npstart_original, npout_original, npstart_merge, npout_merge, np_total):

    N = len(np_total)
    npstart_updated = np.empty(N, dtype=np.int64)

    for i in range(N):
        ntotal_now  = np_total[i]

        if ntotal_now == 0:
            npstart_updated[i] = -999
            continue

        npstart_now = npstart_original[i]
        npout_now   = npout_original[i]
        npstart_updated[i] = offset
        array_joined[offset:offset+npout_now] = array_original[npstart_now:npstart_now+npout_now]
        offset += npout_now
        npstart_merge_now = npstart_merge[i]
        npout_merge_now = npout_merge[i]
        array_joined[offset:offset+npout_merge_now] = array_merge[npstart_merge_now:npstart_merge_now+npout_merge_now]
        offset += npout_merge_now

    return array_joined, npstart_updated, offset

# unpack the eigenvectors
def unpack_euler16(bin_this):
    N = bin_this.shape[0]
    minor = np.zeros((N,3))
    middle = np.zeros((N,3))
    major = np.zeros((N,3))

    cap = bin_this//EULER_ABIN
    iaz = bin_this - cap*EULER_ABIN   # This is the minor axis bin_this
    bin_this = cap
    cap = bin_this//(EULER_TBIN*EULER_TBIN)   # This is the cap
    bin_this = bin_this - cap*(EULER_TBIN*EULER_TBIN)

    it = (np.floor(np.sqrt(bin_this))).astype(int)
    its = np.sum(np.isnan(it))


    ir = bin_this - it*it

    t = (it+0.5)*(1.0/EULER_TBIN)   # [0,1]
    r = (ir+0.5)/(it+0.5)-1.0            # [-1,1]

    # We need to undo the transformation of t to get back to yy/zz
    t *= 1/EULER_NORM
    t = t * np.sqrt(2.0-t*t)/(1.0-t*t)   # Now we have yy/zz

    yy = t
    xx = r*t
    # and zz=1
    norm = 1.0/np.sqrt(1.0+xx*xx+yy*yy)
    zz = norm
    yy *= norm; xx *= norm;  # These are now a unit vector

    # TODO: legacy code, rewrite
    major[cap==0,0] = zz[cap==0]; major[cap==0,1] = yy[cap==0]; major[cap==0,2] = xx[cap==0];
    major[cap==1,0] = zz[cap==1]; major[cap==1,1] =-yy[cap==1]; major[cap==1,2] = xx[cap==1];
    major[cap==2,0] = zz[cap==2]; major[cap==2,1] = xx[cap==2]; major[cap==2,2] = yy[cap==2];
    major[cap==3,0] = zz[cap==3]; major[cap==3,1] = xx[cap==3]; major[cap==3,2] =-yy[cap==3];

    major[cap==4,1] = zz[cap==4]; major[cap==4,2] = yy[cap==4]; major[cap==4,0] = xx[cap==4];
    major[cap==5,1] = zz[cap==5]; major[cap==5,2] =-yy[cap==5]; major[cap==5,0] = xx[cap==5];
    major[cap==6,1] = zz[cap==6]; major[cap==6,2] = xx[cap==6]; major[cap==6,0] = yy[cap==6];
    major[cap==7,1] = zz[cap==7]; major[cap==7,2] = xx[cap==7]; major[cap==7,0] =-yy[cap==7];

    major[cap==8,2] = zz[cap==8]; major[cap==8,0] = yy[cap==8]; major[cap==8,1] = xx[cap==8];
    major[cap==9,2] = zz[cap==9]; major[cap==9,0] =-yy[cap==9]; major[cap==9,1] = xx[cap==9];
    major[cap==10,2] = zz[cap==10]; major[cap==10,0] = xx[cap==10]; major[cap==10,1] = yy[cap==10];
    major[cap==11,2] = zz[cap==11]; major[cap==11,0] = xx[cap==11]; major[cap==11,1] =-yy[cap==11];

    # Next, we can get the minor axis
    az = (iaz+0.5)*(1.0/EULER_ABIN)*np.pi
    xx = np.cos(az)
    yy = np.sin(az)
    # print("az = %f, %f, %f\n", az, xx, yy)
    # We have to derive the 3rd coord, using the fact that the two axes
    # are perpendicular.

    eq2 = (cap//4) == 2
    minor[eq2,0] = xx[eq2]; minor[eq2,1] = yy[eq2];
    minor[eq2,2] = (minor[eq2,0]*major[eq2,0]+minor[eq2,1]*major[eq2,1])/(-major[eq2,2])
    eq4 = (cap//4) == 0
    minor[eq4,1] = xx[eq4]; minor[eq4,2] = yy[eq4];
    minor[eq4,0] = (minor[eq4,1]*major[eq4,1]+minor[eq4,2]*major[eq4,2])/(-major[eq4,0])
    eq1 = (cap//4) == 1
    minor[eq1,2] = xx[eq1]; minor[eq1,0] = yy[eq1];
    minor[eq1,1] = (minor[eq1,2]*major[eq1,2]+minor[eq1,0]*major[eq1,0])/(-major[eq1,1])
    minor *= (1./np.linalg.norm(minor,axis=1).reshape(N,1))

    middle = np.zeros((minor.shape[0],3))
    middle[:,0] = minor[:,1]*major[:,2]-minor[:,2]*major[:,1]
    middle[:,1] = minor[:,2]*major[:,0]-minor[:,0]*major[:,2]
    middle[:,2] = minor[:,0]*major[:,1]-minor[:,1]*major[:,0]
    middle *= (1./np.linalg.norm(middle,axis=1).reshape(N,1))
    return minor, middle, major


"""
struct HaloStat {
    uint64_t id;    ///< A unique halo number.
    uint64_t npstartA;  ///< Where to start counting in the particle output for subsample A
    uint64_t npstartB;  ///< Where to start counting in the particle output for subsample B
    uint32_t npoutA;    ///< Number of taggable particles pos/vel/aux written out in subsample A
    uint32_t npoutB;    ///< Number of taggable particles pos/vel/aux written out in subsample B
    uint32_t ntaggedA;      ///< Number of tagged particle PIDs written out in subsample A. A particle is tagged if it is taggable and is in the largest L2 halo for a given L1 halo.
    uint32_t ntaggedB;
    uint32_t N; ///< The number of particles in this halo
    uint32_t L2_N[N_LARGEST_SUBHALOS];   ///< The number of particles in the largest L2 subhalos
    uint32_t L0_N;    ///< The number of particles in the L0 parent group

    float x_com[3];      ///< Center of mass position
    float v_com[3];      ///< Center of mass velocity
    float sigmav3d_com;  ///< Sum of eigenvalues
    float meanSpeed_com;  ///< Mean speed (the norm of the velocity vector)
    float sigmav3d_r50_com;  ///< Velocity dispersion of the inner 50% of particles
    float meanSpeed_r50_com;  ///< Mean speed of the inner 50% of particles
    float r100_com; ///<Radius of 100% of mass
    float vcirc_max_com; ///< max circular velocity, based on the particles in this L1 halo
    float SO_central_particle[3]; ///< Coordinates of the SO central particle
    float SO_central_density;  ///< Density of the SO central particle.
    float SO_radius;           ///< Radius of SO halo (distance to particle furthest from central particle)

    float x_L2com[3];   ///< Center of mass pos of the largest L2 subhalo
    float v_L2com[3];   ///< Center of mass vel of the largest L2 subhalo
    float sigmav3d_L2com;  ///< Sum of eigenvalues
    float meanSpeed_L2com;  ///< Mean speed
    float sigmav3d_r50_L2com;  ///< Velocity dispersion of the inner 50% of particles
    float meanSpeed_r50_L2com;  ///< Mean speed of the inner 50% of particles
    float r100_L2com; /// Radius of 100% of mass, relative to L2 center.
    float vcirc_max_L2com;   ///< max circular velocity, based on the particles in this L1 halo
    float SO_L2max_central_particle[3]; ///< Coordinates of the SO central particle for the largest L2 subhalo.
    float SO_L2max_central_density;  ///< Density of the SO central particle of the largest L2 subhalo.
    float SO_L2max_radius;           ///< Radius of SO halo (distance to particle furthest from central particle) for the largest L2 subhalo

    int16_t sigmavMin_to_sigmav3d_com; ///< Min(sigmav_eigenvalue) / sigmav3d, compressed
    int16_t sigmavMax_to_sigmav3d_com; ///< Max(sigmav_eigenvalue) / sigmav3d, compressed
    uint16_t sigmav_eigenvecs_com;  ///<Eigenvectors of the velocity dispersion tensor, compressed into 16 bits.
    int16_t sigmavrad_to_sigmav3d_com; ///< sigmav_rad / sigmav3d, compressed
    int16_t sigmavtan_to_sigmav3d_com; ///< sigmav_tan / sigmav3d, compressed

    int16_t r10_com, r25_com, r33_com, r50_com, r67_com, r75_com, r90_com, r95_com, r98_com; ///<Expressed as ratios of r100, and scaled to 32000 to store as int16s.
    int16_t sigmar_com[3]; ///<sqrt( Eigenvalues of the moment of inertia tensor ), sorted largest to smallest
    int16_t sigman_com[3]; ///<sqrt( Eigenvalues of the weighted moment of inertia tensor ), sorted largest to smallest
    uint16_t sigmar_eigenvecs_com;  ///<Eigenvectors of the moment of inertia tensor, compressed into 16 bits. Compression format Euler16.
    uint16_t sigman_eigenvecs_com;  ///<Eigenvectors of the weighted moment of inertia tensor, compressed into 16 bits. Compression format Euler16.
    int16_t rvcirc_max_com; ///< radius of max velocity, stored as int16 ratio of r100 scaled by 32000.

    // The largest (most massive) subhalo center of mass
    int16_t sigmavMin_to_sigmav3d_L2com; ///< Min(sigmav_eigenvalue) / sigmav3d, compressed
    int16_t sigmavMax_to_sigmav3d_L2com; ///< Max(sigmav_eigenvalue) / sigmav3d, compressed
    uint16_t sigmav_eigenvecs_L2com;  ///<Eigenvectors of the velocity dispersion tensor, compressed into 16 bits.
    int16_t sigmavrad_to_sigmav3d_L2com; ///< sigmav_rad / sigmav3d, compressed
    int16_t sigmavtan_to_sigmav3d_L2com; ///< sigmav_tan / sigmav3d, compressed
    int16_t r10_L2com, r25_L2com, r33_L2com, r50_L2com, r67_L2com, r75_L2com, r90_L2com, r95_L2com, r98_L2com;
        ///< Radii of this percentage of mass, relative to L2 center. Expressed as ratios of r100 and compressed to int16.

    int16_t sigmar_L2com[3];
    int16_t sigman_L2com[3];
    uint16_t sigmar_eigenvecs_L2com;   ///< euler16 format
    uint16_t sigman_eigenvecs_L2com;   ///< euler16 format
    int16_t rvcirc_max_L2com;   ///< radius of max circular velocity, stored as ratio to r100, relative to L2 center

};
"""

# Note we never actually create a Numpy array with this dtype
# But it is a useful format for parsing the needed dtypes for the Astropy table columns

clean_dt = np.dtype([('npstartA_merge', np.int64),
                     ('npstartB_merge', np.int64),
                     ('npoutA_merge', np.uint32),
                     ('npoutB_merge', np.uint32),
                     ('N_total', np.uint32),
                     ('N_merge', np.uint32),
                     ('haloindex', np.uint64),
                     ('is_merged_to', np.int64),
                     ('haloindex_mainprog', np.int64),
                     ('v_L2com_mainprog', np.float32, 3),
], align=True)

clean_dt_progen = np.dtype([('npstartA_merge', np.int64),
                            ('npstartB_merge', np.int64),
                            ('npoutA_merge', np.uint32),
                            ('npoutB_merge', np.uint32),
                            ('N_total', np.uint32),
                            ('N_merge', np.uint32),
                            ('haloindex', np.uint64),
                            ('is_merged_to', np.int64),
                            ('N_mainprog', np.uint32),
                            ('vcirc_max_L2com_mainprog', np.float32),
                            ('sigmav3d_L2com_mainprog', np.float32),
                            ('haloindex_mainprog', np.int64),
                            ('v_L2com_mainprog', np.float32, 3),
], align=True)

halo_lc_dt = np.dtype([('N', np.uint32),
                       ('N_interp', np.uint32),
                       ('npstartA', np.uint64),
                       ('npoutA', np.uint32),
                       ('index_halo', np.int64),
                       ('origin', np.int8),
                       ('pos_avg', np.float32, 3),
                       ('pos_interp', np.float32, 3),
                       ('vel_avg', np.float32, 3),
                       ('vel_interp', np.float32, 3),
                       ('redshift_interp', np.float32),
], align=True)

user_dt = np.dtype([('id', np.uint64),
                    ('npstartA', np.uint64),
                    ('npstartB', np.uint64),

                    ('npoutA', np.uint32),
                    ('npoutB', np.uint32),
                    ('ntaggedA', np.uint32),
                    ('ntaggedB', np.uint32),
                    ('N', np.uint32),
                    ('L2_N', np.uint32, 5),
                    ('L0_N', np.uint32),

                    ('x_com', np.float32, 3),
                    ('v_com', np.float32, 3),
                    ('sigmav3d_com', np.float32),
                    ('meanSpeed_com', np.float32),
                    ('sigmav3d_r50_com', np.float32),
                    ('meanSpeed_r50_com', np.float32),
                    ('r100_com', np.float32),
                    ('vcirc_max_com', np.float32),
                    ('SO_central_particle', np.float32, 3),
                    ('SO_central_density', np.float32),
                    ('SO_radius', np.float32),

                    ('x_L2com', np.float32, 3),
                    ('v_L2com', np.float32, 3),
                    ('sigmav3d_L2com', np.float32),
                    ('meanSpeed_L2com', np.float32),
                    ('sigmav3d_r50_L2com', np.float32),
                    ('meanSpeed_r50_L2com', np.float32),
                    ('r100_L2com', np.float32),
                    ('vcirc_max_L2com', np.float32),
                    ('SO_L2max_central_particle', np.float32, 3),
                    ('SO_L2max_central_density', np.float32),
                    ('SO_L2max_radius', np.float32),

                    ('sigmavMin_com', np.float32),
                    ('sigmavMid_com', np.float32),
                    ('sigmavMaj_com', np.float32),

                    ('r10_com', np.float32),
                    ('r25_com', np.float32),
                    ('r33_com', np.float32),
                    ('r50_com', np.float32),
                    ('r67_com', np.float32),
                    ('r75_com', np.float32),
                    ('r90_com', np.float32),
                    ('r95_com', np.float32),
                    ('r98_com', np.float32),

                    ('sigmar_com', np.float32, 3),
                    ('sigman_com', np.float32, 3),
                    ('sigmar_eigenvecsMin_com', np.float32, 3),
                    ('sigmar_eigenvecsMid_com', np.float32, 3),
                    ('sigmar_eigenvecsMaj_com', np.float32, 3),
                    ('sigmav_eigenvecsMin_com', np.float32, 3),
                    ('sigmav_eigenvecsMid_com', np.float32, 3),
                    ('sigmav_eigenvecsMaj_com', np.float32, 3),
                    ('sigman_eigenvecsMin_com', np.float32, 3),
                    ('sigman_eigenvecsMid_com', np.float32, 3),
                    ('sigman_eigenvecsMaj_com', np.float32, 3),

                    ('sigmavrad_com', np.float32),
                    ('sigmavtan_com', np.float32),
                    ('rvcirc_max_com', np.float32),

                    ('sigmavMin_L2com', np.float32),
                    ('sigmavMid_L2com', np.float32),
                    ('sigmavMaj_L2com', np.float32),

                    ('r10_L2com', np.float32),
                    ('r25_L2com', np.float32),
                    ('r33_L2com', np.float32),
                    ('r50_L2com', np.float32),
                    ('r67_L2com', np.float32),
                    ('r75_L2com', np.float32),
                    ('r90_L2com', np.float32),
                    ('r95_L2com', np.float32),
                    ('r98_L2com', np.float32),

                    ('sigmar_L2com', np.float32, 3),
                    ('sigman_L2com', np.float32, 3),
                    ('sigmar_eigenvecsMin_L2com', np.float32, 3),
                    ('sigmar_eigenvecsMid_L2com', np.float32, 3),
                    ('sigmar_eigenvecsMaj_L2com', np.float32, 3),
                    ('sigmav_eigenvecsMin_L2com', np.float32, 3),
                    ('sigmav_eigenvecsMid_L2com', np.float32, 3),
                    ('sigmav_eigenvecsMaj_L2com', np.float32, 3),
                    ('sigman_eigenvecsMin_L2com', np.float32, 3),
                    ('sigman_eigenvecsMid_L2com', np.float32, 3),
                    ('sigman_eigenvecsMaj_L2com', np.float32, 3),

                    ('sigmavrad_L2com', np.float32),
                    ('sigmavtan_L2com', np.float32),
                    ('rvcirc_max_L2com', np.float32),
], align=True)
 
