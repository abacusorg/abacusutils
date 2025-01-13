# The compaso_halo_catalog module loads halo catalogs from CompaSO, Abacus's
# on-the-fly halo finder. The module defines one class, CompaSOHaloCatalog,
# whose constructor takes the path to a halo catalog as an argument.
# Users should use this class as the primary interface to load and manipulate
# halo catalogs.

# A high-level overview of this module is given at
# https://abacusutils.readthedocs.io/en/latest/compaso.html
# or docs/compaso.rst.

import gc
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path, PurePath

import asdf
import astropy.table
import numba
import numpy as np
from astropy.table import Table

try:
    import asdf._compression as asdf_compression
except ImportError:
    import asdf.compression as asdf_compression

from .. import util
from . import asdf as _asdf
from . import bitpacked

try:
    asdf_compression.validate('blsc')
except Exception as e:
    raise Exception(
        'Abacus ASDF extension not properly loaded! Try reinstalling abacusutils, or updating ASDF: `pip install asdf>=2.8`'
    ) from e


# Default to 4 decompression threads, or fewer if fewer cores are available
DEFAULT_BLOSC_THREADS = 4
DEFAULT_BLOSC_THREADS = max(1, min(len(os.sched_getaffinity(0)), DEFAULT_BLOSC_THREADS))

_asdf.set_nthreads(DEFAULT_BLOSC_THREADS)


class CompaSOHaloCatalog:
    """
    A halo catalog from Abacus's on-the-fly group finder.
    """

    # TODO: optional progress meter for loading files
    # TODO: generator mode over superslabs

    def __init__(
        self,
        path,
        cleaned=True,
        subsamples=False,
        convert_units=True,
        unpack_bits=False,
        fields='DEFAULT_FIELDS',
        verbose=False,
        cleandir=None,
        filter_func=None,
        halo_lc=None,
        passthrough=False,
        **kwargs,
    ):
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

                filter_func = lambda h: h['N'] >= 100

        halo_lc: bool or None, optional
            Whether the catalog is a halo light cone catalog, i.e. an output of the CompaSO
            halo light cone pipeline. Default of None means to detect based on the catalog path.

        passthrough: bool, optional
            Do not unpack any of the halo or subsample columns, just load the raw data.  This is useful
            for pipelining, where the data will be unpacked later. Subsample indices, filter_func,
            and cleaning will all still be applied. Defaut: False.

        """
        # Internally, we will use `load_subsamples` as the name of the `subsamples` arg to distinguish it from the `self.subsamples` table
        load_subsamples = subsamples
        del subsamples

        # `cleaned` and `self.cleaned` mean slightly different things.
        # `cleaned` (local var) means to load the cleaning info files,
        # `self.cleaned` means the catalog incorporates cleaning info, either because the user
        # said `cleaned=True` or because this is a halo light cone catalog, which is already cleaned
        self.cleaned = cleaned

        if halo_lc is None:
            halo_lc = self._is_path_halo_lc(
                path[0] if not isinstance(path, (PurePath, str)) else path
            )
            if verbose and halo_lc:
                print('Detected halo light cone catalog.')
        self.halo_lc = halo_lc

        # If loading halo light cones, turn off cleaning and bit unpacking because done already
        if halo_lc:
            if not self.cleaned:
                warnings.warn(
                    '`cleaned=False` was specified but halo light cones always incorporate cleaning'
                )
            cleaned = False
            unpack_bits = False
            self.cleaned = True

        # Check no unknown args!
        if kwargs:
            raise ValueError(
                f'Unknown arguments to CompaSOHaloCatalog constructor: {list(kwargs)}'
            )

        # Parse `path` to determine what files to read
        (
            self.groupdir,
            self.clean_halo_info_dir,
            self.clean_rvpid_dir,
            self.superslab_inds,
            self.halo_fns,
            self.cleaned_halo_fns,
        ) = self._setup_file_paths(
            path, cleaned=cleaned, cleandir=cleandir, halo_lc=halo_lc
        )

        # Figure out what subsamples the user is asking us to loads
        self.load_AB, self.load_pidrv = self._setup_load_subsamples(
            load_subsamples, passthrough=passthrough
        )
        del load_subsamples  # use the parsed values

        # If using halo light cones, only have subsample A available
        if halo_lc and self.load_AB:
            self.load_AB = ['A']

        self.data_key = 'data'
        self.convert_units = convert_units  # let's save, user might want to check later
        self.verbose = verbose
        self.filter_func = filter_func

        unpack_bits = self._setup_unpack_bits(unpack_bits)

        # End parameter parsing, begin opening files

        # Open the first file, just to grab the header
        with asdf.open(self.halo_fns[0], lazy_load=True) as af:
            # will also be available as self.halos.meta
            self.header = af['header']
            # For any applications that propagate the header, record whether they used cleaned halos
            self.header['cleaned_halos'] = self.cleaned

        # If we are using cleaned haloes, want to also grab header information regarding number of preceding timesteps
        if cleaned:
            with asdf.open(self.cleaned_halo_fns[0], lazy_load=True) as af:
                self.header['TimeSliceRedshiftsPrev'] = af['header'][
                    'TimeSliceRedshiftsPrev'
                ]
                self.header['NumTimeSliceRedshiftsPrev'] = len(
                    af['header']['TimeSliceRedshiftsPrev']
                )

        # Read and unpack the catalog into self.halos
        self._setup_halo_field_loaders(passthrough=passthrough)
        N_halo_per_file = self._read_halo_info(
            self.halo_fns,
            fields,
            cleaned=cleaned,
            passthrough=passthrough,
            cleaned_fns=self.cleaned_halo_fns,
        )

        # empty table, to be filled with PIDs and RVs in the loading functions below
        self.subsamples = Table()

        # The subsample loading algorithm is:
        # - load all the (cleaned) halos with their particle indexing info,
        #   maybe with filters applied but not yet reindexed
        # - compute a new set of indices: write start locations and lengths,
        #   both of which combine original and cleaned particles and all files.
        #   The overall ordering will be [[[[original_AB, cleaned_AB] for halo] for file] for A, B].
        # - allocate the output columns for the rv/pid/A/B particles using the new lengths
        # - for each (rv,pid) x (a,b) combo, read a particle file and its cleaned counterpart,
        #   then loop over the corresponding halos (which we know from N_halo_per_file).
        #   For each halo, use the original indices to find the particles.
        #   Unpack them directly into the corresponding column(s), using
        #   the halo's write indexing. Do the original then cleaned particles for each halo.
        #   After writing, check that the number of particles matched the expected write length.

        if halo_lc:
            self._load_halo_lc_subsamples(
                which=self.load_pidrv, unpack_bits=unpack_bits
            )

        elif self.load_AB:
            npstartAB_new = self._compute_new_subsample_indices(
                cleaned=cleaned, load_AB=self.load_AB
            )

            self._load_subsamples(
                N_halo_per_file,
                npstartAB_new,
                which=self.load_pidrv,
                load_AB=self.load_AB,
                cleaned=cleaned,
                unpack_bits=unpack_bits,
            )

            self._update_subsample_index_cols(
                npstartAB_new, load_AB=self.load_AB, cleaned=cleaned
            )

        # If we're reading in cleaned haloes, N should be updated
        if cleaned and not passthrough:
            self.halos.rename_column('N_total', 'N')

        if verbose:
            print('\n' + str(self))

        gc.collect()

    def _setup_file_paths(self, path, cleaned=True, cleandir=None, halo_lc=False):
        """Figure out what files the user is asking for"""

        if isinstance(path, (PurePath, str)):
            path = [Path(path)]  # dir or file
        else:
            path = [Path(p) for p in path]

            # if list, must be all files
            for p in path:
                if p.exists() and not p.is_file():
                    raise ValueError(
                        f'If passing a list of paths, all paths must be files, not dirs. Path "{p}" is not a file.'
                    )

        for p in path:
            if not os.path.exists(p):
                raise FileNotFoundError(f'Path "{p}" does not exist!')

        path = [p.absolute() for p in path]

        # Allow users to pass halo_info dirs, even though redshift dirs remain canonical
        for i, p in enumerate(path):
            if p.name == 'halo_info':
                path[i] = p.parent

        # Can't mix files from different catalogs!
        if path[0].is_file():
            groupdir = path[0].parents[1]
            if halo_lc:
                groupdir = path[0].parent
            for p in path:
                if not groupdir == p.parents[1] and not halo_lc:
                    raise ValueError("Can't mix files from different catalogs!")
                halo_fns = path  # path is list of one or more files

            for i, p in enumerate(path):
                for j, q in enumerate(path[i + 1 :]):
                    if p == q:
                        raise ValueError(
                            f'Cannot pass duplicate halo_info files! Found duplicate "{p}" and at indices {i} and {i + j + 1}'
                        )
        else:
            groupdir = path[0]  # path is a singlet of one dir
            if halo_lc:  # naming convention differs for the light cone catalogs
                globpat = 'lc_halo_info*.asdf'
            else:
                globpat = 'halo_info/halo_info_*.asdf'
            halo_fns = sorted(groupdir.glob(globpat))
            if len(halo_fns) == 0:
                raise FileNotFoundError(
                    f'No halo_info files found! Search pattern was: "{groupdir / globpat}"'
                )

        if halo_lc:
            # halo light cones files aggregate all superslabs into a single file
            superslab_inds = np.array([0])
        else:
            superslab_inds = np.array(
                [int(hfn.stem.split('_')[-1]) for hfn in halo_fns]
            )

        if cleaned:
            if not cleandir:
                for p in groupdir.parents:
                    if (cleandir := (p / 'cleaning')).is_dir():
                        break
                else:
                    raise FileNotFoundError(
                        f'Could not find cleaning info dir, searching upwards from {groupdir}. To load the uncleaned catalog, use `cleaned=False`.'
                    )

            # Check for structures like:
            # cleaning/SimName/z0.000/cleaned_halo_info/cleaned_halo_info_000.asdf
            # cleaning/small/SmallSimName/z0.000/cleaned_halo_info/cleaned_halo_info_000.asdf
            # SimName/cleaning/z0.000/cleaned_halo_info/cleaned_halo_info_000.asdf
            # SimName/cleaning/z0.000/cleaned_halo_info_000.asdf
            relpath = (groupdir.parents[1] / groupdir.name).relative_to(cleandir.parent)
            if (cleandir / relpath / 'cleaned_halo_info').is_dir():
                clean_halo_info_dir = cleandir / relpath / 'cleaned_halo_info'
                clean_rvpid_dir = cleandir / relpath / 'cleaned_rvpid'
            else:
                clean_halo_info_dir = cleandir / relpath
                clean_rvpid_dir = cleandir / relpath

            cleaned_halo_fns = [
                clean_halo_info_dir / f'cleaned_halo_info_{i:03d}.asdf'
                for i in superslab_inds
            ]

            for fn in cleaned_halo_fns:
                if not fn.is_file():
                    raise FileNotFoundError(
                        f'Cleaning info not found. File path was: "{fn}". To load the uncleaned catalog, use `cleaned=False`.'
                    )
        else:
            clean_halo_info_dir = None
            clean_rvpid_dir = None
            cleaned_halo_fns: list[Path] = []

        return (
            groupdir,
            clean_halo_info_dir,
            clean_rvpid_dir,
            superslab_inds,
            halo_fns,
            cleaned_halo_fns,
        )

    def _setup_unpack_bits(self, unpack_bits):
        # validate unpack_bits
        if isinstance(unpack_bits, str):
            unpack_bits = [unpack_bits]
        if unpack_bits not in (True, False):
            try:
                for _f in unpack_bits:
                    assert _f in bitpacked.PID_FIELDS
            except Exception:
                raise ValueError(
                    f'`unpack_bits` must be True, False, or one of: "{bitpacked.PID_FIELDS}"'
                )
        return unpack_bits

    def _setup_load_subsamples(self, load_subsamples, passthrough=False):
        """
        Figure out if the user wants A, B, pid, pos, vel.
        Will be returned as lists of strings in `load_AB` and `load_pidrv`.
        """
        if load_subsamples is False:
            # stub
            load_AB = []
            load_pidrv = []
        else:
            # If user has not specified which subsamples, then assume user wants to load everything
            if load_subsamples is True:
                if passthrough:
                    load_subsamples = dict(A=True, B=True, rvint=True, packedpid=True)
                else:
                    load_subsamples = dict(A=True, B=True, rv=True, pid=True)

            if isinstance(load_subsamples, dict):
                load_AB = [k for k in 'AB' if load_subsamples.get(k)]  # ['A', 'B']

                # Check for conflicts between rv, pos, vel. Must be done before list-ifying to distinguish False and not given.
                if 'rv' in load_subsamples:
                    if 'pos' in load_subsamples or 'vel' in load_subsamples:
                        raise ValueError(
                            'Cannot pass `rv` and `pos` or `vel` in `load_subsamples`.'
                        )

                load_pidrv = [
                    k
                    for k in load_subsamples
                    if k in ('pid', 'pos', 'vel', 'rv', 'rvint', 'packedpid')
                    and load_subsamples.get(k)
                ]  # ['pid', 'pos', 'vel']

                # set some intelligent defaults
                if load_pidrv and not load_AB:
                    warnings.warn(
                        f'Loading of {load_pidrv} was requested but neither subsample A nor B was specified. Assuming subsample A. Can specify with `load_subsamples=dict(A=True)`.'
                    )
                    load_AB = ['A']
                elif not load_pidrv and load_AB:
                    if load_subsamples.get('pos') is not False:
                        load_pidrv += ['pos']
                    if load_subsamples.get('vel') is not False:
                        load_pidrv += ['vel']
                    if not load_pidrv:
                        warnings.warn(
                            f'Loading of subsample {load_AB} was requested but none of `pos`, `vel`, `rv`, `pid` was specified. Assuming `rv`. Can specify with `load_subsamples=dict(rv=True)`.'
                        )
                        load_pidrv = ['rv']

                if load_subsamples.pop('field', False):
                    raise ValueError(
                        'Loading field particles through CompaSOHaloCatalog is not supported. Read the particle files directly with `abacusnbody.data.read_abacus.read_asdf()`.'
                    )

                # Pop all known keys, so if anything is left, that's an error!
                for k in [
                    'A',
                    'B',
                    'rv',
                    'pid',
                    'pos',
                    'vel',
                    'unpack',
                    'rvint',
                    'packedpid',
                ]:
                    load_subsamples.pop(k, None)

                if load_subsamples:
                    raise ValueError(
                        f'Unrecognized keys in `load_subsamples`: {list(load_subsamples)}'
                    )

        if 'rv' in load_pidrv:
            load_pidrv.remove('rv')
            load_pidrv += ['pos', 'vel']

        return load_AB, load_pidrv

    def _setup_fields(
        self,
        fields,
        cleaned=True,
        load_AB=None,
        halo_lc=False,
        passthrough=False,
        halo_info_af=None,
        cleaned_halo_info_af=None,
    ):
        """Determine the halo catalog fields to load based on user input"""

        if passthrough:
            # In passthrough mode, the fields are determined by the file contents
            raw_fields = list(halo_info_af[self.data_key])
            raw_cleaned_fields = list(cleaned_halo_info_af[self.data_key])

            if fields == 'all':
                fields = raw_fields
                cleaned_fields = raw_cleaned_fields
            else:
                if isinstance(fields, str):
                    fields = [fields]

                fields = [r for r in raw_fields if r in fields]
                cleaned_fields = [r for r in raw_cleaned_fields if r in fields]

            return fields, cleaned_fields

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

        if isinstance(fields, str):
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

        if load_AB is None:
            load_AB = []

        if cleaned:
            # If the user has not asked to load npstart{AB}_merge columns, we need to do so ourselves for indexing
            for AB in load_AB:
                if 'npstart' + AB not in fields:
                    fields += ['npstart' + AB]
                if 'npout' + AB not in fields:
                    fields += ['npout' + AB]
                if 'npstart' + AB + '_merge' not in cleaned_fields:
                    cleaned_fields += ['npstart' + AB + '_merge']
                if 'npout' + AB + '_merge' not in cleaned_fields:
                    cleaned_fields += ['npout' + AB + '_merge']

        return fields, cleaned_fields

    def _read_halo_info(
        self,
        halo_fns,
        fields,
        cleaned=False,
        cleaned_fns=None,
        passthrough=False,
    ):
        if not cleaned_fns:
            cleaned_fns = []
        else:
            assert len(cleaned_fns) == len(halo_fns)

        # Open all the files, validate them, and count the halos
        # Lazy load, but don't use mmap
        afs = [asdf.open(hfn, lazy_load=True, memmap=False) for hfn in halo_fns]
        cleaned_afs = [
            asdf.open(hfn, lazy_load=True, memmap=False) for hfn in cleaned_fns
        ]

        # Parse `fields` to determine halo catalog fields to read.
        # If using passthrough, the fields will be (a subset of) the raw columns
        # as determined from the files on disk
        fields, cleaned_fields = self._setup_fields(
            fields,
            cleaned=cleaned,
            load_AB=self.load_AB,
            halo_lc=self.halo_lc,
            passthrough=passthrough,
            halo_info_af=afs[0],
            cleaned_halo_info_af=cleaned_afs[0] if cleaned else None,
        )
        self.fields = fields
        self.cleaned_fields = cleaned_fields

        N_halo_per_file = np.array(
            [len(af[self.data_key][list(af[self.data_key].keys())[0]]) for af in afs]
        )
        for _N, caf in zip(N_halo_per_file, cleaned_afs):
            assert (
                len(caf[self.data_key][next(iter(caf[self.data_key]))]) == _N
            )  # check cleaned/regular file consistency

        N_halos = N_halo_per_file.sum()

        # Make an empty table for the concatenated, unpacked values
        # Note that np.empty is being smart here and creating 2D arrays when the dtype is a vector

        cols = {}
        if not passthrough:
            for col in fields:
                if col in halo_lc_dt.names:
                    cols[col] = np.empty(N_halos, dtype=halo_lc_dt[col])
                else:
                    cols[col] = np.empty(N_halos, dtype=user_dt[col])
            for col in cleaned_fields:
                cols[col] = np.empty(N_halos, dtype=clean_dt_progen[col])
        else:
            # For passthrough, the file contents determine the shapes/dtypes
            raw_cols = afs[0][self.data_key]
            for field in fields:
                col = raw_cols[field]
                cols[field] = np.empty((N_halos,) + col.shape[1:], dtype=col.dtype)

            raw_cols = cleaned_afs[0][self.data_key]
            for field in cleaned_fields:
                col = raw_cols[field]
                cols[field] = np.empty((N_halos,) + col.shape[1:], dtype=col.dtype)

        all_fields = list(cols)

        # Figure out what raw columns we need to read based on the fields the user requested
        # TODO: provide option to drop un-requested columns
        raw_dependencies, fields_with_deps, extra_fields = (
            self._get_halo_fields_dependencies(all_fields)
        )

        if passthrough:
            assert set(raw_dependencies) == set(fields_with_deps)
            assert len(extra_fields) == 0

        # save for informational purposes
        if not hasattr(self, 'dependency_info'):
            self.dependency_info = defaultdict(list)
        self.dependency_info['raw_dependencies'] += raw_dependencies
        self.dependency_info['fields_with_deps'] += fields_with_deps
        self.dependency_info['extra_fields'] += extra_fields

        if self.verbose:
            print(
                f'{len(fields)} halo catalog fields ({len(cleaned_fields)} cleaned) requested. '
                f'Reading {len(raw_dependencies)} fields from disk. '
                f'Computing {len(extra_fields)} intermediate fields.'
            )
            if self.halo_lc:
                print(
                    '\nFor more information on the halo light cone catalog fields, see https://abacussummit.readthedocs.io/en/latest/data-products.html#halo-light-cone-catalogs'
                )

        self.halos = Table(cols, copy=False)
        self.halos.meta.update(self.header)

        # If we're loading main progenitor info, do this:

        if not passthrough:
            # TODO: this shows the limits of querying the types from a numpy dtype, should query from a function
            r = re.compile('.*mainprog')
            prog_fields = list(filter(r.match, cleaned_fields))
            for fields in prog_fields:
                if fields in ['v_L2com_mainprog', 'haloindex_mainprog']:
                    continue
                else:
                    self.halos.replace_column(
                        fields,
                        np.empty(
                            N_halos,
                            dtype=(
                                clean_dt_progen[fields],
                                self.header['NumTimeSliceRedshiftsPrev'],
                            ),
                        ),
                        copy=False,
                    )

        # Unpack the cats into the concatenated array
        # The writes would probably be more efficient if the outer loop was over column
        # and the inner was over cats, but wow that would be ugly
        N_written = 0
        for i, af in enumerate(afs):
            caf = cleaned_afs[i] if cleaned_afs else None

            # This is where the IO on the raw columns happens
            # There are some fields that we'd prefer to directly read into the concatenated table,
            # but ASDF doesn't presently support that, so this is the best we can do
            rawhalos = {}
            for field in raw_dependencies:
                src = caf if field in clean_dt_progen.names else af
                rawhalos[field] = src[self.data_key][field][:]
            rawhalos = Table(data=rawhalos, copy=False)
            af.close()
            if caf:
                caf.close()

            # `halos` will be a "pointer" to the next open space in the master table
            halos = self.halos[N_written : N_written + len(rawhalos)]

            # For temporary (extra) columns, only need to construct the per-file version
            for field in extra_fields:
                src = clean_dt_progen if field in clean_dt_progen.names else user_dt
                halos.add_column(
                    np.empty(len(rawhalos), dtype=src[col]), name=field, copy=False
                )
                # halos[field][:] = np.nan  # for debugging

            loaded_fields = []
            for field in fields_with_deps:
                if field in loaded_fields:
                    continue
                loaded_fields += self._load_halo_field(halos, rawhalos, field)

            if self.filter_func:
                # N_total from the cleaning replaces N. For filtering purposes, allow the user to use 'N'
                if self.cleaned and not passthrough:
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

    def _setup_halo_field_loaders(self, passthrough=False):
        # Loaders is a dict of regex -> lambda
        # The lambda is responsible for unpacking the rawhalos field
        # The first regex that matches will be used, so they must be precise
        self.halo_field_loaders = {}

        if passthrough:
            pat = re.compile(r'.*')
            self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]]
            return

        if self.convert_units:
            box = self.header['BoxSize']
            # TODO: correct velocity units? There is an earlier comment claiming that velocities are already in km/s
            zspace_to_kms = self.header['VelZSpace_to_kms']
        else:
            box = 1.0
            zspace_to_kms = 1.0

        # The first argument to the following lambdas is the match object from re.match()
        # We will use m[0] to access the full match (i.e. the full field name)
        # Other indices, like m['com'], will access the sub-match with that group name

        # r10,r25,r33,r50,r67,r75,r90,r95,r98
        pat = re.compile(r'(?:r\d{1,2}|rvcirc_max)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = (
            lambda m, raw, halos: raw[m[0] + '_i16']
            * raw['r100' + m['com']]
            / INT16SCALE
            * box
        )

        # sigmavMin, sigmavMaj, sigmavrad, sigmavtan
        pat = re.compile(r'(?P<stem>sigmav(?:Min|Maj|rad|tan))(?P<com>_(?:L2)?com)')

        def _sigmav_loader(m, raw, halos):
            stem = m['stem'].replace('Maj', 'Max')
            return (
                raw[stem + '_to_sigmav3d' + m['com'] + '_i16']
                * raw['sigmav3d' + m['com']]
                / INT16SCALE
                * box
            )

        self.halo_field_loaders[pat] = _sigmav_loader

        # sigmavMid
        pat = re.compile(r'sigmavMid(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m, raw, halos: np.sqrt(
            raw['sigmav3d' + m['com']] * raw['sigmav3d' + m['com']] * box**2
            - halos['sigmavMaj' + m['com']] ** 2
            - halos['sigmavMin' + m['com']] ** 2
        )

        # sigmar
        pat = re.compile(r'sigmar(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = (
            lambda m, raw, halos: raw[m[0] + '_i16']
            * raw['r100' + m['com']].reshape(-1, 1)
            / INT16SCALE
            * box
        )

        # sigman
        pat = re.compile(r'sigman(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = (
            lambda m, raw, halos: raw[m[0] + '_i16'] / INT16SCALE * box
        )

        # x,r100 (box-scaled fields)
        pat = re.compile(r'(x|r100)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]] * box

        # v,sigmav,sigmav3d,meanSpeed,sigmav3d_r50,meanSpeed_r50,vcirc_max (vel-scaled fields)
        pat = re.compile(
            r'(v|sigmav3d|meanSpeed|sigmav3d_r50|meanSpeed_r50|vcirc_max)(?P<com>_(?:L2)?com)'
        )
        self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]] * zspace_to_kms

        # id,npstartA,npstartB,npoutA,npoutB,ntaggedA,ntaggedB,N,L2_N,L0_N (raw/passthrough fields)
        # If ASDF could read into a user-provided array, could avoid these copies
        pat = re.compile(
            r'id|npstartA|npstartB|npoutA|npoutB|ntaggedA|ntaggedB|N|L2_N|L0_N|N_total|N_merge|npstartA_merge|npstartB_merge|npoutA_merge|npoutB_merge|npoutA_L0L1|npoutB_L0L1|is_merged_to|N_mainprog|vcirc_max_L2com_mainprog|sigmav3d_L2com_mainprog|haloindex|haloindex_mainprog|v_L2com_mainprog'
        )
        self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]]

        # SO_central_particle,SO_radius (and _L2max) (box-scaled fields)
        pat = re.compile(r'SO(?:_L2max)?(?:_central_particle|_radius)')
        self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]] * box

        # SO_central_density (and _L2max)
        pat = re.compile(r'SO(?:_L2max)?(?:_central_density)')
        self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]]

        # loader for halo light cone catalog specific fields
        pat = re.compile(r'index_halo|pos_avg|vel_avg|redshift_interp|N_interp')
        self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]]

        # loader for halo light cone catalog field `origin`
        pat = re.compile(r'origin')
        self.halo_field_loaders[pat] = lambda m, raw, halos: raw[m[0]] % 3

        # loader for halo light cone catalog fields: interpolated position and velocity
        pat = re.compile(r'(?P<pv>pos|vel)_interp')

        def lc_interp_loader(m, raw, halos):
            columns = {}
            pa = np.atleast_2d(raw['pos_avg'])
            avg_avail = np.any(pa, axis=1)  # abacusnbody/hod/prepare_sim.py
            if m[0] == 'pos_interp' or 'pos_interp' in halos.colnames:
                columns['pos_interp'] = np.where(
                    avg_avail[:, None], raw['pos_avg'], raw['pos_interp']
                )
            if m[0] == 'vel_interp' or 'vel_interp' in halos.colnames:
                columns['vel_interp'] = np.where(
                    avg_avail[:, None], raw['vel_avg'], raw['vel_interp']
                )
            return columns

        self.halo_field_loaders[pat] = lc_interp_loader

        # eigvecs loader
        pat = re.compile(
            r'(?P<rnv>sigma(?:r|n|v)_eigenvecs)(?P<which>Min|Mid|Maj)(?P<com>_(?:L2)?com)'
        )

        def eigvecs_loader(m, raw, halos):
            minor, middle, major = _unpack_euler16(raw[m['rnv'] + m['com'] + '_u16'])
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
        """Each of the loaders accesses some raw columns on disk to
        produce the user-facing halo catalog columns. This function
        will determine which of those raw fields needs to be read
        by calling the loader functions with a dummy object that
        records field accesses.
        """

        # TODO: define pre-set subsets of common fields

        class DepCapture:
            def __init__(self):
                self.keys = []
                self.colnames = []

            def __getitem__(self, key):
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
                        raise KeyError(
                            f'Found more than one way to load field "{field}"'
                        )
                    capturer, raw_capturer = DepCapture(), DepCapture()
                    self.halo_field_loaders[pat](match, raw_capturer, capturer)
                    raw_dependencies += raw_capturer.keys

                    # these are fields of `halos`
                    for k in capturer.keys:
                        # Add fields regardless of whether they have already been encountered
                        iter_fields += [k]
                        if k not in fields:
                            field_dependencies += [k]
                    have_match = True
                    # break  # comment out for debugging
            else:
                if not have_match:
                    raise KeyError(f'Don\'t know how to load halo field "{field}"')

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
                column = self.halo_field_loaders[pat](match, rawhalos, halos)

                # Note that we missed an opportunity to load the fields in-place.
                # However, the extra copy should be on a per-field, per-file basis.
                # TODO: if we get asdf.read_into(), we should refactor this

                # The loader is allowed to return a dict if it incidentally loaded multiple columns
                if isinstance(column, dict):
                    assert field in column
                    for k in column:
                        halos[k][:] = column[k]
                    loaded_fields += list(column)
                else:
                    halos[field][:] = column
                    loaded_fields += [field]

                have_match = True
                # break  # comment out for debugging
        else:
            if not have_match:
                raise KeyError(f'Don\'t know how to load halo field "{field}"')

        astropy.table.conf.replace_warnings = _oldwarn

        return loaded_fields

    def _compute_new_subsample_indices(self, cleaned=True, load_AB=None):
        # Return the npstart{AB}_new arrays. This is where subsamples will be written.
        # One special thing about these arrays is that they will be oversized by 1
        # so that we know where the end of the last halo is.
        # Later, we'll shrink it by 1 then make it a halo column.
        # The order is original followed by clean for each halo, with all A before all B.

        offset = np.uint64(0)

        if cleaned:
            cleaned_mask = self.halos['N_total'] == 0

        npstartAB_new = {}
        for AB in load_AB:
            npoutAB = self.halos[f'npout{AB}']
            if cleaned:
                # Need to modify the originals, because these halos have been cleaned away.
                # Their subsample particles are already in another halo's incoming cleaned particles.
                self.halos[f'npout{AB}'][cleaned_mask] = 0

                # But can't add the merged counts yet to the originals,
                # we still need the original indexing for the reads.
                npoutAB = npoutAB + self.halos[f'npout{AB}_merge']

            npstartAB_new[AB] = np.empty(len(self.halos) + 1, dtype=np.uint64)
            offset = util.cumsum(
                npoutAB,
                npstartAB_new[AB],
                initial=True,
                final=True,
                offset=offset,
            )

        return npstartAB_new

    def _load_subsamples(
        self,
        N_halo_per_file,
        npstartAB_new,
        which=['pos', 'vel', 'pid'],  # 'rvint'
        load_AB=None,
        cleaned=True,
        check_pids=False,
        unpack_bits=False,
    ):
        # Read each requested subsample file.
        # Unpack the data directly into the subsample table,
        # using the write indices computed in _compute_new_subsample_indices().
        # The read indices are simply the unaltered npstart/npout columns!

        N_subsamp = npstartAB_new['B'][-1] if 'B' in load_AB else npstartAB_new['A'][-1]
        for w in which:
            if w in ('pos', 'vel', 'rvint'):
                shape = (N_subsamp, 3)
                dtype = np.int32 if w == 'rvint' else np.float32
                self.subsamples.add_column(
                    np.empty(shape, dtype=dtype), name=w, copy=False
                )

        if 'pid' in which or 'packedpid' in which:
            # TODO: the distiction between `which` and `unpack_bits` is getting a bit muddled
            if unpack_bits is False:
                unpack_bits = 'packedpid' if 'packedpid' in which else 'pid'
            # add any PID fields
            self.subsamples.update(
                bitpacked.empty_bitpacked_arrays(N_subsamp, unpack_bits),
                copy=False,
            )

        which_files = []
        if 'pos' in which or 'vel' in which or 'rvint' in which:
            which_files += ['rv']
        if 'pid' in which or 'packedpid' in which:
            which_files += ['pid']

        # The boundaries of each halo file in the self.halos table
        halo_file_offsets = np.empty(len(N_halo_per_file) + 1, dtype=np.uint64)
        util.cumsum(N_halo_per_file, halo_file_offsets, initial=True, final=True)

        if cleaned:
            # these will be reused
            clean_afs = [
                asdf.open(
                    self.clean_rvpid_dir / f'cleaned_rvpid_{i:03d}.asdf',
                    lazy_load=True,
                    memmap=False,
                )
                for i in self.superslab_inds
            ]

        for rvpid in which_files:
            colname = {'rv': 'rvint', 'pid': 'packedpid'}[rvpid]
            for AB in load_AB:
                for i in range(len(self.superslab_inds)):
                    fn = (
                        Path(self.groupdir)
                        / f'halo_{rvpid}_{AB}'
                        / f'halo_{rvpid}_{AB}_{self.superslab_inds[i]:03d}.asdf'
                    )
                    with asdf.open(fn, lazy_load=True, memmap=False) as af:
                        slab_particles = af[self.data_key][colname][:]
                    if cleaned:
                        clean_af = clean_afs[i]
                        clean_slab_particles = clean_af[self.data_key][
                            f'{colname}_{AB}'
                        ][:]

                    keys = [f'npstart{AB}', f'npout{AB}']
                    if cleaned:
                        keys += [f'npstart{AB}_merge', f'npout{AB}_merge']
                    slab_halos = {
                        k: self.halos[k][
                            halo_file_offsets[i] : halo_file_offsets[i + 1]
                        ]
                        for k in keys
                    }

                    # We grab an extra element at the end to know where the last halo ends
                    slab_write_offsets = npstartAB_new[AB][
                        halo_file_offsets[i] : halo_file_offsets[i + 1] + np.uint64(1)
                    ]

                    kwargs = {
                        'slab_read_offsets': slab_halos[f'npstart{AB}'],
                        'slab_read_lens': slab_halos[f'npout{AB}'],
                        'slab_write_offsets': slab_write_offsets,
                        'boxsize': self.header['BoxSize'],
                    }

                    if cleaned:
                        kwargs.update(
                            {
                                'clean_slab_read_offsets': slab_halos[
                                    f'npstart{AB}_merge'
                                ],
                                'clean_slab_read_lens': slab_halos[f'npout{AB}_merge'],
                            }
                        )

                    if rvpid == 'rv':
                        kwargs['slab_rvint'] = slab_particles
                        if cleaned:
                            kwargs['clean_slab_rvint'] = clean_slab_particles

                        kwargs['pos'] = self.subsamples.columns.get('pos')
                        kwargs['vel'] = self.subsamples.columns.get('vel')
                        kwargs['rvint'] = self.subsamples.columns.get('rvint')

                        self._unpack_rv_subsamples(**kwargs)
                    else:
                        kwargs['slab_packedpid'] = slab_particles
                        if cleaned:
                            kwargs['clean_slab_packedpid'] = clean_slab_particles

                        for pidfield in bitpacked.PID_FIELDS:
                            kwargs[pidfield] = self.subsamples.columns.get(pidfield)

                        kwargs['ppd'] = self.header['ppd']
                        self._unpack_pid_subsamples(**kwargs)

        if cleaned:
            for af in clean_afs:
                af.close()

    @staticmethod
    @numba.njit
    def _unpack_rv_subsamples(
        pos,
        vel,
        rvint,
        slab_rvint,
        slab_read_offsets,
        slab_read_lens,
        slab_write_offsets,
        boxsize,
        clean_slab_rvint=None,
        clean_slab_read_offsets=None,
        clean_slab_read_lens=None,
    ):
        # Zipper togther the original and cleaned subsamples on a per-halo basis.
        # Reads may not be contiguous (e.g. halos could be filtered out, and we skip L0),
        # but writes are.

        N_halo = len(slab_read_offsets)
        for i in range(N_halo):
            halo_rvint = slab_rvint[
                slab_read_offsets[i] : slab_read_offsets[i] + slab_read_lens[i]
            ]

            wstart = slab_write_offsets[i]
            wend = slab_write_offsets[i + 1]

            halo_posout = pos[wstart:wend] if pos is not None else None
            halo_velout = vel[wstart:wend] if vel is not None else None
            halo_rvintout = rvint[wstart:wend] if rvint is not None else None

            if rvint is not None:
                halo_rvintout[: len(halo_rvint)] = halo_rvint

            bitpacked._unpack_rvint(halo_rvint, boxsize, halo_posout, halo_velout)

            if clean_slab_rvint is not None:
                clean_halo_rvint = clean_slab_rvint[
                    clean_slab_read_offsets[i] : clean_slab_read_offsets[i]
                    + clean_slab_read_lens[i]
                ]

                # fast-forward the write index
                woff = slab_read_lens[i]

                if pos is not None:
                    halo_posout = halo_posout[woff:]
                if vel is not None:
                    halo_velout = halo_velout[woff:]
                if rvint is not None:
                    halo_rvintout = halo_rvintout[woff:]
                    halo_rvintout[: len(clean_halo_rvint)] = clean_halo_rvint
                bitpacked._unpack_rvint(
                    clean_halo_rvint, boxsize, halo_posout, halo_velout
                )

    @staticmethod
    @numba.njit
    def _unpack_pid_subsamples(
        pid,
        slab_packedpid,
        slab_read_offsets,
        slab_read_lens,
        slab_write_offsets,
        boxsize,
        ppd,
        clean_slab_packedpid=None,
        clean_slab_read_offsets=None,
        clean_slab_read_lens=None,
        lagr_pos=None,
        tagged=None,
        density=None,
        lagr_idx=None,
        packedpid=None,
    ):
        N_halo = len(slab_read_offsets)
        for i in range(N_halo):
            halo_packedpid = slab_packedpid[
                slab_read_offsets[i] : slab_read_offsets[i] + slab_read_lens[i]
            ]

            wstart = slab_write_offsets[i]
            wend = slab_write_offsets[i + 1]

            # Because these have different types and shapes, and can be None, it's hard to use a Numba dict
            halo_pidout = pid[wstart:wend] if pid is not None else None
            halo_lagr_posout = lagr_pos[wstart:wend] if lagr_pos is not None else None
            halo_taggedout = tagged[wstart:wend] if tagged is not None else None
            halo_densityout = density[wstart:wend] if density is not None else None
            halo_lagr_idxout = lagr_idx[wstart:wend] if lagr_idx is not None else None
            halo_packedpidout = (
                packedpid[wstart:wend] if packedpid is not None else None
            )

            if packedpid is not None:
                halo_packedpidout[: len(halo_packedpid)] = halo_packedpid

            bitpacked._unpack_pids(
                halo_packedpid,
                boxsize,
                ppd,
                pid=halo_pidout,
                lagr_pos=halo_lagr_posout,
                tagged=halo_taggedout,
                density=halo_densityout,
                lagr_idx=halo_lagr_idxout,
            )

            if clean_slab_packedpid is not None:
                clean_halo_packedpid = clean_slab_packedpid[
                    clean_slab_read_offsets[i] : clean_slab_read_offsets[i]
                    + clean_slab_read_lens[i]
                ]

                # fast-forward the write index
                woff = slab_read_lens[i]

                if pid is not None:
                    halo_pidout = halo_pidout[woff:]
                if lagr_pos is not None:
                    halo_lagr_posout = halo_lagr_posout[woff:]
                if tagged is not None:
                    halo_taggedout = halo_taggedout[woff:]
                if density is not None:
                    halo_densityout = halo_densityout[woff:]
                if lagr_idx is not None:
                    halo_lagr_idxout = halo_lagr_idxout[woff:]
                if packedpid is not None:
                    halo_packedpidout = halo_packedpidout[woff:]
                    halo_packedpidout[: len(clean_halo_packedpid)] = (
                        clean_halo_packedpid
                    )

                bitpacked._unpack_pids(
                    clean_halo_packedpid,
                    boxsize,
                    ppd,
                    pid=halo_pidout,
                    lagr_pos=halo_lagr_posout,
                    tagged=halo_taggedout,
                    density=halo_densityout,
                    lagr_idx=halo_lagr_idxout,
                )

    def _update_subsample_index_cols(self, npstartAB_new, load_AB='AB', cleaned=True):
        # Now that we've used the original npout/npstart columns to read the subsamples,
        # we can move the new indices into the old columns.

        for AB in load_AB:
            self.halos.remove_column(f'npstart{AB}')
            self.halos.remove_column(f'npout{AB}')
            if cleaned:
                self.halos.remove_column(f'npstart{AB}_merge')
                self.halos.remove_column(f'npout{AB}_merge')

            self.halos.add_column(
                npstartAB_new[AB][:-1], name=f'npstart{AB}', copy=False
            )

            # We knew the writes were contiguous, so we never computed npout{AB}_new
            # Reconstruct it here
            self.halos.add_column(
                np.diff(npstartAB_new[AB]).astype(np.uint32),
                name=f'npout{AB}',
                copy=False,
            )

        gc.collect()

    def _load_halo_lc_subsamples(self, which=['pos', 'vel', 'pid'], unpack_bits=False):
        # Halo LC subsamples are loaded separately because the data model is different
        # and way simpler: just one file, no slab divisions, no B particles, no unpacking, no cleaning.

        fn = Path(self.groupdir) / 'lc_pid_rv.asdf'

        with asdf.open(fn, lazy_load=True, memmap=False) as af:
            for w in which:
                self.subsamples.add_column(af[self.data_key][w][:], name=w, copy=False)

        if 'pid' in which and unpack_bits:
            self.subsamples.update(
                bitpacked.unpack_pids(
                    self.subsamples['pid'],
                    box=self.header['BoxSize'],
                    ppd=self.header['ppd'],
                    **{f: True for f in unpack_bits},
                ),
                copy=False,
            )

    def nbytes(self, halos=True, subsamples=True):
        """Return the memory usage of the big arrays: the halo catalog and the particle subsamples"""
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
    def _is_path_halo_lc(path):
        path = Path(path)
        return 'halo_light_cones' in str(path) or any(path.glob('lc_*.asdf'))

    def __repr__(self):
        # TODO: there's probably some more helpful info we could put in here
        # Formally, this is supposed to be unambiguous, but mostly we just want it to look good in a notebook
        lines = [
            'CompaSO Halo Catalog',
            '====================',
            f'{self.header["SimName"]} @ z={self.header["Redshift"]:.5g}',
        ]
        n_halo_field = len(self.halos.columns)
        n_subsamp_field = len(self.subsamples.columns)
        lines += [
            '-' * len(lines[-1]),
            f'     Halos: {len(self.halos):8.3g} halos,     {n_halo_field:3d} {"fields" if n_halo_field != 1 else "field "}, {self.nbytes(halos=True, subsamples=False) / 1e9:7.3g} GB',
            f'Subsamples: {len(self.subsamples):8.3g} particles, {n_subsamp_field:3d} {"fields" if n_subsamp_field != 1 else "field "}, {self.nbytes(halos=False, subsamples=True) / 1e9:7.3g} GB',
            f'Cleaned halos: {self.cleaned}',
            f'Halo light cone: {self.halo_lc}',
        ]
        return '\n'.join(lines)


####################################################################################################
# The following constants and functions relate to unpacking our compressed halo and particle formats
####################################################################################################

# Constants
EULER_ABIN = 45
EULER_TBIN = 11
EULER_NORM = 1.8477590650225735122  # 1/sqrt(1-1/sqrt(2))

INT16SCALE = 32000.0


# unpack the eigenvectors
def _unpack_euler16(bin_this):
    N = bin_this.shape[0]
    minor = np.zeros((N, 3))
    middle = np.zeros((N, 3))
    major = np.zeros((N, 3))

    cap = bin_this // EULER_ABIN
    iaz = bin_this - cap * EULER_ABIN  # This is the minor axis bin_this
    bin_this = cap
    cap = bin_this // (EULER_TBIN * EULER_TBIN)  # This is the cap
    bin_this = bin_this - cap * (EULER_TBIN * EULER_TBIN)

    it = (np.floor(np.sqrt(bin_this))).astype(int)
    # its = np.sum(np.isnan(it))

    ir = bin_this - it * it

    t = (it + 0.5) * (1.0 / EULER_TBIN)  # [0,1]
    r = (ir + 0.5) / (it + 0.5) - 1.0  # [-1,1]

    # We need to undo the transformation of t to get back to yy/zz
    t *= 1 / EULER_NORM
    t = t * np.sqrt(2.0 - t * t) / (1.0 - t * t)  # Now we have yy/zz

    yy = t
    xx = r * t
    # and zz=1
    norm = 1.0 / np.sqrt(1.0 + xx * xx + yy * yy)
    zz = norm
    yy *= norm
    xx *= norm  # These are now a unit vector

    # TODO: legacy code, rewrite
    major[cap == 0, 0] = zz[cap == 0]
    major[cap == 0, 1] = yy[cap == 0]
    major[cap == 0, 2] = xx[cap == 0]
    major[cap == 1, 0] = zz[cap == 1]
    major[cap == 1, 1] = -yy[cap == 1]
    major[cap == 1, 2] = xx[cap == 1]
    major[cap == 2, 0] = zz[cap == 2]
    major[cap == 2, 1] = xx[cap == 2]
    major[cap == 2, 2] = yy[cap == 2]
    major[cap == 3, 0] = zz[cap == 3]
    major[cap == 3, 1] = xx[cap == 3]
    major[cap == 3, 2] = -yy[cap == 3]

    major[cap == 4, 1] = zz[cap == 4]
    major[cap == 4, 2] = yy[cap == 4]
    major[cap == 4, 0] = xx[cap == 4]
    major[cap == 5, 1] = zz[cap == 5]
    major[cap == 5, 2] = -yy[cap == 5]
    major[cap == 5, 0] = xx[cap == 5]
    major[cap == 6, 1] = zz[cap == 6]
    major[cap == 6, 2] = xx[cap == 6]
    major[cap == 6, 0] = yy[cap == 6]
    major[cap == 7, 1] = zz[cap == 7]
    major[cap == 7, 2] = xx[cap == 7]
    major[cap == 7, 0] = -yy[cap == 7]

    major[cap == 8, 2] = zz[cap == 8]
    major[cap == 8, 0] = yy[cap == 8]
    major[cap == 8, 1] = xx[cap == 8]
    major[cap == 9, 2] = zz[cap == 9]
    major[cap == 9, 0] = -yy[cap == 9]
    major[cap == 9, 1] = xx[cap == 9]
    major[cap == 10, 2] = zz[cap == 10]
    major[cap == 10, 0] = xx[cap == 10]
    major[cap == 10, 1] = yy[cap == 10]
    major[cap == 11, 2] = zz[cap == 11]
    major[cap == 11, 0] = xx[cap == 11]
    major[cap == 11, 1] = -yy[cap == 11]

    # Next, we can get the minor axis
    az = (iaz + 0.5) * (1.0 / EULER_ABIN) * np.pi
    xx = np.cos(az)
    yy = np.sin(az)
    # print("az = %f, %f, %f\n", az, xx, yy)
    # We have to derive the 3rd coord, using the fact that the two axes
    # are perpendicular.

    eq2 = (cap // 4) == 2
    minor[eq2, 0] = xx[eq2]
    minor[eq2, 1] = yy[eq2]
    minor[eq2, 2] = (minor[eq2, 0] * major[eq2, 0] + minor[eq2, 1] * major[eq2, 1]) / (
        -major[eq2, 2]
    )
    eq4 = (cap // 4) == 0
    minor[eq4, 1] = xx[eq4]
    minor[eq4, 2] = yy[eq4]
    minor[eq4, 0] = (minor[eq4, 1] * major[eq4, 1] + minor[eq4, 2] * major[eq4, 2]) / (
        -major[eq4, 0]
    )
    eq1 = (cap // 4) == 1
    minor[eq1, 2] = xx[eq1]
    minor[eq1, 0] = yy[eq1]
    minor[eq1, 1] = (minor[eq1, 2] * major[eq1, 2] + minor[eq1, 0] * major[eq1, 0]) / (
        -major[eq1, 1]
    )
    minor *= 1.0 / np.linalg.norm(minor, axis=1).reshape(N, 1)

    middle = np.zeros((minor.shape[0], 3))
    middle[:, 0] = minor[:, 1] * major[:, 2] - minor[:, 2] * major[:, 1]
    middle[:, 1] = minor[:, 2] * major[:, 0] - minor[:, 0] * major[:, 2]
    middle[:, 2] = minor[:, 0] * major[:, 1] - minor[:, 1] * major[:, 0]
    middle *= 1.0 / np.linalg.norm(middle, axis=1).reshape(N, 1)
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

clean_dt = np.dtype(
    [
        ('npstartA_merge', np.int64),
        ('npstartB_merge', np.int64),
        ('npoutA_merge', np.uint32),
        ('npoutB_merge', np.uint32),
        ('N_total', np.uint32),
        ('N_merge', np.uint32),
        ('haloindex', np.uint64),
        ('is_merged_to', np.int64),
        ('haloindex_mainprog', np.int64),
        ('v_L2com_mainprog', np.float32, 3),
    ],
    align=True,
)

clean_dt_progen = np.dtype(
    [
        ('npstartA_merge', np.int64),
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
    ],
    align=True,
)

halo_lc_dt = np.dtype(
    [
        ('N', np.uint32),
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
    ],
    align=True,
)

user_dt = np.dtype(
    [
        ('id', np.uint64),
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
    ],
    align=True,
)
