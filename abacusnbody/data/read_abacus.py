"""
This is an interface to read various Abacus file formats,
like ASDF, pack9, and RVint.

For particle-oriented access to the data, one can use
this interface.  For halo-oriented access (e.g. associating
halo particles with their host halo), one should use the
relevant halo module (like :mod:`abacusnbody.data.compaso_halo_catalog`).

The decoding of the binary formats is generally contained
in other modules (e.g. bitpacked); this interface mainly
deals with the container formats and high-level logic of
file names, Astropy tables, etc.

Per-format unpacking is routed through a small registry
(``_HANDLERS``) so adding a new format means writing one new
unpack module and adding one entry here.
"""

# TODO: generator to iterate over files
# TODO: load multiple files into concatenated table

import warnings
from os.path import basename

import numpy as np
from astropy.table import Table

from .bitpacked import unpack_pids, unpack_rvint
from .healstruct import LAYOUT as _HEALSTRUCT_LAYOUT, unpack_healstruct
from .maplog import unpack_maplog
from .output_particle import unpack_output_particle
from .pack9 import unpack_pack9

__all__ = ['read_asdf']

ASDF_DATA_KEY = 'data'
ASDF_HEADER_KEY = 'header'


_DEFAULT_LOAD = {
    'rvint': ('pos', 'vel'),
    'pack9': ('pos', 'vel'),
    'pid': ('pid',),
    'packedpid': ('pid',),
    'lightcone_particle': ('pos', 'vel', 'is_map', 'mult'),
    'timeslice_subsample': ('pos', 'vel', 'is_map', 'mult'),
    'lightcone_healpix': ('pixel', 'count'),
    'maplogs': ('pos', 'vel', 'mult', 'control'),
}


def read_asdf(fn, load=None, colname=None, dtype=np.float32, verbose=True, **kwargs):
    """
    Read an Abacus ASDF file.  The result will be returned in an Astropy table.

    Parameters
    ----------
    fn: str
        The filename of the ASDF file to load

    load: list of str or None, optional
        A list of columns to load. The default (``None``) is to load columns based on
        what's in the file. If the file contains positions and velocities, those will
        be loaded; if it contains PIDs, those will be loaded.

        For AbacusSummit formats (``rvint``, ``pack9``, ``pid``, ``packedpid``), the
        valid load keys are: ``'pos', 'vel', 'pid', 'lagr_pos', 'tagged', 'density',
        'lagr_idx', 'aux'``.

        For Aurora ``output_particle`` / ``lightcone_particle``, the valid load keys
        are: ``'pos', 'vel', 'pid', 'density', 'vel_disp', 'is_map', 'mult',
        'rel_vel'``.

        For Aurora ``lightcone_healpix``, the valid load keys are:
        ``'pixel', 'dist_bin', 'count', 'healstruct', 'voxel_id'``.

        For Aurora ``maplogs``, the valid load keys are: ``'pos', 'vel',
        'mult', 'control', 'node_type', 'timestep', 'mult_sec', 'pid',
        'density', 'vel_disp', 'length', 'vel_rel', 'lc_label'``.

    colname: str or None, optional
        The internal column name in the ASDF file to load.  Probably one of ``'rvint'``,
        ``'packedpid'``, ``'pid'``, ``'pack9'``, ``'output_particle'``,
        ``'lightcone_particle'``, ``'lightcone_healpix'``, or ``'maplogs'``.
        In most cases, the name can be automatically detected, which is the default
        behavior (``None``).

    dtype: np.dtype, optional
        The precision in which to unpack any floating
        point arrays.  Default: np.float32

    verbose: bool, optional
        Print informational messages. Default: True

    Returns
    -------
    table: astropy.Table
        A table whose columns contain the particle
        data from the ASDF file.  The ``meta`` field
        of the table contains the header.  For files
        with a single top-level ``header`` section
        (AbacusSummit), ``meta`` is that section
        flattened. For files with multiple ``header*``
        sections (Aurora), ``meta`` is a dict-of-dicts
        keyed by section name (e.g. ``'header_read'``,
        ``'header_write'``).
    """

    import asdf

    try:
        import asdf._compression as asdf_compression
    except ImportError:
        import asdf.compression as asdf_compression

    try:
        asdf_compression.validate('blsc')
    except Exception as e:
        raise Exception(
            "Abacus ASDF extension not properly loaded! \
                        Try reinstalling abacusutils: `pip install 'abacusutils>=1'`, \
                        or updating ASDF: `pip install 'asdf>=2.8'`"
        ) from e

    data_key = kwargs.get('data_key', ASDF_DATA_KEY)
    # `header_key` is honored explicitly for back-compat; otherwise we scan for `header*`
    header_key = kwargs.pop('header_key', None)

    with asdf.open(fn, lazy_load=True, memmap=False) as af:
        if colname is None:
            _colnames = list(_DEFAULT_LOAD.keys())
            for cn in _colnames:
                if cn in af.tree[data_key]:
                    if colname is not None:
                        raise ValueError(
                            f'More than one key of {_colnames} found in asdf file {fn}. Need to specify colname!'
                        )
                    colname = cn
            if colname is None:
                raise ValueError(
                    f'Could not find any of {_colnames} in asdf file {fn}. Need to specify colname!'
                )

        # determine what fields to unpack
        load = _resolve_columns(colname, load, kwargs)

        if header_key is not None:
            meta = af.tree[header_key]
            header = meta
        else:
            meta, header = _gather_headers(af.tree)

        data = af.tree[data_key][colname]

        # determine subsample fraction and add to header (Summit-only)
        OutputType = header.get('OutputType', None)
        if OutputType == 'LightCone' and header.get('SimSet') == 'AbacusSummit':
            SubsampleFraction = (
                header['ParticleSubsampleA'] + header['ParticleSubsampleB']
            )
            header['SubsampleFraction'] = SubsampleFraction
            if verbose:
                print(
                    f'Loading "{basename(fn)}", which contains the A and B subsamples ({int(SubsampleFraction * 100):d}% total)'
                )

        handler = _HANDLERS[colname]
        handler_kwargs = {k: kwargs[k] for k in ('ppd',) if k in kwargs}
        cols, nread = handler(data, header, load, dtype, **handler_kwargs)

        table = Table(meta=meta)
        for name, col in cols.items():
            table.add_column(col, name=name, copy=False)

    table = table[:nread]  # truncate to amount actually read
    # TODO: could drop some memory here

    return table


def _gather_headers(tree):
    """
    Collect top-level ``header*`` sections from an ASDF tree.

    Returns
    -------
    meta : dict
        Value to use as ``table.meta``. Exactly one section → that section's dict
        (flat, matching the legacy single-``header`` behavior). Multiple sections
        → a dict mapping section name to section (nested).
    header : dict
        Flat dict for field lookup by handlers. Same object as ``meta`` when there
        is a single section (so mutations propagate). For multiple sections this
        is a fresh union of the sections, first-match-wins on conflict.
    """
    sections = {k: v for k, v in tree.items() if k.startswith('header')}
    if len(sections) == 1:
        section = next(iter(sections.values()))
        return section, section
    union = {}
    for section in sections.values():
        for k, v in section.items():
            union.setdefault(k, v)
    return sections, union


def _handle_rvint(data, header, load, dtype, **_unused):
    cols = {}
    if 'pos' in load:
        cols['pos'] = np.empty((len(data), 3), dtype=dtype)
    if 'vel' in load:
        cols['vel'] = np.empty((len(data), 3), dtype=dtype)
    posout = cols.get('pos', False)
    velout = cols.get('vel', False)
    npos, nvel = unpack_rvint(
        data, header['BoxSize'], float_dtype=dtype, posout=posout, velout=velout
    )
    return cols, max(npos, nvel)


def _handle_pack9(data, header, load, dtype, **_unused):
    cols = {}
    if 'pos' in load:
        cols['pos'] = np.empty((len(data), 3), dtype=dtype)
    if 'vel' in load:
        cols['vel'] = np.empty((len(data), 3), dtype=dtype)
    posout = cols.get('pos', False)
    velout = cols.get('vel', False)
    npos, nvel = unpack_pack9(
        data,
        header['BoxSize'],
        header['VelZSpace_to_kms'],
        float_dtype=dtype,
        posout=posout,
        velout=velout,
    )
    return cols, max(npos, nvel)


def _handle_pid(data, header, load, dtype, *, ppd=None, **_unused):
    if ppd is None:
        ppd = int(round(header['ppd']))
    pid_kwargs = {
        k: (k in load) for k in ('pid', 'lagr_pos', 'tagged', 'density', 'lagr_idx')
    }
    cols = unpack_pids(
        data, box=header['BoxSize'], ppd=ppd, float_dtype=dtype, **pid_kwargs
    )
    if 'aux' in load:
        cols['aux'] = data
    return cols, len(data)


def _handle_output_particle(data, header, load, dtype, **_unused):
    cols = unpack_output_particle(data, fields=load, float_dtype=dtype)
    return cols, len(data)


def _handle_healstruct(data, header, load, dtype, **_unused):
    layout = header.get('HealStructLayout')
    if layout is not None and layout != _HEALSTRUCT_LAYOUT:
        raise ValueError(
            f'HealStructLayout {layout!r} in file does not match the only '
            f'layout this reader supports ({_HEALSTRUCT_LAYOUT!r})'
        )
    cols = unpack_healstruct(data, fields=load)
    return cols, len(data)


def _handle_maplog(data, header, load, dtype, **_unused):
    cols = unpack_maplog(data, fields=load, float_dtype=dtype)
    return cols, len(data)


_HANDLERS = {
    'rvint': _handle_rvint,
    'pack9': _handle_pack9,
    'pid': _handle_pid,
    'packedpid': _handle_pid,
    'lightcone_particle': _handle_output_particle,
    'timeslice_subsample': _handle_output_particle,
    'lightcone_healpix': _handle_healstruct,
    'maplogs': _handle_maplog,
}


def _resolve_columns(colname, load, kwargs):
    """Figure out what columns to read. `colname` is the data column in the file,
    `load` is the tuple of strings, `kwargs` might have deprecated load_pos/vel"""

    load_pos = kwargs.pop('load_pos', None)
    load_vel = kwargs.pop('load_vel', None)
    if load_pos is not None or load_vel is not None:
        if load is None:
            warnings.warn(
                '`load_pos` and `load_vel` are deprecated; use '
                '`load=("pos","vel")` instead.',
                FutureWarning,
            )
            load = []
            if load_pos or (load_pos is None and load_vel is False):
                load += ['pos']
            if load_vel or (load_vel is None and load_pos is False):
                load += ['vel']
        else:
            warnings.warn(
                '`load` and deprecated `load_pos` or `load_vel` specified. '
                'Ignoring deprecated parameters.'
            )

    if load is None:
        load = _DEFAULT_LOAD[colname]
    return tuple(load)
