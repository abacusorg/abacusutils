'''
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
'''

# TODO: generator to iterate over files
# TODO: load multiple files into concatenated table

from os.path import basename
import warnings

import numpy as np
from astropy.table import Table

from .bitpacked import unpack_rvint, unpack_pids
from .pack9 import unpack_pack9

__all__ = ['read_asdf']

ASDF_DATA_KEY = 'data'
ASDF_HEADER_KEY = 'header'

def read_asdf(fn, load=None, colname=None, dtype=np.float32, verbose=True, **kwargs):
    '''
    Read an Abacus ASDF file.  The result will be returned in an Astropy table.

    Parameters
    ----------
    fn: str
        The filename of the ASDF file to load
        
    load: list of str or None, optional
        A list of columns to load. The default (``None``) is to load columns based on
        what's in the file. If the file contains positions and velocities, those will
        be loaded; if it contains PIDs, those will be loaded.
        
        The list of fields that can be specified is: \
        ``'pos', 'vel', 'pid', 'lagr_pos', 'tagged', 'density', 'lagr_idx', 'aux'``
            
        All except ``pos`` & ``vel`` are PID-derived fields (see
        :func:`abacusnbody.data.bitpacked.unpack_pids`)
        
    colname: str or None, optional
        The internal column name in the ASDF file to load.  Probably one of ``'rvint'``,
        ``'packedpid'``, ``'pid'``, or ``'pack9'``.  In most cases, the name can be
        automatically detected, which is the default behavior (``None``).
    
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
        of the table contains the header.
    '''

    import asdf
    import asdf.compression
    try:
        asdf.compression.validate('blsc')
    except Exception as e:
        raise Exception("Abacus ASDF extension not properly loaded! \
                        Try reinstalling abacusutils: `pip install 'abacusutils>=1'`, \
                        or updating ASDF: `pip install 'asdf>=2.8'`") from e

    data_key = kwargs.get('data_key', ASDF_DATA_KEY)
    header_key = kwargs.get('header_key', ASDF_HEADER_KEY)

    with asdf.open(fn, lazy_load=True, copy_arrays=True) as af:
        if colname is None:
            _colnames = ['rvint', 'pack9', 'packedpid', 'pid']
            for cn in _colnames:
                if cn in af.tree[data_key]:
                    if colname is not None:
                        raise ValueError(f"More than one key of {_colnames} found in asdf file {fn}. Need to specify colname!")
                    colname = cn
            if colname is None:
                raise ValueError(f"Could not find any of {_colnames} in asdf file {fn}. Need to specify colname!")
        
        # determine what fields to unpack
        load = _resolve_columns(colname, load, kwargs)

        header = af.tree[header_key]
        data = af.tree[data_key][colname]

        Nmax = len(data)  # will shrink later

        # determine subsample fraction and add to header
        OutputType = header.get('OutputType', None)
        if OutputType == 'LightCone':
            if header['SimSet'] == 'AbacusSummit':
                SubsampleFraction = header['ParticleSubsampleA'] + header['ParticleSubsampleB']
                header['SubsampleFraction'] = SubsampleFraction
                if verbose:
                    print(f'Loading "{basename(fn)}", which contains the A and B subsamples ({int(SubsampleFraction*100):d}% total)')
        
        table = Table(meta=header)
        if 'pos' in load:
            table.add_column(np.empty((Nmax,3), dtype=dtype), copy=False, name='pos')
        if 'vel' in load:
            table.add_column(np.empty((Nmax,3), dtype=dtype), copy=False, name='vel')
        if 'aux' in load:
            table.add_column(data, copy=False, name='aux')  # 'aux' is the raw aux field
        # For the PID columns, we'll let `unpack_pids` build those for us
        # Eventually, we'll need to be able to pass output arrays
        
        if colname == 'rvint':
            _posout = table['pos'] if 'pos' in load else False
            _velout = table['vel'] if 'vel' in load else False
            npos,nvel = unpack_rvint(data, header['BoxSize'], float_dtype=dtype, posout=_posout, velout=_velout)
            nread = max(npos,nvel)
        elif colname == 'pack9':
            _posout = table['pos'] if 'pos' in load else False
            _velout = table['vel'] if 'vel' in load else False
            npos,nvel = unpack_pack9(data, header['BoxSize'], header['VelZSpace_to_kms'], float_dtype=dtype, posout=_posout, velout=_velout)
            nread = max(npos,nvel)
        elif 'pid' in colname:
            ppd = kwargs.get('ppd', int(round(header['ppd'])))
            pid_kwargs = {k:(k in load) for k in ('pid','lagr_pos','tagged','density','lagr_idx')}
            cols = unpack_pids(data, box=header['BoxSize'], ppd=ppd, float_dtype=dtype, **pid_kwargs)
            for n,col in cols.items():
                table.add_column(col, name=n, copy=False)
            nread = len(data)
            
    table = table[:nread]  # truncate to amount actually read
    # TODO: could drop some memory here
    
    return table


def _resolve_columns(colname, load, kwargs):
    '''Figure out what columns to read. `colname` is the data column in the file,
    `load` is the tuple of strings, `kwargs` might have deprecated load_pos/vel'''
    
    load_pos = kwargs.pop('load_pos', None)
    load_vel = kwargs.pop('load_vel', None)
    if load_pos is not None or load_vel is not None:
        if load is None:
            warnings.warn('`load_pos` and `load_vel` are deprecated; use '
                          '`load=("pos","vel")` instead.', FutureWarning)
            load = []
            if load_pos or (load_pos is None and load_vel is False):
                load += ['pos']
            if load_vel or (load_vel is None and load_pos is False):
                load += ['vel']
        else:
            warnings.warn('`load` and deprecated `load_pos` or `load_vel` specified. '
                          'Ignoring deprecated parameters.')
            
    if load is None:
        load = []
        if colname in ('pack9','rvint'):
            load += ['pos']
            load += ['vel']
        if 'pid' in colname:
            load += ['pid']
    return tuple(load)
