'''
This is an interface to read various Abacus file formats,
like ASDF, pack9, and RVint.

For particle-oriented access to the data, one can use
this interface.  For halo-oriented access (e.g. associating
halo particles with their host halo), one should use the
relevant halo module (like compaso_halo_catalog).

The decoding of the binary formats is generally contained
in other modules (e.g. bitpacked); this interface mainly
deals with the container formats and high-level logic of
file names, Astropy tables, etc.
'''

# TODO: generator to iterate over files
# TODO: load multiple files into concatenated table

from os.path import basename

import numpy as np
from astropy.table import Table

from .bitpacked import unpack_rvint, unpack_pids

ASDF_DATA_KEY = 'data'
ASDF_HEADER_KEY = 'header'

def read_asdf(fn, colname=None,
                load_pos=None, load_vel=None, load_pid=None,
                dtype=np.float32, **kwargs):
    '''
    Read an Abacus ASDF file.  The result will be returned
    in an Astropy table.

    Parameters
    ----------
    fn: str
        The filename of the ASDF file to load
    colname: str or None, optional
        The ASDF column name to load.  Probably one of
        'rvint', 'packedpid', or 'pack9'.  In most cases,
        the name can be automatically detected, which is
        the default behavior.
    load_pos: bool or None, optional
        Read and unpack the positions, and return them
        in the final table.  True or False toggles
        this on or off; None (the default) will load
        the positions if the file is an "rv" file.
    load_vel: optional
        Like load_pos, but for the velocities
    load_pid: optional
        Like load_pos, but for the particle IDs.
        Under development.
    dtype: np.dtype, optional
        The precision in which to unpack any floating
        point arrays.  Default: np.float32

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
    except:
        raise RuntimeError('Error: your ASDF installation does not support Blosc compression.  Please install the fork with Blosc support with the following command: "pip install git+https://github.com/lgarrison/asdf.git"')

    base = basename(fn)
    if load_pos is None:
        load_pos = 'rv' in base
    if load_vel is None:
        load_vel = 'rv' in base
    if load_pid is None:
        load_pid = 'pid' in base

    if load_pid:
        raise NotImplementedError('pid reading not finished')

    data_key = kwargs.get('data_key', ASDF_DATA_KEY)
    header_key = kwargs.get('data_key', ASDF_HEADER_KEY)

    with asdf.open(fn, lazy_load=True, copy_arrays=True) as af:
        if colname is None:
            _colnames = ['rvint', 'pack9', 'packedpid']
            for cn in _colnames:
                if cn in af.tree[data_key]:
                    if colname is not None:
                        raise ValueError(f"More than one key of {_colnames} found in asdf file {fn}. Need to specify colname!")
                    colname = cn
            if colname is None:
                raise ValueError(f"Could not find any of {_colnames} in asdf file {fn}. Need to specify colname!")

        header = af.tree[header_key]
        data = af.tree[data_key][colname]

        N = len(data)

        table = Table(meta=header)
        if load_pos:
            table.add_column(np.empty((N,3), dtype=dtype), copy=False, name='pos')
        if load_vel:
            table.add_column(np.empty((N,3), dtype=dtype), copy=False, name='vel')


        if colname == 'rvint':
            _posout = table['pos'] if load_pos else False
            _velout = table['vel'] if load_vel else False
            npos,nvel = unpack_rvint(data, header['BoxSize'], float_dtype=dtype, posout=_posout, velout=_velout)
            nread = max(npos,nvel)
        elif colname == 'pack9':
            raise NotImplementedError('pack9 via asdf not yet implemented')
        elif colname == 'packedpid':
            justpid, lagr_pos, tagged, density = unpack_pids(data, header['BoxSize'], header['ppd'], float_dtype=dtype)

    return table
