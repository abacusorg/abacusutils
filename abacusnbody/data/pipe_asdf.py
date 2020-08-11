#!/usr/bin/env python3

'''
``pipe_asdf`` is a Python script to unpack Abacus ASDF files (such as
halo catalog or particle data) and write them out via a Unix pipe (stdout).
The intention is to provide a simple way for C, C++, Fortran, etc,
codes to read ASDF files while letting Python handle the details
of the file formats, compression, and other things Python does well.

Usage
=====
.. code-block:: bash

    pipe_asdf [-h] [-f FIELD] [--nthread NTHREAD] asdf-file [asdf-file ...] | ./client

positional arguments
--------------------
  asdf-file
    An ASDF file. Multiple may be specified.

optional arguments
------------------
  -h, --help            show this help message and exit
  -f FIELD, --field FIELD
                        A field/column to pipe. Multiple -f flags are allowed, in which case fields will be piped in the order they are specified. (default:
                        None)
  --nthread NTHREAD     Number of blosc decompression threads (when applicable). For AbacusSummit, use 1 to 4. (default: 4)



Binary Format of Piped Data
===========================

The binary format of the piped data is simple:

1) an 8-byte int indicating the number of data values
2) a 4-byte int indicating the width of the primitive data type that composes the data
   (e.g. 4 for float, 8 for double).  Largely provided as a sanity check.
3) the data, consisting of a number of bytes equal to the product of the preceeding ints
4) Repeat from (1) for all fields requested

So the expected pattern for the client code is to read the int64 and int32,
take the product, allocate that many bytes, then read the data into that allocation.

When passing multiple files, a single column will be read from all files
before moving to the next column.  In other words, the client sees
the concatenated data.

From a performance perspective, the pipe operation probably amounts
to a memcpy. So a small performance hit, but likely vanishingly small
compared to the actual IO and analysis.

Ultimately, this pipe scheme is not a replacement for direct access
to the files, but it may be helpful for applications with simple data
access patterns.

Entry Points
============
Technically, ``pipe_asdf`` is a "console script" alias provided by setuptools to invoke
the ``abacusnbody.data.pipe_asdf`` module as a script.  This alias is
usually installed in a user's PATH environment variable when installing
abacusutils via pip, but if not, one could equivalently invoke the
script with:

.. code-block:: bash

    $ python3 -m abacusnbody.data.pipe_asdf

The ``abacusnbody/pipe_asdf`` directory also contains a symlink to this
file, so from this directory one can also run

.. code-block:: bash
    
    $ ./pipe_asdf.py

To-do
======
- Add a "-k/--key" flag to read header fields. Decide on a wire protocol.
- Add CompaSOHaloCatalog hooks to pipe the unpacked data (?)
'''

import sys
import argparse
from os.path import isfile, join as pjoin
import warnings
import gc
import time
from timeit import default_timer as timer

import asdf
import numpy as np

import asdf.compression
try:
    asdf.compression.validate('blsc')
except:
    # Note: this is a temporary solution until blosc is integrated into ASDF, or until we package a pluggable decompressor
    warnings.warn('Your ASDF installation does not support Blosc compression.  May not be able to read AbacusSummit ASDF files.  Please install the fork with Blosc support with the following command: "pip install git+https://github.com/lgarrison/asdf.git"')

DEFAULT_DATA_KEY = 'data'
DEFAULT_HEADER_KEY = 'header'

def unpack_to_pipe(asdf_fns, fields, data_key=DEFAULT_DATA_KEY, header_key=DEFAULT_HEADER_KEY, pipe=sys.stdout.buffer, nthread=4,
                    verbose=True):
    if pipe.isatty():
        raise RuntimeError('Output pipe appears to be a terminal! Did you mean to pipe or redirect stdout?')

    # begin input validation and header reads
    assert pipe is not None  # can this happen?
    for fn in asdf_fns:
        if not isfile(fn):
            raise FileNotFoundError(fn)
    afs = []
    for fn in asdf_fns:
        afs += [asdf.open(fn, mode='r', copy_arrays=True, lazy_load=True)]
    for af in afs:
        for field in fields:
            if field not in af.tree[data_key]:
                raise ValueError(f'Field "{field}" not found in "{af.uri}"')

    # begin IO loop
    nbytes_tot = 0
    start_time = timer()
    read_time = 0
    for field in fields:
        N = np.int64(0)
        for af in afs:
            _N = np.prod(af[data_key][field].shape)
            N += _N
            field_width = np.int32(af[data_key][field].dtype.itemsize)
        pipe.write(N)
        pipe.write(field_width)
        for af in afs:
            read_start_time = timer()
            arr = af[data_key][field][:]  # read + decompression happens here
            read_time += timer() - read_start_time
            pipe.write(arr)
            del arr; gc.collect()
        nbytes_tot += N*field_width
    pipe.close()  # signal EOF
    tot_time = timer() - start_time
    if verbose:
        print(f'[pipe_asdf.py] Read + decompressed {nbytes_tot/1e6:.3g} MB in {read_time:.3g} s at {nbytes_tot/1e6/read_time:.3g} MB/s', file=sys.stderr)
        print(f'[pipe_asdf.py] Processed {nbytes_tot/1e6:.3g} MB in {tot_time:.3g} s at {nbytes_tot/1e6/tot_time:.3g} MB/s', file=sys.stderr)



class _ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

def main():
    '''Invoke the command-line interface'''
    parser = argparse.ArgumentParser(description='A script to unpack Abacus ASDF files and write the raw data to stdout. '
                                                    'See https://abacusutils.readthedocs.io/en/latest/pipes.html',
                                        formatter_class=_ArgParseFormatter)

    parser.add_argument('asdf-file', help='An ASDF file. Multiple may be specified.', nargs='+')
    parser.add_argument('-f', '--field', help='A field/column to pipe. Multiple -f flags are allowed, in which case fields will be piped in the order they are specified.', action='append')
    parser.add_argument('--nthread', help='Number of blosc decompression threads (when applicable).  For AbacusSummit, use 1 to 4.', type=int, default=4)

    args = parser.parse_args()
    args = vars(args)

    # rename a few args
    args['asdf_fns'] = args.pop('asdf-file')
    args['fields'] = args.pop('field')

    unpack_to_pipe(**args)


if __name__ == '__main__':
    main()
