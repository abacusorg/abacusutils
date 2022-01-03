#!/usr/bin/env python3

'''
Generate a halo or HOD 2PCF on a single AbacusSummit halo catalog.

$ ./generate_cf.py --help

'''

import os
import gc
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import Corrfunc
from astropy.table import Table
import asdf

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog


DEFAULT_NDENS = 1e-4
DEFAULT_NTHREAD = len(os.sched_getaffinity(0))  # guess based on affinity mask
DEFAULT_OUTDIR = '.'

log = lambda *args,**kwargs: print(*args,**kwargs,flush=True)

def prepare_cat(halo_cat_path, ndens):
    '''Load and downsample the cat
    '''
    # TODO: could use way less memory loading slab-by-slab
    cat = CompaSOHaloCatalog(halo_cat_path,
                             subsamples=False,
                             fields=('N', 'x_L2com'),
                             cleaned=False  # TODO
                            )
    log(f'Loading cat used {cat.nbytes()/1e9:.3g} GB')
    # Determine number of objects
    box = cat.header['BoxSize']
    N_select = int(box**3 * ndens)
    log(f'Selecting {N_select} objects')
    assert N_select > 0
    
    # Downsample catalog to N most massive
    iord = np.argsort(cat.halos['N'])[::-1]
    cat.halos = cat.halos[iord[:N_select]]
    del iord
    gc.collect()  # maybe can drop some memory
    
    return cat


def generate_cf(cat, nthread):
    '''Run corrfunc
    '''
    log(f'Using {nthread} threads')
    
    N = len(cat.halos)
    pos = cat.halos['x_L2com'].T
    box = cat.header['BoxSize']
    rbins = np.geomspace(0.1, 50, 13)
    cf = Corrfunc.theory.DD(autocorr=1, nthreads=nthread, binfile=rbins,
                            X1=pos[0], Y1=pos[1], Z1=pos[2],
                            periodic=True, boxsize=box,
                           )
    cf = Table(cf, meta=cat.header)
    RR = N*(N-1)/box**3 * 4/3*np.pi*np.diff(rbins**3)
    cf['xi'] = cf['npairs'] / RR - 1
    cf['rmid'] = (cf['rmin'] + cf['rmax'])/2.
    
    cf.meta['zname'] = Path(cat.groupdir).name
    
    return cf


def write_cf(cf, outdir, generate_cf_args=None):
    '''Write the result to disk
    '''
    outdir = Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    
    fn = outdir / f'{cf.meta["SimName"]}-{cf.meta["zname"]}-cf.asdf'
    
    af = asdf.AsdfFile(tree=dict(data=cf, generate_cf_args=generate_cf_args))
    af.write_to(fn)
    
    
def main(halo_cat_path, ndens=DEFAULT_NDENS, nthread=DEFAULT_NTHREAD, outdir=DEFAULT_OUTDIR):
    t0 = perf_counter()
    
    t1 = perf_counter()
    cat = prepare_cat(halo_cat_path, ndens)
    log(f'prepare_cat() took {perf_counter() - t1:.2f} seconds')
    
    t1 = perf_counter()
    cf = generate_cf(cat, nthread)
    log(f'generate_cf() took {perf_counter() - t1:.2f} seconds')
    
    generate_cf_args = dict(halo_cat_path=halo_cat_path, ndens=ndens, nthread=nthread)
    write_cf(cf, outdir,
             generate_cf_args=generate_cf_args)
    
    log(f'Total time: {perf_counter() - t0:.2f} seconds')


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter,
                        argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('halo_cat_path',
                        help='Path to the halo catalog redshift, like "AbacusSummit_base_c000_ph000/halos/z0.100/"',
                       )
    parser.add_argument('--ndens', type=float,
                        help='Number density of tracers to use, in (Mpc/h)^-3. Currently selects most massive tracers down to this abundance.',
                        default=DEFAULT_NDENS,
                       )
    parser.add_argument('--nthread', type=int,
                        help='Number of threads to use (primarily in pair counting)',
                        default=DEFAULT_NTHREAD,
                       )
    parser.add_argument('--outdir',
                        help='Directory in which to write the output file (will write to OUTDIR/SimName-z-cf.asdf)',
                        default=DEFAULT_OUTDIR,
                       )

    args = parser.parse_args()
    args = vars(args)

    main(**args)
