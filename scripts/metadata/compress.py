#!/usr/bin/env python3

from pathlib import Path
import pickle

import click
import asdf
import numpy as np

@click.command()
@click.argument('fn')
@click.option('--rmstate', default=False, is_flag=True)
@click.option('--rmpk', default=False, is_flag=True)
@click.option('--pickle', 'dopickle', default=False, is_flag=True)
def compress(fn, rmstate, rmpk, dopickle):
    '''Compress metadata file FN
    '''
    fn = Path(fn)

    with asdf.open(fn, copy_arrays=True, lazy_load=False) as af:
        meta = dict(af.tree)

    del meta['history'], meta['asdf_library']
    for k in meta:
        if rmstate:
            del meta[k]['state']
        if rmpk:
            del meta[k]['CLASS_power_spectrum']

    if dopickle:
        p = pickle.dumps(meta)
        meta = dict(pickle=np.frombuffer(p, dtype=np.byte))
    
    with asdf.AsdfFile(tree=meta) as af:
        af.write_to(fn.parent / (fn.stem + "_compressed.asdf"),
        all_array_compression='blsc',
        compression_kwargs=dict(shuffle=None, compression_block_size=1<<30, blosc_block_size=1<<30, clevel=9),
        )

if __name__ == '__main__':
    compress()
