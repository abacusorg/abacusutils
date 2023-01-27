#!/usr/bin/env python3

import json
import pickle
from pathlib import Path

import asdf
import click
import numpy as np


@click.command()
@click.argument('fn')
@click.option('--rmstate', default=False, is_flag=True)
@click.option('--rmpk', default=False, is_flag=True)
@click.option('--pickle', 'dopickle', default=False, is_flag=True)
@click.option('--msgpack', 'domsgpack', default=True, is_flag=True)
@click.option('--json', 'dojson', default=False, is_flag=True)
def compress(fn, rmstate, rmpk, dopickle, domsgpack, dojson):
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
        for sim in meta:
            meta[sim]['state'] = np.frombuffer(pickle.dumps(meta[sim]['state']), dtype=np.byte)
            meta[sim]['param'] = np.frombuffer(pickle.dumps(meta[sim]['param']), dtype=np.byte)

        # p = pickle.dumps(meta)
        # meta = dict(pickle=np.frombuffer(p, dtype=np.byte))

    if domsgpack:
        import msgpack
        for sim in meta:
            meta[sim]['state'] = np.frombuffer(msgpack.dumps(meta[sim]['state']), dtype=np.byte)
            meta[sim]['param'] = np.frombuffer(msgpack.dumps(meta[sim]['param']), dtype=np.byte)

    if dojson:
        for sim in meta:
            meta[sim]['state'] = np.frombuffer(json.dumps(meta[sim]['state']).encode(), dtype=np.byte)
            meta[sim]['param'] = np.frombuffer(json.dumps(meta[sim]['param']).encode(), dtype=np.byte)

        # p = pickle.dumps(meta)
        # meta = dict(pickle=np.frombuffer(p, dtype=np.byte))

    if not rmpk:
        # de-dup
        for i,sim1 in enumerate(meta):
            pk1 = meta[sim1]['CLASS_power_spectrum']
            for j,sim2 in list(enumerate(meta))[i+1:]:
                pk2 = meta[sim2]['CLASS_power_spectrum']
                for col in pk1.colnames:
                    if np.array_equal(pk1[col], pk2[col]):
                        pk2.replace_column(col, pk1[col], copy=False)


    with asdf.AsdfFile(tree=meta) as af:
        # the following needs https://github.com/astropy/asdf-astropy/issues/156
        # for sim in meta:
        #     for k in ('state','param'):
        #         af.set_array_compression(af[sim][k], 'blsc',
        #             shuffle='shuffle', compression_block_size=1<<15, blosc_block_size=1<<30, clevel=9, cname='zstd',
        #         )
            # if 'CLASS_power_spectrum' in af[sim]:
            #     for k in ('k (h/Mpc)', 'P (Mpc/h)^3'):
            #         af.set_array_compression(af[sim]['CLASS_power_spectrum'][k], 'lz4',
            #             shuffle='shuffle', compression_block_size=1<<22, blosc_block_size=1<<18, clevel=9, cname='zstd',
            #             typesize=8,
            #         )
        af.write_to(fn.parent / (fn.stem + "_compressed.asdf"),
            all_array_compression='blsc',
            compression_kwargs = dict(
                shuffle='shuffle',
                compression_block_size=1<<22,
                blosc_block_size=1<<20,
                clevel=9,
                cname='zstd',
                #typesize=16,
            ),
        )

if __name__ == '__main__':
    compress()
