#!/usr/bin/env python3

import os
from pathlib import Path

import click
import asdf
from tqdm import tqdm

from Abacus.InputFile import InputFile

ABACUSSUMMIT = Path(os.getenv('CFS',' /global/cfs/cdirs')) / 'desi/cosmosim/Abacus'

def get_state(fn):
    with open(fn) as fp:
        headerlines = fp.readlines()
    i = next(i for i,l in enumerate(headerlines) if l.startswith('#created'))
    statestr = ''.join(headerlines[i+1:])

    return dict(InputFile(str_source=statestr))

@click.command()
def main():
    '''Gather the simulation headers from the IC files.

    We use the IC files because they contain the growth tables, which
    the other headers don't.
    '''
    icdir = ABACUSSUMMIT / 'ic'

    simnames = sorted(list(ABACUSSUMMIT.glob('AbacusSummit_*')))

    headers = {}
    for i,sim in enumerate(tqdm(simnames)):
        param = dict(InputFile(sim / 'abacus.par'))  # static

        state = {}
        for zdir in (sim / 'halos').glob('z*'):
            try:
                afn = next(zdir.glob('*/*.asdf'))  # any asdf file
                with asdf.open(afn, lazy_load=True, copy_arrays=True) as af:
                    zheader = af['header'].copy()
            except StopIteration:
                # maybe a header?
                try:
                    zheader = dict(InputFile(zdir / 'header'))
                except:
                    continue  # nothing!
            
            state[zdir.name] = {k:v for k,v in zheader.items() if k not in param}

        with asdf.open(icdir / sim.name / 'ic_dens_N576.asdf', lazy_load=True, copy_arrays=True) as af:
            icparam = af['header'].copy()
            class_pk = af['CLASS_power_spectrum'].copy()
            icparam.update(param)  # conflicts revert to param
            param = icparam
        
        headers[sim.name] = {}
        headers[sim.name]['param'] = param
        headers[sim.name]['state'] = state
        headers[sim.name]['CLASS_power_spectrum'] = class_pk
    
    af = asdf.AsdfFile(tree=headers)
    af.write_to('headers.asdf')

if __name__ == '__main__':
    main()
