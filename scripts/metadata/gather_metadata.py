#!/usr/bin/env python3

import os
from pathlib import Path

import asdf
import click
from Abacus.InputFile import InputFile
from tqdm import tqdm

ABACUSSUMMIT = Path(os.getenv('CFS', '/global/cfs/cdirs')) / 'desi/cosmosim/Abacus'
COSMOLOGIES = Path(os.getenv('ABACUS')) / 'external/AbacusSummit/Cosmologies'
COSM_KEYS = ('A_s', 'alpha_s')

def get_state(fn):
    with open(fn) as fp:
        headerlines = fp.readlines()
    i = next(i for i,ell in enumerate(headerlines) if ell.startswith('#created'))
    statestr = ''.join(headerlines[i+1:])

    return dict(InputFile(str_source=statestr))

@click.command()
@click.option('--small', is_flag=True)
def main(small=False):
    '''Gather the simulation headers from the IC files.

    We use the IC files because they contain the growth tables
    and linear Pk, which the other headers don't.
    '''
    icdir = ABACUSSUMMIT / 'ic'

    if not small:
        simnames = sorted(list(ABACUSSUMMIT.glob('AbacusSummit_*'))) + \
            [ABACUSSUMMIT / 'small' / 'AbacusSummit_small_c000_ph3000']
    else:
        simnames = sorted(list((ABACUSSUMMIT / 'small').glob('AbacusSummit_*')))

    headers = {}
    for i,sim in enumerate(tqdm(simnames)):
        param = dict(InputFile(sim / 'abacus.par'))  # static
        ctag = sim.name.split('_')[-2][1:]  # '000'

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
                except Exception:
                    continue  # nothing!

            state[zdir.name] = {k:v for k,v in zheader.items() if k not in param}

        if 'small' in sim.name:
            _icdir = icdir / 'small'
        else:
            _icdir = icdir

        with asdf.open(_icdir / sim.name / 'ic_dens_N576.asdf', lazy_load=True, copy_arrays=True) as af:
            icparam = af['header'].copy()
            class_pk = af['CLASS_power_spectrum'].copy()

        icparam.update(param)  # conflicts revert to param
        param = icparam

        class_ini = dict(InputFile(COSMOLOGIES / f'abacus_cosm{ctag}' / 'CLASS.ini'))
        for k in COSM_KEYS:
            assert k not in param
            assert k not in state
            param[k] = class_ini[k]

        headers[sim.name] = {}
        headers[sim.name]['param'] = param
        headers[sim.name]['state'] = state
        headers[sim.name]['CLASS_power_spectrum'] = class_pk

    af = asdf.AsdfFile(tree=headers)
    af.write_to('headers.asdf')

if __name__ == '__main__':
    main()
