import tempfile
import filecmp
import os.path

import pytest

EXAMPLE_SIM = os.path.join(os.path.dirname(__file__), 'Mini_N64_L32')
HALOS_OUTPUT = os.path.join(os.path.dirname(__file__), 'halos.txt')
PARTICLES_OUTPUT = os.path.join(os.path.dirname(__file__), 'particles.txt')


def test_loading(tmp_path):
    '''Test loading a halo catalog
    '''

    from abacusnbody.hod.AbacusHOD import load_sims

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), load_subsamples=True, fields='all')

    with open(tmp_path/'halos_test.txt', 'w') as fp:
        f = cat.halos[::5].pformat_all()
        fp.write('\n'.join(f))

    assert filecmp.cmp(HALOS_OUTPUT,tmp_path/'halos_test.txt')

