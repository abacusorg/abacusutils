import os.path
EXAMPLE_SIM = os.path.join(os.path.dirname(__file__), 'Mini_N64_L32')
HALOS_OUTPUT = os.path.join(os.path.dirname(__file__), 'halos.txt')
PARTICLES_OUTPUT = os.path.join(os.path.dirname(__file__), 'particles.txt')

import tempfile
import filecmp

def test_halos(tmp_path):
    '''Test loading a halo catalog
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), load_subsamples=True, fields='all')

    with open(tmp_path/'halos_test.txt', 'w') as fp:
        f = cat.halos[::5].pformat_all()
        fp.write('\n'.join(f))

    assert filecmp.cmp(HALOS_OUTPUT,tmp_path/'halos_test.txt')


def test_subsamples(tmp_path):
    '''Test loading particle subsamples
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), load_subsamples=True, fields='all')

    with open(tmp_path/'particles_test.txt', 'w') as fp:
        f = cat.subsamples[::50].pformat_all()
        fp.write('\n'.join(f))

    assert filecmp.cmp(PARTICLES_OUTPUT,tmp_path/'particles_test.txt')


def test_field_subset_loading():
    '''Test loading a subset of halo catalog columns
    '''
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), fields=['N','x_com'])
    assert set(cat.halos.colnames) == set(['N','x_com'])


def test_one_halo_info():
    '''Test loading a single halo_info file
    '''
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000', 'halo_info', 'halo_info_000.asdf'),
        load_subsamples=True)
    assert len(cat.halos) == 127
    assert len(cat.subsamples) == 9306
