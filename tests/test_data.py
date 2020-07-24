import os.path
EXAMPLE_SIM = os.path.join(os.path.dirname(__file__), 'Mini_N64_L32')
HALOS_OUTPUT = os.path.join(os.path.dirname(__file__), 'halos.txt')
PARTICLES_OUTPUT = os.path.join(os.path.dirname(__file__), 'particles.txt')

import tempfile
import filecmp

def test_data_compaso_halo_catalog(tmp_path):
    '''
    Test abacusnbody.data.compaso_halo_catalog
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), load_subsamples=True, fields='all')

    with open(tmp_path/'halos_test.txt', 'w') as fp:
        f = cat.halos[::5].pformat_all()
        fp.write('\n'.join(f))

    assert filecmp.cmp(HALOS_OUTPUT,tmp_path/'halos_test.txt')

    with open(tmp_path/'particles_test.txt', 'w') as fp:
        f = cat.subsamples[::50].pformat_all()
        fp.write('\n'.join(f))

    assert filecmp.cmp(PARTICLES_OUTPUT,tmp_path/'particles_test.txt')

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), fields=['N','x_com'])
    assert(set(cat.halos.colnames) == set(['N','x_com']))
