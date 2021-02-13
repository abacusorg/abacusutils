import tempfile
import filecmp
import os.path

import pytest
from astropy.table import Table
import numpy as np

EXAMPLE_SIM = os.path.join(os.path.dirname(__file__), 'Mini_N64_L32')
HALOS_OUTPUT_UNCLEAN = os.path.join(os.path.dirname(__file__), 'test_halos_unclean.asdf')
PARTICLES_OUTPUT_UNCLEAN = os.path.join(os.path.dirname(__file__), 'test_subsamples_unclean.asdf')

def test_halos_unclean(tmp_path):
    '''Test loading a base (uncleaned) halo catalog
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), subsamples=True, fields='all', cleaned_halos=False)
    
    # to regenerate reference
    #ref = Table(cat.halos[::10])
    #ref.write('test_halos_unclean.asdf', all_array_storage='internal')
    
    ref = Table.read(HALOS_OUTPUT_UNCLEAN)
    
    halos = cat.halos[::10]
    for col in ref.colnames:
        if col == 'npstartB':
            continue
        assert np.allclose(halos[col], ref[col])
   
    assert halos.meta == ref.meta


def test_subsamples_unclean(tmp_path):
    '''Test loading particle subsamples
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), subsamples=True, fields='all', cleaned_halos=False)
    
    # to regenerate reference
    #ref = Table(cat.subsamples[::10])
    #ref.write('test_subsamples_unclean.asdf', format='asdf', all_array_storage='internal')
    
    ref = Table.read(PARTICLES_OUTPUT_UNCLEAN)
    
    ss = cat.subsamples[::10]
    for i in range(len(cat.halos)):
        h = cat.halos[i]
        
        _ss = ss[h['npstartA']:h['npstartA']+h['npoutA']]
        _ref = ref[h['npstartA']:h['npstartA']+h['npoutA']]
        for col in _ss.colnames:
            assert np.allclose(_ss[col], _ref[col])
        
        _ss = ss[h['npstartB']:h['npstartB']+h['npoutB']]
        _ref = ref[h['npstartB']:h['npstartB']+h['npoutB']]
        for col in _ss.colnames:
            assert np.allclose(_ss[col], _ref[col])
   
    assert cat.subsamples.meta == ref.meta


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
        subsamples=True)
    assert len(cat.halos) == 127
    assert len(cat.subsamples) == 9306

    
def test_halo_info_list():
    '''Test list of halo infos
    '''
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        [os.path.join(EXAMPLE_SIM, 'halos', 'z0.000', 'halo_info', 'halo_info_000.asdf'),
         os.path.join(EXAMPLE_SIM, 'halos', 'z0.000', 'halo_info', 'halo_info_001.asdf')],
        subsamples=True)
    assert len(cat.halos) == 281
    assert len(cat.subsamples) == 19555
    
    # check fail on dups
    with pytest.raises(ValueError):
        cat = CompaSOHaloCatalog(
        [os.path.join(EXAMPLE_SIM, 'halos', 'z0.000', 'halo_info', 'halo_info_000.asdf'),
         os.path.join(EXAMPLE_SIM, 'halos', 'z0.000', 'halo_info', 'halo_info_000.asdf')])


def test_unpack_bits():
    '''Test unpack_bits
    '''
    
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    from abacusnbody.data.bitpacked import PID_FIELDS
    
    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), subsamples=True, unpack_bits=True, fields='N')
    assert set(PID_FIELDS) <= set(cat.subsamples.colnames)  # check subset
    
    cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), subsamples=True, unpack_bits='density', fields='N')
    assert 'density' in cat.subsamples.colnames
    assert 'lagr_pos' not in cat.subsamples.colnames  # too many?
    
    # bad bits field name
    with pytest.raises(ValueError):
        cat = CompaSOHaloCatalog(os.path.join(EXAMPLE_SIM, 'halos', 'z0.000'), subsamples=True, unpack_bits=['blah'], fields='N')
