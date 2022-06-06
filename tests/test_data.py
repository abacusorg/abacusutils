'''
This module tests `abacusnbody.data`: the basic interfaces to read halos and particles.
The reference files are stored in `tests/ref_data`.
'''

from pathlib import Path

import pytest
from astropy.table import Table
import numpy as np

from common import check_close

curdir = Path(__file__).parent
refdir = curdir / 'ref_data'
EXAMPLE_SIM = curdir / 'Mini_N64_L32'
HALOS_OUTPUT_UNCLEAN = refdir / 'test_halos_unclean.asdf'
PARTICLES_OUTPUT_UNCLEAN = refdir / 'test_subsamples_unclean.asdf'
HALOS_OUTPUT_CLEAN = refdir / 'test_halos_clean.asdf'
PARTICLES_OUTPUT_CLEAN = refdir / 'test_subsamples_clean.asdf'
PACK9_OUTPUT = refdir / 'test_pack9.asdf'
PACK9_PID_OUTPUT = refdir / 'test_pack9_pid.asdf'

def test_halos_unclean(tmp_path):
    '''Test loading a base (uncleaned) halo catalog
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=True, fields='all', cleaned=False)

    # to regenerate reference
    #ref = cat.halos
    #ref.write(HALOS_OUTPUT_UNCLEAN, all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(HALOS_OUTPUT_UNCLEAN)

    halos = cat.halos
    for col in ref.colnames:
        assert check_close(ref[col], halos[col])

    assert halos.meta == ref.meta

def test_halos_clean(tmp_path):
    '''Test loading a base (uncleaned) halo catalog
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=True, fields='all', cleaned=True)

    # to regenerate reference
    #ref = cat.halos
    #ref.write(HALOS_OUTPUT_CLEAN, all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(HALOS_OUTPUT_CLEAN)

    halos = cat.halos
    for col in ref.colnames:
        assert check_close(ref[col], halos[col])

    # all haloindex values should point to this slab
    assert np.all((halos['haloindex']/1e12).astype(int) == cat.header['FullStepNumber'])
    # ensure that all deleted halos in ref are marked as merged in EXAMPLE_SIM
    assert np.all(halos['is_merged_to'][ref['N']==0] != -1)
    # no deleted halos in ref should have merged particles in EXAMPLE_SIM
    assert np.all(halos['N_merge'][ref['N']==0] == 0)

    assert halos.meta == ref.meta

def test_subsamples_unclean(tmp_path):
    '''Test loading particle subsamples
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    
    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=dict(A=True), fields='all', cleaned=False)
    lenA = len(cat.subsamples)
    assert lenA == 2975
    assert cat.subsamples.colnames == ['pos', 'vel']
    
    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=dict(B=True), fields='all', cleaned=False)
    lenB = len(cat.subsamples)
    assert lenB == 7082

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=True, fields='all', cleaned=False)
    
    assert len(cat.subsamples) == lenA + lenB

    # to regenerate reference
    #ref = cat.subsamples
    #import asdf; asdf.compression.set_compression_options(typesize='auto')
    #ref.write(PARTICLES_OUTPUT_UNCLEAN, format='asdf', all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(PARTICLES_OUTPUT_UNCLEAN)

    ss = cat.subsamples
    for col in ref.colnames:
        assert check_close(ref[col], ss[col])

    assert cat.subsamples.meta == ref.meta

def test_subsamples_clean(tmp_path):
    '''Test loading particle subsamples
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=True, fields='all', cleaned=True)

    # to regenerate reference
    #ref = cat.subsamples
    #import asdf; asdf.compression.set_compression_options(typesize='auto')
    #ref.write(PARTICLES_OUTPUT_CLEAN, format='asdf', all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(PARTICLES_OUTPUT_CLEAN)

    ss = cat.subsamples
    for col in ref.colnames:
        assert check_close(ref[col], ss[col])

    # total number of particles in ref should be equal to the sum total of npout{AB} in EXAMPLE_SIM
    assert len(ref) == np.sum(cat.halos['npoutA']) + np.sum(cat.halos['npoutB'])

    assert cat.subsamples.meta == ref.meta

def test_field_subset_loading():
    '''Test loading a subset of halo catalog columns
    '''
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', fields=['N','x_com'])
    assert set(cat.halos.colnames) == set(['N','x_com'])


def test_one_halo_info():
    '''Test loading a single halo_info file
    '''
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000'/'halo_info'/'halo_info_000.asdf',
        subsamples=True)
    assert len(cat.halos) == 127
    assert len(cat.subsamples) == 3209 #9306


def test_halo_info_list():
    '''Test list of halo infos
    '''
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        [EXAMPLE_SIM/'halos'/'z0.000'/'halo_info'/'halo_info_000.asdf',
         EXAMPLE_SIM/'halos'/'z0.000'/'halo_info'/'halo_info_001.asdf'],
        subsamples=True)
    assert len(cat.halos) == 281
    assert len(cat.subsamples) == 6900 #19555

    # check fail on dups
    with pytest.raises(ValueError):
        cat = CompaSOHaloCatalog(
        [EXAMPLE_SIM/'halos'/'z0.000'/'halo_info'/'halo_info_000.asdf',
         EXAMPLE_SIM/'halos'/'z0.000'/'halo_info'/'halo_info_000.asdf'])


def test_unpack_bits():
    '''Test unpack_bits
    '''

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    from abacusnbody.data.bitpacked import PID_FIELDS

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=True, unpack_bits=True, fields='N')
    assert set(PID_FIELDS) <= set(cat.subsamples.colnames)  # check subset

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=True, unpack_bits='density', fields='N')
    assert 'density' in cat.subsamples.colnames
    assert 'lagr_pos' not in cat.subsamples.colnames  # too many?

    # bad bits field name
    with pytest.raises(ValueError):
        cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', subsamples=True, unpack_bits=['blah'], fields='N')


def test_filter_func():
    '''Test CHC filter_func
    '''
    
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(EXAMPLE_SIM/'halos'/'z0.000', fields=['N','x_L2com'],
                            filter_func = lambda c: c['N'] > 100,
                            subsamples=True)
    assert (cat.halos['N'] > 100).all()
    assert len(cat.halos) == 146
    assert len(cat.subsamples) == 7193


def test_pack9():
    '''Test reading a pack9 timeslice file
    '''
    from abacusnbody.data.read_abacus import read_asdf
    fn = EXAMPLE_SIM/'slices'/'z0.000'/'L0_pack9'/'slab000.L0.pack9.asdf'
    p = read_asdf(fn, load=('pos','vel'),
                dtype=np.float32)
    
    #p.write(PACK9_OUTPUT, format='asdf', all_array_compression='blsc')
    ref = Table.read(PACK9_OUTPUT)
    
    for k in ref.colnames:
        assert np.all(p[k] == ref[k])
    assert p.meta == ref.meta
    
    p = read_asdf(fn, dtype=np.float32)
    assert sorted(p.colnames) == ['pos','vel']

    # pid checks
    pidfn = EXAMPLE_SIM/'slices'/'z0.000'/'L0_pack9_pid'/'slab000.L0.pack9.pid.asdf'
    p = read_asdf(pidfn, load=('aux','pid','lagr_pos','tagged','density','lagr_idx'))
    
    #p.write(PACK9_PID_OUTPUT, format='asdf', all_array_compression='blsc')
    ref = Table.read(PACK9_PID_OUTPUT)
    
    for k in ref.colnames:
        assert np.all(p[k] == ref[k])
    assert p.meta == ref.meta
    
    p = read_asdf(pidfn, dtype=np.float32)
    assert p.colnames == ['pid']

    
def test_halo_lc():
    '''Test loading halo light cones
    '''
    
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(curdir / 'halo_light_cones/AbacusSummit_base_c000_ph001-abridged/z2.250/',
                             fields='all',
                             subsamples=True)
    assert(cat.halo_lc == True)
    
    HALO_LC_CAT = refdir / 'halo_lc_cat.asdf'
    HALO_LC_SUBSAMPLES = refdir / 'halo_lc_subsample.asdf'
    
    # generate reference
    #ref = cat.halos
    #ref.write(HALO_LC_CAT, format='asdf', all_array_storage='internal', all_array_compression='blsc')
    
    #ref = cat.subsamples
    #ref.write(HALO_LC_SUBSAMPLES, format='asdf', all_array_storage='internal', all_array_compression='blsc')
    
    ref = Table.read(HALO_LC_CAT)
    halos = cat.halos
    for col in ref.colnames:
        assert check_close(ref[col], halos[col])
    assert halos.meta == ref.meta
    
    ref = Table.read(HALO_LC_SUBSAMPLES)
    ss = cat.subsamples
    for col in ref.colnames:
        assert check_close(ref[col], ss[col])
            
    assert ss.meta == ref.meta
