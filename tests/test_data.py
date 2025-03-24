"""
This module tests `abacusnbody.data`: the basic interfaces to read halos and particles.
The reference files are stored in `tests/ref_data`.
"""

from pathlib import Path

import asdf
import numpy as np
import numpy.testing as npt
import pytest
import astropy.table
from astropy.table import Table
from common import assert_close

curdir = Path(__file__).parent
refdir = curdir / 'ref_data'
EXAMPLE_SIM = curdir / 'Mini_N64_L32'
HALOS_OUTPUT_UNCLEAN = refdir / 'test_halos_unclean.asdf'
PARTICLES_OUTPUT_UNCLEAN = refdir / 'test_subsamples_unclean.asdf'
HALOS_OUTPUT_CLEAN = refdir / 'test_halos_clean.asdf'
PARTICLES_OUTPUT_CLEAN = refdir / 'test_subsamples_clean.asdf'
PACK9_OUTPUT = refdir / 'test_pack9.asdf'
PACK9_PID_OUTPUT = refdir / 'test_pack9_pid.asdf'
UNPACK_BITS_OUTPUT = refdir / 'test_unpack_bits.asdf'
READ_ASDF_OUTPUT = refdir / 'test_read_asdf.asdf'


def test_halos_unclean():
    """Test loading a base (uncleaned) halo catalog"""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000', subsamples=True, fields='all', cleaned=False
    )

    # to regenerate reference
    # ref = cat.halos
    # ref.write(HALOS_OUTPUT_UNCLEAN, all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(HALOS_OUTPUT_UNCLEAN)

    halos = cat.halos
    for col in ref.colnames:
        assert_close(ref[col], halos[col])

    assert halos.meta == ref.meta


def test_halos_clean():
    """Test loading a cleaned halo catalog"""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000', subsamples=True, fields='all', cleaned=True
    )

    # to regenerate reference
    # ref = cat.halos
    # ref.write(HALOS_OUTPUT_CLEAN, all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(HALOS_OUTPUT_CLEAN)

    halos = cat.halos
    assert_close(ref, halos)

    # all haloindex values should point to this slab
    npt.assert_equal(
        (halos['haloindex'] / 1e12).astype(int), cat.header['FullStepNumber']
    )
    # ensure that all deleted halos in ref are marked as merged in EXAMPLE_SIM
    assert np.all(halos['is_merged_to'][ref['N'] == 0] != -1)
    # no deleted halos in ref should have merged particles in EXAMPLE_SIM
    npt.assert_equal(halos['N_merge'][ref['N'] == 0], 0)

    assert halos.meta == ref.meta


def test_subsamples_unclean():
    """Test loading particle subsamples"""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000',
        subsamples=dict(A=True),
        fields='all',
        cleaned=False,
    )
    lenA = len(cat.subsamples)
    assert lenA == 2536
    assert cat.subsamples.colnames == ['pos', 'vel']

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000',
        subsamples=dict(B=True),
        fields='all',
        cleaned=False,
    )
    lenB = len(cat.subsamples)
    assert lenB == 6128

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000', subsamples=True, fields='all', cleaned=False
    )

    assert len(cat.subsamples) == lenA + lenB

    # to regenerate reference
    # ref = cat.subsamples
    # ref.write(PARTICLES_OUTPUT_UNCLEAN, format='asdf', all_array_storage='internal', all_array_compression='blsc', compression_kwargs={'typesize': 'auto'})

    ref = Table.read(PARTICLES_OUTPUT_UNCLEAN)
    ref_halos = Table.read(HALOS_OUTPUT_UNCLEAN)

    ss = cat.subsamples
    for i in range(len(cat.halos)):
        for AB in 'AB':
            assert_close(
                ref[
                    ref_halos[f'npstart{AB}'][i] : ref_halos[f'npstart{AB}'][i]
                    + ref_halos[f'npout{AB}'][i]
                ],
                ss[
                    cat.halos[f'npstart{AB}'][i] : cat.halos[f'npstart{AB}'][i]
                    + cat.halos[f'npout{AB}'][i]
                ],
            )

    assert cat.subsamples.meta == ref.meta


def test_subsamples_clean():
    """Test loading particle subsamples"""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000', subsamples=True, fields='all', cleaned=True
    )

    # to regenerate reference
    # ref = cat.subsamples
    # import asdf; asdf.compression.set_compression_options(typesize='auto')
    # ref.write(PARTICLES_OUTPUT_CLEAN, format='asdf', all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(PARTICLES_OUTPUT_CLEAN)

    ss = cat.subsamples
    assert_close(ref, ss)

    # total number of particles in ref should be equal to the sum total of npout{AB} in EXAMPLE_SIM
    assert len(ref) == np.sum(cat.halos['npoutA']) + np.sum(cat.halos['npoutB'])

    assert cat.subsamples.meta == ref.meta


def test_field_subset_loading():
    """Test loading a subset of halo catalog columns"""
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(EXAMPLE_SIM / 'halos' / 'z0.000', fields=['N', 'x_com'])
    assert set(cat.halos.colnames) == set(['N', 'x_com'])


def test_one_halo_info():
    """Test loading a single halo_info file"""
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000' / 'halo_info' / 'halo_info_000.asdf',
        subsamples=True,
    )
    assert len(cat.halos) == 127
    assert len(cat.subsamples) == 3209  # 9306


def test_halo_info_list():
    """Test list of halo infos"""
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        [
            EXAMPLE_SIM / 'halos' / 'z0.000' / 'halo_info' / 'halo_info_000.asdf',
            EXAMPLE_SIM / 'halos' / 'z0.000' / 'halo_info' / 'halo_info_001.asdf',
        ],
        subsamples=True,
    )
    assert len(cat.halos) == 281
    assert len(cat.subsamples) == 6900  # 19555

    # check fail on dups
    with pytest.raises(ValueError):
        cat = CompaSOHaloCatalog(
            [
                EXAMPLE_SIM / 'halos' / 'z0.000' / 'halo_info' / 'halo_info_000.asdf',
                EXAMPLE_SIM / 'halos' / 'z0.000' / 'halo_info' / 'halo_info_000.asdf',
            ]
        )


def test_unpack_bits():
    """Test unpack_bits"""

    from abacusnbody.data.bitpacked import PID_FIELDS
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000', subsamples=True, unpack_bits=True, fields='N'
    )
    assert set(PID_FIELDS) <= set(cat.subsamples.colnames)  # check subset

    # to regenerate reference
    # ref = cat.subsamples
    # ref.write(UNPACK_BITS_OUTPUT, format='asdf', all_array_storage='internal', all_array_compression='blsc', compression_kwargs={'typesize': 'auto'})

    ref = Table.read(UNPACK_BITS_OUTPUT)

    assert_close(ref, cat.subsamples)

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000',
        subsamples=True,
        unpack_bits='density',
        fields='N',
    )
    assert 'density' in cat.subsamples.colnames
    assert 'lagr_pos' not in cat.subsamples.colnames  # too many?

    # bad bits field name
    with pytest.raises(ValueError):
        cat = CompaSOHaloCatalog(
            EXAMPLE_SIM / 'halos' / 'z0.000',
            subsamples=True,
            unpack_bits=['blah'],
            fields='N',
        )


def test_filter_func():
    """Test CHC filter_func"""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000',
        fields=['N', 'x_L2com'],
        filter_func=lambda c: c['N'] > 100,
        subsamples=True,
    )
    assert (cat.halos['N'] > 100).all()
    assert len(cat.halos) == 146
    assert len(cat.subsamples) == 7193


def test_pack9():
    """Test reading a pack9 timeslice file"""
    from abacusnbody.data.read_abacus import read_asdf

    fn = EXAMPLE_SIM / 'slices' / 'z0.000' / 'L0_pack9' / 'slab000.L0.pack9.asdf'
    p = read_asdf(fn, load=('pos', 'vel'), dtype=np.float32)

    # p.write(PACK9_OUTPUT, format='asdf', all_array_compression='blsc')
    ref = Table.read(PACK9_OUTPUT)

    for k in ref.colnames:
        npt.assert_equal(p[k], ref[k])
    assert p.meta == ref.meta

    p = read_asdf(fn, dtype=np.float32)
    assert sorted(p.colnames) == ['pos', 'vel']

    # pid checks
    pidfn = (
        EXAMPLE_SIM / 'slices' / 'z0.000' / 'L0_pack9_pid' / 'slab000.L0.pack9.pid.asdf'
    )
    p = read_asdf(
        pidfn, load=('aux', 'pid', 'lagr_pos', 'tagged', 'density', 'lagr_idx')
    )

    # p.write(PACK9_PID_OUTPUT, format='asdf', all_array_compression='blsc')
    ref = Table.read(PACK9_PID_OUTPUT)

    for k in ref.colnames:
        npt.assert_equal(p[k], ref[k])
    assert p.meta == ref.meta

    p = read_asdf(pidfn, dtype=np.float32)
    assert p.colnames == ['pid']


def test_read_asdf():
    """Test reading rvint and pid files with read_asdf"""
    from abacusnbody.data.read_abacus import read_asdf

    halo_zdir = EXAMPLE_SIM / 'halos' / 'z0.000'

    fn = halo_zdir / 'field_rv_A' / 'field_rv_A_000.asdf'
    rv = read_asdf(fn, load=('pos', 'vel'), dtype=np.float32)

    pidfn = halo_zdir / 'field_pid_A' / 'field_pid_A_000.asdf'
    pid = read_asdf(
        pidfn, load=('aux', 'pid', 'lagr_pos', 'tagged', 'density', 'lagr_idx')
    )

    # with asdf.AsdfFile({'rv_data': rv, 'pid_data': pid}) as af:
    #     af.write_to(READ_ASDF_OUTPUT, all_array_compression='blsc')

    rvref = Table.read(READ_ASDF_OUTPUT, data_key='rv_data')
    pidref = Table.read(READ_ASDF_OUTPUT, data_key='pid_data')

    for k in rvref.colnames:
        np.testing.assert_equal(rv[k], rvref[k])
    assert rv.meta == rvref.meta

    for k in pidref.colnames:
        np.testing.assert_equal(pid[k], pidref[k])
    assert pid.meta == pidref.meta

    p = read_asdf(fn, dtype=np.float32)
    assert sorted(p.colnames) == ['pos', 'vel']

    p = read_asdf(pidfn, dtype=np.float32)
    assert p.colnames == ['pid']


def test_halo_lc():
    """Test loading halo light cones"""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    cat = CompaSOHaloCatalog(
        curdir / 'halo_light_cones/AbacusSummit_base_c000_ph001-abridged/z2.250/',
        fields='all',
        subsamples=True,
    )
    assert cat.halo_lc is True

    HALO_LC_CAT = refdir / 'halo_lc_cat.asdf'
    HALO_LC_SUBSAMPLES = refdir / 'halo_lc_subsample.asdf'

    # generate reference
    # ref = cat.halos
    # ref.write(HALO_LC_CAT, format='asdf', all_array_storage='internal', all_array_compression='blsc')

    # ref = cat.subsamples
    # ref.write(HALO_LC_SUBSAMPLES, format='asdf', all_array_storage='internal', all_array_compression='blsc')

    ref = Table.read(HALO_LC_CAT)
    halos = cat.halos
    assert_close(ref, halos)
    assert halos.meta == ref.meta

    ref = Table.read(HALO_LC_SUBSAMPLES)
    ss = cat.subsamples
    assert_close(ref, ss)

    assert ss.meta == ref.meta


def test_passthrough():
    """Tests passthrough mode, where we load the raw halo info columns without unpacking"""

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    from abacusnbody.util import cumsum

    cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000',
        subsamples=True,
        fields='all',
        cleaned=True,
        passthrough=True,
    )

    def read_asdf(fn):
        with asdf.open(fn, lazy_load=False, memmap=False) as af:
            return Table(af['data'], meta=af['header'])

    raw_halo_info_fns = sorted(
        (EXAMPLE_SIM / 'halos' / 'z0.000' / 'halo_info').glob('*.asdf')
    )
    _tables = [read_asdf(fn) for fn in raw_halo_info_fns]
    raw_halo_info = astropy.table.vstack(_tables)
    raw_halo_info.meta = _tables[0].meta

    raw_cleaned_halo_info_fns = sorted(
        (
            EXAMPLE_SIM.parent
            / 'cleaning'
            / EXAMPLE_SIM.name
            / 'z0.000'
            / 'cleaned_halo_info'
        ).glob('*.asdf')
    )
    raw_cleaned_halo_info = astropy.table.vstack(
        [read_asdf(fn) for fn in raw_cleaned_halo_info_fns]
    )

    for AB in 'AB':
        raw_halo_info[f'npout{AB}'] = (
            raw_halo_info[f'npout{AB}'] + raw_cleaned_halo_info[f'npout{AB}_merge']
        )
        raw_halo_info[f'npout{AB}'][raw_cleaned_halo_info['N_total'] == 0] = 0

    cumsum(
        raw_halo_info['npoutA'],
        initial=True,
        final=False,
        out=raw_halo_info['npstartA'],
    )
    cumsum(
        raw_halo_info['npoutB'],
        initial=True,
        final=False,
        offset=raw_halo_info['npstartA'][-1],
        out=raw_halo_info['npstartB'],
    )

    # check halo info
    for name, col in raw_halo_info.columns.items():
        npt.assert_equal(cat.halos[name], col)

    # check header
    for k, v in raw_halo_info.meta.items():
        assert cat.halos.meta[k] == v

    # check for 'rvint'
    assert cat.subsamples.colnames == ['rvint', 'packedpid']
    assert cat.halos['npoutA'].sum() + cat.halos['npoutB'].sum() == len(cat.subsamples)

    # It's kind of hard to check the subsamples, since they're a mix of original and cleaned
    # and skip L0.
    # As a basic test, though, we can unpack the rvint and packedpid and check that it matches
    # a non-passthrough catalog.

    from abacusnbody.data.bitpacked import unpack_rvint, unpack_pids

    cat.subsamples['pos'], cat.subsamples['vel'] = unpack_rvint(
        cat.subsamples['rvint'], cat.header['BoxSize']
    )
    cat.subsamples.update(
        unpack_pids(cat.subsamples['packedpid'], pid=True), copy=False
    )

    regular_cat = CompaSOHaloCatalog(
        EXAMPLE_SIM / 'halos' / 'z0.000',
        subsamples=True,
        fields=[],
        cleaned=True,
        passthrough=False,
    )

    for k in ('pos', 'vel'):
        npt.assert_allclose(cat.subsamples[k], regular_cat.subsamples[k])

    npt.assert_equal(cat.subsamples['pid'], regular_cat.subsamples['pid'])

    # double-check that packedpid isn't just pid
    assert not np.all(cat.subsamples['packedpid'] == regular_cat.subsamples['pid'])


@pytest.mark.parametrize(
    'layout_dir',
    [
        '1/Mini_N64_L32/halos/z0.000',
        '2/subsuite/Mini_N64_L32/halos/z0.000',
        '3/Mini_N64_L32/halos/z0.000',
        '4/Mini_N64_L32/halos/z0.000',
    ],
    ids=['1', '2', '3', '4'],
)
def test_cleaning_layouts(layout_dir):
    full_groupdir = curdir / 'cleaning_layouts' / layout_dir

    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

    CompaSOHaloCatalog(
        full_groupdir,
        subsamples=True,
        fields='N',
        cleaned=True,
    )
