"""
Test the metadata module
"""

def test_meta():
    from abacusnbody.metadata import get_meta

    meta = get_meta('AbacusSummit_base_c000_ph000', redshift=0.1)

    assert meta['SimName'] == 'AbacusSummit_base_c000_ph000'
    assert meta['OmegaNow_m'] == 0.379887444945823
    assert meta['GrowthTable'][1.] == 47.30480505646196
