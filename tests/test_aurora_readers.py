"""
Tests for the Aurora bit-packed readers: `output_particle`, `maplog`, and the
Nside=4 HEALPix direction table they share.

These build rows by hand rather than reading reference files. The formats are pure
bit layouts with no compression or header dependency, so a hand-built row is a
complete and independent statement of what the layout is -- and, unlike a reference
file, it says so in a form a reviewer can check against the C++ by eye.

The MAP relative-velocity direction is the newest field and the one most easily got
wrong: it shares `pid_k` with the MAP flag in `output_particle`, and shares a word
with the multiplicity in `maplog`. Both splits are covered at their boundaries.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from abacusnbody.data._healpix_nside4 import NPIX, NSIDE, rel_vel_healpix_lookup
from abacusnbody.data.maplog import (
    NODE_FORMATION,
    NODE_LIGHTCONE,
    NODE_MERGER,
    NODE_TIMESLICE,
    unpack_maplog,
)
from abacusnbody.data.output_particle import unpack_output_particle

# ---------------------------------------------------------------------------
# Row builders. These mirror output_particle::_pack and MapNode::pack_base.
# ---------------------------------------------------------------------------

POS_QUANTUM = 1.0 / 131072.0  # Mpc/h per stored unit


def _output_particle_row(
    pos_units=(0, 0, 0),
    vel_word=0,
    *,
    pid=None,
    hpix=None,
    mult=0,
    rel_vel_byte=0,
    density_byte=0,
    vel_disp_byte=0,
):
    """One 6-uint32 output_particle row.

    Pass `pid` for a DM row, or `hpix` for a MAP row (mutually exclusive).
    """
    if (pid is None) == (hpix is None):
        raise ValueError('pass exactly one of pid= (DM) or hpix= (MAP)')
    w = np.zeros(6, dtype=np.uint32)
    for i in range(3):
        w[i] = np.uint32(np.int32(pos_units[i]))
    w[3] = vel_word
    if hpix is not None:
        # MAP: pid_k = 0xff00 | hpix, and word 4 is mult<<8 | rel_vel
        pid_k = 0xFF00 | hpix
        w[4] = (np.uint32(mult) << 8) | np.uint32(rel_vel_byte)
    else:
        pid_k = pid[2]
        w[4] = np.uint32(pid[0]) | (np.uint32(pid[1]) << 16)
    w[5] = (
        np.uint32(pid_k)
        | (np.uint32(density_byte) << 16)
        | (np.uint32(vel_disp_byte) << 24)
    )
    return w


def _mapnode_row(
    node_type,
    *,
    timestep=0,
    mult=0,
    hpix=0,
    pos_units=(0, 0, 0),
    vel_word=0,
    density_byte=0,
    vel_disp_byte=0,
    length=0,
    rel_vel_byte=0,
    lc_label=0,
    pid=(0, 0, 0),
    mult_sec=0,
):
    """One 6-uint32 MapNode row for the given modality."""
    w = np.zeros(6, dtype=np.uint32)
    packed_pos = (
        np.uint64(pos_units[2])
        | (np.uint64(pos_units[1]) << np.uint64(21))
        | (np.uint64(pos_units[0]) << np.uint64(42))
    )
    w[0] = np.uint32(packed_pos & np.uint64(0xFFFFFFFF))
    w[1] = np.uint32(packed_pos >> np.uint64(32))
    w[2] = vel_word
    # Multiplicity in the low 24 bits; direction in the top byte, epoch nodes only.
    w[3] = np.uint32(mult) | (np.uint32(hpix) << 24)
    control = np.uint32(node_type) | (np.uint32(timestep) << 2)
    if node_type == NODE_FORMATION:
        flex1 = np.uint32(pid[0])
        flex2 = np.uint32(pid[1]) | (np.uint32(pid[2]) << 16)
    else:
        flex1 = np.uint32(density_byte) | (np.uint32(vel_disp_byte) << 8)
        if node_type == NODE_MERGER:
            flex2 = np.uint32(mult_sec)
        else:
            flex2 = (
                np.uint32(length)
                | (np.uint32(rel_vel_byte) << 16)
                | (np.uint32(lc_label) << 24)
            )
    w[4] = control | (flex1 << 16)
    w[5] = flex2
    return w


# ---------------------------------------------------------------------------
# The HEALPix direction table
# ---------------------------------------------------------------------------


def _vec2pix_nest(nside, vec):
    """Transcription of vec2pix_nest64 from abacus/src/include/healpix_shortened.c.

    The point of writing it out rather than calling healpy is that this is the
    function Abacus actually uses to *produce* the indices, so round-tripping the
    table through it is the invariant that matters. healpy would test a different
    (if equivalent) implementation, and is not a dependency of this package.
    """
    utab = [
        (m & 0x1)
        | ((m & 0x2) << 1)
        | ((m & 0x4) << 2)
        | ((m & 0x8) << 3)
        | ((m & 0x10) << 4)
        | ((m & 0x20) << 5)
        | ((m & 0x40) << 6)
        | ((m & 0x80) << 7)
        for m in range(256)
    ]

    def spread(x):
        r = 0
        for i in range(4):
            r |= utab[(x >> (8 * i)) & 0xFF] << (16 * i)
        return r

    def fmodulo(a, b):
        if a >= 0:
            return a if a < b else math.fmod(a, b)
        t = math.fmod(a, b) + b
        return 0.0 if t == b else t

    vlen = math.sqrt(sum(c * c for c in vec))
    z = vec[2] / vlen
    s = math.sqrt(vec[0] ** 2 + vec[1] ** 2) / vlen if abs(z) > 0.99 else -5.0
    phi = math.atan2(vec[1], vec[0])

    za = abs(z)
    tt = fmodulo(phi, 2 * math.pi) * (2.0 / math.pi)
    if za <= 2.0 / 3.0:
        temp1 = nside * (0.5 + tt)
        temp2 = nside * (z * 0.75)
        jp = int(temp1 - temp2)
        jm = int(temp1 + temp2)
        ifp, ifm = jp // nside, jm // nside
        face = (ifp | 4) if ifp == ifm else (ifp if ifp < ifm else ifm + 8)
        ix = jm & (nside - 1)
        iy = nside - (jp & (nside - 1)) - 1
    else:
        ntt = min(int(tt), 3)
        tp = tt - ntt
        tmp = (
            nside * s / math.sqrt((1.0 + za) / 3.0)
            if s > -2.0
            else nside * math.sqrt(3 * (1 - za))
        )
        jp = min(int(tp * tmp), nside - 1)
        jm = min(int((1.0 - tp) * tmp), nside - 1)
        if z >= 0:
            face, ix, iy = ntt, nside - jm - 1, nside - jp - 1
        else:
            face, ix, iy = ntt + 8, jp, jm
    return face * nside * nside + spread(ix) + (spread(iy) << 1)


def test_healpix_lookup_shape():
    assert (NSIDE, NPIX) == (4, 192)
    assert NPIX == 12 * NSIDE**2
    assert NPIX <= 256, 'the whole point is that a pixel index fits in a uint8'
    assert rel_vel_healpix_lookup.shape == (NPIX, 3)
    assert rel_vel_healpix_lookup.dtype == np.float32


def test_healpix_lookup_unit_vectors():
    # Readers scale these by the stored relative speed, so a non-unit row would
    # bias the reconstructed velocity. vec2pix normalizes internally and so would
    # not catch it.
    norms = np.linalg.norm(rel_vel_healpix_lookup.astype(np.float64), axis=1)
    npt.assert_allclose(norms, 1.0, atol=1e-6)


def test_healpix_lookup_inverts_vec2pix():
    """Every table row must map back to its own index under the function Abacus
    uses to build the index. If this fails, every MAP direction in the outputs is
    silently pointing somewhere else."""
    got = [_vec2pix_nest(NSIDE, rel_vel_healpix_lookup[i]) for i in range(NPIX)]
    npt.assert_array_equal(got, np.arange(NPIX))


# ---------------------------------------------------------------------------
# output_particle
# ---------------------------------------------------------------------------


def test_output_particle_dm_row():
    data = np.array(
        [_output_particle_row(pos_units=(131072, -262144, 0), pid=(7, 11, 13))]
    )
    c = unpack_output_particle(data)
    npt.assert_allclose(c['pos'][0], [1.0, -2.0, 0.0])
    npt.assert_array_equal(c['pid'][0], [7, 11, 13])
    assert not c['is_map'][0]
    # MAP-only fields are zero by convention for DM
    assert c['mult'][0] == 0
    assert c['rel_vel'][0] == 0
    assert c['rel_vel_healpix'][0] == 0


def test_output_particle_map_row():
    data = np.array([_output_particle_row(hpix=37, mult=1234, rel_vel_byte=0x40)])
    c = unpack_output_particle(data)
    assert c['is_map'][0]
    assert c['mult'][0] == 1234
    assert c['rel_vel_healpix'][0] == 37
    assert c['rel_vel'][0] > 0


@pytest.mark.parametrize('hpix', [0, 1, 100, 190, NPIX - 1])
def test_output_particle_healpix_roundtrip(hpix):
    """Both ends of the pixel range, including 0 -- which is also the DM sentinel
    value, so it has to be distinguishable via is_map rather than by being nonzero."""
    data = np.array([_output_particle_row(hpix=hpix, mult=1)])
    c = unpack_output_particle(data)
    assert c['is_map'][0]
    assert c['rel_vel_healpix'][0] == hpix


@pytest.mark.parametrize('pid_k', [0, 1, 0xFEFF, 0xFEFF - 1])
def test_output_particle_pid_k_below_map_band_is_dm(pid_k):
    """0xFF00 is the first MAP value; everything below it is a real Lagrangian
    index. 0xFEFF is the largest DM pid_k, and the most likely off-by-one."""
    data = np.array([_output_particle_row(pid=(1, 2, pid_k))])
    c = unpack_output_particle(data)
    assert not c['is_map'][0]
    assert c['pid'][0, 2] == pid_k


@pytest.mark.parametrize('pid_k', [0xFF00, 0xFF01, 0xFFBF, 0xFFFF])
def test_output_particle_pid_k_in_map_band_is_map(pid_k):
    """The entire 0xFF00-0xFFFF band flags a MAP, including the legacy 0xFFFF."""
    data = np.array([_output_particle_row(pid=(1, 2, pid_k))])
    c = unpack_output_particle(data)
    assert c['is_map'][0]
    assert c['rel_vel_healpix'][0] == (pid_k & 0xFF)


def test_output_particle_mixed_rows():
    """MAPs and DM are interleaved in real files; masks must be applied per-row."""
    data = np.array(
        [
            _output_particle_row(pid=(1, 2, 3)),
            _output_particle_row(hpix=191, mult=99, rel_vel_byte=0x30),
            _output_particle_row(pid=(4, 5, 0xFEFF)),
            _output_particle_row(hpix=0, mult=7, rel_vel_byte=0x20),
        ]
    )
    c = unpack_output_particle(data)
    npt.assert_array_equal(c['is_map'], [False, True, False, True])
    npt.assert_array_equal(c['mult'], [0, 99, 0, 7])
    npt.assert_array_equal(c['rel_vel_healpix'], [0, 191, 0, 0])
    npt.assert_array_equal(c['rel_vel'][[0, 2]], [0, 0])
    assert (c['rel_vel'][[1, 3]] > 0).all()


def test_output_particle_field_selection():
    """rel_vel_healpix needs pid_k and is_map as intermediates; make sure asking
    for it alone still computes them."""
    data = np.array([_output_particle_row(hpix=42, mult=5)])
    c = unpack_output_particle(data, fields=('rel_vel_healpix',))
    assert set(c) == {'rel_vel_healpix'}
    assert c['rel_vel_healpix'][0] == 42


def test_output_particle_rejects_unknown_field():
    data = np.array([_output_particle_row(pid=(0, 0, 0))])
    with pytest.raises(ValueError, match='unknown fields'):
        unpack_output_particle(data, fields=('vel_rel_healpix',))


def test_output_particle_direction_reconstruction():
    """The documented usage: magnitude times unit vector."""
    data = np.array([_output_particle_row(hpix=55, mult=3, rel_vel_byte=0x50)])
    c = unpack_output_particle(data)
    v3 = c['rel_vel'][:, None] * rel_vel_healpix_lookup[c['rel_vel_healpix']]
    npt.assert_allclose(np.linalg.norm(v3, axis=1), c['rel_vel'], rtol=1e-5)


# ---------------------------------------------------------------------------
# maplog
# ---------------------------------------------------------------------------


def test_maplog_timeslice_node():
    data = np.array(
        [
            _mapnode_row(
                NODE_TIMESLICE,
                timestep=17,
                mult=5000,
                hpix=123,
                length=4,
                rel_vel_byte=0x40,
                density_byte=0x50,
                vel_disp_byte=0x30,
            )
        ]
    )
    c = unpack_maplog(data)
    assert c['node_type'][0] == NODE_TIMESLICE
    assert c['timestep'][0] == 17
    assert c['mult'][0] == 5000
    assert c['rel_vel_healpix'][0] == 123
    assert c['length'][0] == 4
    assert c['rel_vel'][0] > 0
    assert c['density'][0] > 0
    assert c['vel_disp'][0] > 0


def test_maplog_lightcone_node():
    data = np.array(
        [_mapnode_row(NODE_LIGHTCONE, mult=12, hpix=7, lc_label=3, rel_vel_byte=0x40)]
    )
    c = unpack_maplog(data)
    assert c['node_type'][0] == NODE_LIGHTCONE
    assert c['mult'][0] == 12
    assert c['rel_vel_healpix'][0] == 7
    assert c['lc_label'][0] == 3


def test_maplog_mult_masks_off_direction_byte():
    """The multiplicity word is shared. A full-24-bit count next to a nonzero
    direction byte is the case that a stale (unmasked) reader gets wrong."""
    data = np.array([_mapnode_row(NODE_TIMESLICE, mult=0xFFFFFF, hpix=0xFF)])
    c = unpack_maplog(data)
    assert c['mult'][0] == 0xFFFFFF
    assert c['rel_vel_healpix'][0] == 0xFF


@pytest.mark.parametrize('node_type', [NODE_FORMATION, NODE_MERGER])
def test_maplog_non_epoch_nodes_have_no_direction(node_type):
    """Formation and Merger nodes carry no relative velocity, so they must report
    no direction even if the byte were somehow set."""
    data = np.array([_mapnode_row(node_type, mult=42, hpix=99)])
    c = unpack_maplog(data)
    assert c['node_type'][0] == node_type
    assert c['mult'][0] == 42
    assert c['rel_vel_healpix'][0] == 0
    assert c['rel_vel'][0] == 0


def test_maplog_formation_node_pid():
    data = np.array([_mapnode_row(NODE_FORMATION, mult=1, pid=(11, 22, 33))])
    c = unpack_maplog(data)
    npt.assert_array_equal(c['pid'][0], [11, 22, 33])


def test_maplog_merger_node_mult_sec():
    data = np.array([_mapnode_row(NODE_MERGER, mult=100, mult_sec=40)])
    c = unpack_maplog(data)
    assert c['mult'][0] == 100
    assert c['mult_sec'][0] == 40


def test_maplog_all_modalities_together():
    data = np.array(
        [
            _mapnode_row(NODE_TIMESLICE, mult=1000, hpix=5, rel_vel_byte=0x40),
            _mapnode_row(NODE_FORMATION, mult=1, pid=(1, 2, 3)),
            _mapnode_row(NODE_MERGER, mult=900, mult_sec=100),
            _mapnode_row(NODE_LIGHTCONE, mult=1000, hpix=190, lc_label=2),
        ]
    )
    c = unpack_maplog(data)
    npt.assert_array_equal(
        c['node_type'], [NODE_TIMESLICE, NODE_FORMATION, NODE_MERGER, NODE_LIGHTCONE]
    )
    npt.assert_array_equal(c['mult'], [1000, 1, 900, 1000])
    npt.assert_array_equal(c['rel_vel_healpix'], [5, 0, 0, 190])


def test_maplog_rejects_old_field_name():
    """`vel_rel` was renamed to `rel_vel` to match output_particle. Fail loudly
    rather than silently returning nothing for it."""
    data = np.array([_mapnode_row(NODE_TIMESLICE)])
    with pytest.raises(ValueError, match='unknown fields'):
        unpack_maplog(data, fields=('vel_rel',))


def test_maplog_healpix_indices_are_in_range():
    """Anything the reader reports must be a usable index into the lookup table."""
    data = np.array([_mapnode_row(NODE_TIMESLICE, mult=1, hpix=h) for h in range(NPIX)])
    c = unpack_maplog(data, fields=('rel_vel_healpix',))
    npt.assert_array_equal(c['rel_vel_healpix'], np.arange(NPIX))
    rel_vel_healpix_lookup[c['rel_vel_healpix']]  # must not raise
