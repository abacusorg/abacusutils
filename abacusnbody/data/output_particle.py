"""
Unpack the Aurora ``output_particle`` format.

The ``output_particle`` format is part of the **Aurora** simulation data model
(distinct from AbacusSummit's ``rvint``/``pack9``/``pid`` formats handled by
:mod:`abacusnbody.data.bitpacked` and :mod:`abacusnbody.data.pack9`). The same
binary layout is stored under the ASDF data column name ``output_particle`` for
timeslice outputs and ``lightcone_particle`` for lightcone outputs.

Each row of the on-disk array is 6 ``uint32`` words (24 bytes) encoding **nine
distinct fields**: pos, vel, pid, density, vel_disp, is_map, mult, rel_vel,
rel_vel_healpix.
DM particles and MAPs (multi-particle aggregates) are interleaved; a row is a
MAP iff the upper byte of ``pid_k`` (the low 16 bits of ``data[:, 5]``) is
``0xFF``, i.e. ``pid_k`` lies in ``0xFF00``--``0xFFFF``. That whole band is out
of range for a real Lagrangian index. For MAP rows the ``mult``, ``rel_vel`` and
``rel_vel_healpix`` fields are meaningful and ``pid`` is undefined; for DM rows
those three are zero by convention.

``rel_vel`` is only the *magnitude* of the MAP's velocity relative to the mean
local dark matter. Its *direction* is ``rel_vel_healpix``, the index of an
Nside=4 NEST HEALPix pixel (0--191, hence one byte, stored in the low byte of
``pid_k``). It is deliberately coarse: it exists so that analyses can treat some
MAPs as moving with the DM halo rather than with the MAP itself, which is a
statistical correction, and a near-random direction would not compress anyway.
Multiply the unit vector for that pixel by ``rel_vel`` to recover the vector::

    from abacusnbody.data.output_particle import rel_vel_healpix_lookup

    v3 = t['rel_vel'][:, None] * rel_vel_healpix_lookup[t['rel_vel_healpix']]

The same 192x3 table is written into every Abacus ASDF header as the
``rel_vel_healpix_lookup`` ndarray, so files are self-describing; the module-level
copy here just saves opening one.

Most users will access this through :func:`abacusnbody.data.read_abacus.read_asdf`
rather than calling :func:`unpack_output_particle` directly.
"""

import numpy as np

from ._aurora_encodings import _unpack_ufloat8_35, _unpack_ufloat8_44, _unpack_vect32
from ._healpix_nside4 import rel_vel_healpix_lookup

__all__ = ['unpack_output_particle', 'LAYOUT', 'rel_vel_healpix_lookup']

# The LightconeParticleLayout / TimesliceSubsampleLayout string stored in file
# headers under those keys. This module only supports this exact layout;
# readers should validate before unpacking.
#
# The "b" revision moved the MAP flag from the single pid_k value 0xFFFF to the
# 0xFF00-0xFFFF band, freeing pid_k's low byte for rel_vel_healpix. An older
# reader would decode most MAPs as DM particles without any error, which is why
# the layout string had to change even though the record size did not.
LAYOUT = 'output_particle_24b'

# pid_k values with this upper byte flag a MAP; the low byte is rel_vel_healpix.
_MAP_MASK = np.uint32(0xFF00)

_ALL_FIELDS = (
    'pos',
    'vel',
    'pid',
    'density',
    'vel_disp',
    'is_map',
    'mult',
    'rel_vel',
    'rel_vel_healpix',
)


def _field_shape(name, N):
    if name in ('pos', 'vel', 'pid'):
        return (N, 3)
    return (N,)


def _default_dtype(name, float_dtype):
    if name in ('pos', 'vel', 'density', 'vel_disp', 'rel_vel'):
        return float_dtype
    if name == 'pid':
        return np.uint16
    if name == 'is_map':
        return np.bool_
    if name == 'mult':
        # int32, not uint32: this is a math-able count, use signed type for user safety
        return np.int32
    if name == 'rel_vel_healpix':
        # Deliberately left as a raw pixel index rather than expanded to 3 floats:
        # keeps the post-processing footprint at 1 byte/particle. Index into
        # rel_vel_healpix_lookup when you actually need the vector.
        return np.uint8
    raise KeyError(name)


def unpack_output_particle(data, out=None, fields=None, float_dtype=np.float32):
    """
    Unpack an Aurora ``output_particle`` array into per-field numpy arrays.

    Parameters
    ----------
    data : ndarray
        The raw packed data, shape ``(N, 6)`` ``uint32`` (or equivalently
        ``(N, 24)`` ``uint8`` — the function reinterprets as ``uint32``).

    out : dict[str, ndarray] or None, optional
        Pre-allocated output arrays to fill in place, keyed by field name.
        Field selection is determined by ``out.keys()``. The caller is
        responsible for providing arrays of the correct shape (see below).
        Mutually exclusive with ``fields``.

    fields : tuple of str or None, optional
        Which fields to allocate and return. Used only when ``out`` is None.
        ``None`` (the default, when ``out`` is also None) means all nine:
        ``'pos', 'vel', 'pid', 'density', 'vel_disp', 'is_map', 'mult',
        'rel_vel', 'rel_vel_healpix'``.

    float_dtype : np.dtype, optional
        Dtype used only for newly-allocated float fields. Ignored for the
        in-place case (the caller's arrays keep their own dtype).

    Returns
    -------
    dict[str, ndarray]
        Either ``out`` itself (when given), or newly-allocated arrays.
        Field shapes (with ``N = len(data)``):

        - ``pos``: ``(N, 3)`` float — Mpc/h
        - ``vel``: ``(N, 3)`` float — km/s (NaN on overflow)
        - ``pid``: ``(N, 3)`` ``uint16`` — Lagrangian coords (undefined for MAPs)
        - ``density``: ``(N,)`` float
        - ``vel_disp``: ``(N,)`` float
        - ``is_map``: ``(N,)`` ``bool`` — True for MAP rows, False for DM
        - ``mult``: ``(N,)`` ``int32`` — multiplicity; 0 for DM
        - ``rel_vel``: ``(N,)`` float — relative speed (magnitude only); 0 for DM
        - ``rel_vel_healpix``: ``(N,)`` ``uint8`` — Nside=4 NEST HEALPix pixel of
          the relative-velocity direction, 0--191; 0 for DM. Index
          :data:`rel_vel_healpix_lookup` with it to get a unit vector.
    """
    if out is not None and fields is not None:
        raise ValueError('pass either `out` or `fields`, not both')

    data = np.ascontiguousarray(data).view(np.uint32)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(
            f'expected [N,6] uint32 (or [N,24] uint8); got shape {data.shape} '
            'after reinterpretation as uint32'
        )
    N = data.shape[0]

    # Preserve caller's order when given; otherwise fall back to _ALL_FIELDS order.
    if out is not None:
        requested = list(out.keys())
    elif fields is not None:
        requested = list(fields)
    else:
        requested = list(_ALL_FIELDS)

    requested_set = set(requested)
    unknown = requested_set - set(_ALL_FIELDS)
    if unknown:
        raise ValueError(
            f'unknown fields {sorted(unknown)}; valid: {list(_ALL_FIELDS)}'
        )

    result = {}
    for name in requested:
        expected_shape = _field_shape(name, N)
        if out is not None:
            arr = out[name]
            if arr.shape != expected_shape:
                raise ValueError(
                    f'out[{name!r}] has shape {arr.shape}, expected {expected_shape}'
                )
            result[name] = arr
        else:
            result[name] = np.empty(
                expected_shape, dtype=_default_dtype(name, float_dtype)
            )

    # Shared intermediates, computed only if needed
    pid_k = None
    is_map = None
    if requested_set & {'pid', 'is_map', 'mult', 'rel_vel', 'rel_vel_healpix'}:
        pid_k = data[:, 5] & np.uint32(0xFFFF)
    if requested_set & {'is_map', 'mult', 'rel_vel', 'rel_vel_healpix'}:
        # The whole 0xFF00-0xFFFF band flags a MAP, not just 0xFFFF: the low byte
        # carries rel_vel_healpix.
        is_map = (pid_k & _MAP_MASK) == _MAP_MASK

    if 'pos' in requested:
        # data[:, 0:3] are int32 bit patterns; scale by 1/131072 to get Mpc/h
        packed = np.ascontiguousarray(data[:, 0:3]).view(np.int32)
        result['pos'][...] = packed.astype(np.float32) * np.float32(1.0 / 131072.0)

    if 'vel' in requested:
        result['vel'][...] = _unpack_vect32(data[:, 3])

    if 'pid' in requested:
        pid_arr = result['pid']
        pid_arr[:, 0] = data[:, 4] & 0xFFFF
        pid_arr[:, 1] = (data[:, 4] >> 16) & 0xFFFF
        pid_arr[:, 2] = pid_k

    if 'is_map' in requested:
        result['is_map'][...] = is_map

    if 'mult' in requested:
        mult_arr = result['mult']
        mult_arr[...] = 0
        mult_arr[is_map] = data[is_map, 4] >> 8

    if 'rel_vel' in requested:
        rel_vel_arr = result['rel_vel']
        rel_vel_arr[...] = 0
        rel_vel_byte = (data[is_map, 4] & 0xFF).astype(np.uint8)
        rel_vel_arr[is_map] = _unpack_ufloat8_35(rel_vel_byte)

    if 'rel_vel_healpix' in requested:
        # Low byte of pid_k. Zero for DM, where pid_k is a real Lagrangian index.
        healpix_arr = result['rel_vel_healpix']
        healpix_arr[...] = 0
        healpix_arr[is_map] = (pid_k[is_map] & 0xFF).astype(np.uint8)

    if 'density' in requested:
        density_byte = ((data[:, 5] >> 16) & 0xFF).astype(np.uint8)
        result['density'][...] = _unpack_ufloat8_44(density_byte)

    if 'vel_disp' in requested:
        vel_disp_byte = ((data[:, 5] >> 24) & 0xFF).astype(np.uint8)
        result['vel_disp'][...] = _unpack_ufloat8_35(vel_disp_byte)

    return result
