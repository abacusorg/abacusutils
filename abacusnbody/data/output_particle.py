"""
Unpack the Aurora ``output_particle`` format.

The ``output_particle`` format is part of the **Aurora** simulation data model
(distinct from AbacusSummit's ``rvint``/``pack9``/``pid`` formats handled by
:mod:`abacusnbody.data.bitpacked` and :mod:`abacusnbody.data.pack9`). The same
binary layout is stored under the ASDF data column name ``output_particle`` for
timeslice outputs and ``lightcone_particle`` for lightcone outputs.

Each row of the on-disk array is 6 ``uint32`` words (24 bytes) encoding **eight
distinct fields**: pos, vel, pid, density, vel_disp, is_map, mult, rel_vel.
DM particles and MAPs (multi-particle aggregates) are interleaved; a row is a
MAP iff its low 16 bits of ``data[:, 5]`` equal ``0xFFFF``. For MAP rows the
``mult`` and ``rel_vel`` fields are meaningful and ``pid`` is undefined; for DM
rows ``mult`` and ``rel_vel`` are zero by convention.

Most users will access this through :func:`abacusnbody.data.read_abacus.read_asdf`
rather than calling :func:`unpack_output_particle` directly.
"""

import numpy as np

__all__ = ['unpack_output_particle']

_ALL_FIELDS = (
    'pos',
    'vel',
    'pid',
    'density',
    'vel_disp',
    'is_map',
    'mult',
    'rel_vel',
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
        return np.uint32
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
        ``None`` (the default, when ``out`` is also None) means all eight:
        ``'pos', 'vel', 'pid', 'density', 'vel_disp', 'is_map', 'mult', 'rel_vel'``.

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
        - ``mult``: ``(N,)`` ``uint32`` — multiplicity; 0 for DM
        - ``rel_vel``: ``(N,)`` float — relative velocity; 0 for DM
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
    if requested_set & {'pid', 'is_map', 'mult', 'rel_vel'}:
        pid_k = data[:, 5] & np.uint32(0xFFFF)
    if requested_set & {'is_map', 'mult', 'rel_vel'}:
        is_map = pid_k == 0xFFFF

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

    if 'density' in requested:
        density_byte = ((data[:, 5] >> 16) & 0xFF).astype(np.uint8)
        result['density'][...] = _unpack_ufloat8_44(density_byte)

    if 'vel_disp' in requested:
        vel_disp_byte = ((data[:, 5] >> 24) & 0xFF).astype(np.uint8)
        result['vel_disp'][...] = _unpack_ufloat8_35(vel_disp_byte)

    return result


def _unpack_vect32(packed):
    """
    Decode the vect32 velocity encoding: one uint32 → three signed floats.

    Byte layout (little-endian within the uint32):
      byte 0: sign bits (bit 7=vx, bit 6=vy, bit 5=vz) and 5-bit exponent (low 5 bits)
      bytes 1, 2, 3: 8-bit mantissas for vx, vy, vz

    Exponent == 31 indicates overflow → NaN.
    """
    packed = np.ascontiguousarray(packed)
    N = packed.shape[0]
    b = packed.view(np.uint8).reshape(N, 4)
    signs = b[:, 0:1] >> np.array([7, 6, 5], dtype=np.uint8)
    signs = (signs & 1).astype(np.float32) * np.float32(-2.0) + np.float32(1.0)
    exponent = b[:, 0] & 0x1F
    overflow = exponent == 31
    mantissas = b[:, 1:4].astype(np.float32)
    # cap exponent to avoid int32 overflow in 2**shift; overflowed rows are masked to NaN below
    shift = np.minimum(exponent, 23)[:, np.newaxis]
    vel = (signs * mantissas * (2.0**shift)).astype(np.float32)
    vel[overflow] = np.nan
    return vel


def _unpack_ufloat8_44(values):
    """ufloat8_44: 4-bit exponent, 4-bit mantissa. Encoded as the bottom 8 bits
    of a float32 with implicit scale 2^-130; decode by view-as-float32 then
    multiply by 2^130."""
    u = values.astype(np.uint32) << 15  # shift into float32 mantissa position
    f = u.view(np.float32)
    return (f.astype(np.float64) * np.float64(2.0**130)).astype(np.float32)


def _unpack_ufloat8_35(values):
    """ufloat8_35: 3-bit exponent, 5-bit mantissa. Implicit scale 2^-131."""
    u = values.astype(np.uint32) << 15
    f = u.view(np.float32)
    return (f.astype(np.float64) * np.float64(2.0**131)).astype(np.float32)
