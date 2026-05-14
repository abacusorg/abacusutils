"""
Unpack the Aurora ``MapNode`` format (ASDF colname ``maplogs``).

The ``maplogs`` format is part of the **Aurora** simulation data model. Each
on-disk element is a 24-byte ``MapNode``: one entry in a per-particle MAP
binary tree log. A MapNode has four modalities (FORMATION / MERGER /
LIGHTCONE / TIMESLICE) distinguished by the low 2 bits of a ``control``
word; several fields are overlapping C unions whose meaning depends on the
modality. This module exposes the full union as separate columns; fields
that aren't meaningful for a row's modality are zeroed (matching the
reference unpacker's pattern for ``mult_sec`` and ``pid``) so columns are
safe to read without filtering. Use ``node_type`` as the discriminator.

Position is returned in Mpc/h (5 kpc/h quantum, fixed); velocity in km/s
(via the vect32 encoding shared with ``output_particle``).

The C++ source-of-truth is ``maplog.cc`` / ``vect32.cc`` in the Abacus
codebase. The reference Python unpacker is
``scripts/aurora/dje-maplogs.py``.

Most users will access this through
:func:`abacusnbody.data.read_abacus.read_asdf` rather than calling
:func:`unpack_maplog` directly.
"""

import numpy as np

from ._aurora_encodings import _unpack_ufloat8_35, _unpack_ufloat8_44, _unpack_vect32

__all__ = [
    'unpack_maplog',
    'NODE_FORMATION',
    'NODE_MERGER',
    'NODE_LIGHTCONE',
    'NODE_TIMESLICE',
    'NODE_CONTROL_MASK',
]

# Node modality codes (low 2 bits of the `control` word). Mirrors the
# #defines in maplog.cc.
NODE_FORMATION = 0
NODE_MERGER = 1
NODE_LIGHTCONE = 2
NODE_TIMESLICE = 3
NODE_CONTROL_MASK = 0x3

# Position encoding: 21 bits per axis, centered at 2^20, 5 kpc/h quantum.
_POS_OFFSET = 1 << 20  # 1048576
_POS_SCALE = 200.0  # quanta per Mpc/h (1 / 0.005)
_POS_MASK = (1 << 21) - 1  # 0x1FFFFF


_ALL_FIELDS = (
    'pos',
    'vel',
    'mult',
    'control',
    'node_type',
    'timestep',
    'mult_sec',
    'pid',
    'density',
    'vel_disp',
    'length',
    'vel_rel',
    'lc_label',
)


def _field_shape(name, N):
    if name in ('pos', 'vel', 'pid'):
        return (N, 3)
    return (N,)


def _default_dtype(name, float_dtype):
    if name in ('pos', 'vel', 'density', 'vel_disp', 'vel_rel'):
        return float_dtype
    return {
        'mult': np.uint32,
        'control': np.uint16,
        'node_type': np.uint8,
        'timestep': np.uint16,
        'mult_sec': np.uint32,
        'pid': np.uint16,
        'length': np.uint16,
        'lc_label': np.uint8,
    }[name]


def unpack_maplog(data, out=None, fields=None, float_dtype=np.float32):
    """
    Unpack an Aurora ``MapNode`` array into per-field numpy arrays.

    Parameters
    ----------
    data : ndarray
        The raw packed data. Accepted shapes after reinterpretation as
        ``uint32``: ``(N, 6)`` (the canonical form), or any contiguous
        buffer of ``N * 24`` bytes that views cleanly as ``(N, 6) uint32``.

    out : dict[str, ndarray] or None, optional
        Pre-allocated output arrays to fill in place, keyed by field name.
        Field selection is determined by ``out.keys()``. Mutually exclusive
        with ``fields``.

    fields : tuple of str or None, optional
        Which fields to allocate and return. Used only when ``out`` is None.
        ``None`` (the default, when ``out`` is also None) means all 13.

    float_dtype : np.dtype, optional
        Dtype used only for newly-allocated float fields.

    Returns
    -------
    dict[str, ndarray]
        Mapping from requested field name to ndarray. See module docstring
        for the full field list and modality semantics. Fields that are
        not meaningful for a row's modality are zeroed.
    """
    if out is not None and fields is not None:
        raise ValueError('pass either `out` or `fields`, not both')

    data = np.ascontiguousarray(data).view(np.uint32)
    if data.ndim == 1:
        if data.size % 6 != 0:
            raise ValueError(
                f'flat input size {data.size} is not a multiple of 6 uint32 '
                '(MapNode = 24 bytes = 6 uint32)'
            )
        data = data.reshape(-1, 6)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(
            f'expected (N, 6) uint32 after reinterpretation; got shape {data.shape}'
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

    # Lazy shared intermediates
    control = None
    node_type = None
    flex1 = None
    flex2 = None
    if requested_set & {
        'control',
        'node_type',
        'timestep',
        'mult_sec',
        'pid',
        'density',
        'vel_disp',
        'length',
        'vel_rel',
        'lc_label',
    }:
        control = data[:, 4] & 0xFFFF
    if requested_set & {
        'node_type',
        'mult_sec',
        'pid',
        'density',
        'vel_disp',
        'length',
        'vel_rel',
        'lc_label',
    }:
        node_type = control & NODE_CONTROL_MASK
    if requested_set & {'pid', 'density', 'vel_disp'}:
        flex1 = (data[:, 4] >> 16) & 0xFFFF
    if requested_set & {'mult_sec', 'pid', 'length', 'vel_rel', 'lc_label'}:
        flex2 = data[:, 5]

    # Always-meaningful fields
    if 'pos' in requested_set:
        packed_pos = data[:, 0].astype(np.uint64) | (data[:, 1].astype(np.uint64) << 32)
        z_raw = (packed_pos & _POS_MASK).astype(np.int64)
        y_raw = ((packed_pos >> 21) & _POS_MASK).astype(np.int64)
        x_raw = ((packed_pos >> 42) & _POS_MASK).astype(np.int64)
        result['pos'][:, 0] = (x_raw - _POS_OFFSET) / _POS_SCALE
        result['pos'][:, 1] = (y_raw - _POS_OFFSET) / _POS_SCALE
        result['pos'][:, 2] = (z_raw - _POS_OFFSET) / _POS_SCALE

    if 'vel' in requested_set:
        result['vel'][...] = _unpack_vect32(data[:, 2])

    if 'mult' in requested_set:
        result['mult'][...] = data[:, 3]

    if 'control' in requested_set:
        result['control'][...] = control

    if 'node_type' in requested_set:
        result['node_type'][...] = node_type

    if 'timestep' in requested_set:
        result['timestep'][...] = control >> 2

    # Modality-dependent (zeroed where not applicable)
    if 'mult_sec' in requested_set:
        is_merger = node_type == NODE_MERGER
        result['mult_sec'][...] = flex2 * is_merger

    if 'pid' in requested_set:
        is_formation = node_type == NODE_FORMATION
        result['pid'][:, 0] = (flex1 & 0xFFFF) * is_formation
        result['pid'][:, 1] = (flex2 & 0xFFFF) * is_formation
        result['pid'][:, 2] = ((flex2 >> 16) & 0xFFFF) * is_formation

    if 'density' in requested_set:
        is_non_formation = node_type != NODE_FORMATION
        density_byte = (flex1 & 0xFF).astype(np.uint8)
        result['density'][...] = _unpack_ufloat8_44(density_byte) * is_non_formation

    if 'vel_disp' in requested_set:
        is_non_formation = node_type != NODE_FORMATION
        vel_disp_byte = ((flex1 >> 8) & 0xFF).astype(np.uint8)
        result['vel_disp'][...] = _unpack_ufloat8_35(vel_disp_byte) * is_non_formation

    if 'length' in requested_set:
        is_epoch = (node_type == NODE_LIGHTCONE) | (node_type == NODE_TIMESLICE)
        result['length'][...] = (flex2 & 0xFFFF) * is_epoch

    if 'vel_rel' in requested_set:
        is_epoch = (node_type == NODE_LIGHTCONE) | (node_type == NODE_TIMESLICE)
        vel_rel_byte = ((flex2 >> 16) & 0xFF).astype(np.uint8)
        result['vel_rel'][...] = _unpack_ufloat8_35(vel_rel_byte) * is_epoch

    if 'lc_label' in requested_set:
        is_lightcone = node_type == NODE_LIGHTCONE
        result['lc_label'][...] = ((flex2 >> 24) & 0xFF) * is_lightcone

    return result
