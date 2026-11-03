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

For epoch nodes (LIGHTCONE / TIMESLICE), ``rel_vel`` is the *magnitude* of the
MAP's velocity relative to the mean local dark matter and ``rel_vel_healpix`` is
its *direction*, as an Nside=4 NEST HEALPix pixel index (0--191). The direction
occupies the top byte of the multiplicity word, so ``mult`` is the low 24 bits of
that word rather than all 32. Multiply the unit vector for the pixel by
``rel_vel`` to recover the vector::

    from abacusnbody.data.maplog import rel_vel_healpix_lookup

    v3 = t['rel_vel'][:, None] * rel_vel_healpix_lookup[t['rel_vel_healpix']]

Both are zero for FORMATION and MERGER nodes. See
:mod:`abacusnbody.data.output_particle` for the same pair of fields on the
particle outputs.

The C++ source-of-truth is ``maplog.cc`` / ``vect32.cc`` in the Abacus
codebase. The reference Python unpacker is
``scripts/aurora/dje-maplogs.py``.

Most users will access this through
:func:`abacusnbody.data.read_abacus.read_asdf` rather than calling
:func:`unpack_maplog` directly.
"""

import numpy as np

from ._aurora_encodings import (
    _check_int_range,
    _unpack_ufloat8_35,
    _unpack_ufloat8_44,
    _unpack_vect32,
)
from ._healpix_nside4 import rel_vel_healpix_lookup

__all__ = [
    'unpack_maplog',
    'select_maplog',
    'timesteps_in_redshift_range',
    'NODE_TYPES',
    'LAYOUT',
    'NODE_FORMATION',
    'NODE_MERGER',
    'NODE_LIGHTCONE',
    'NODE_TIMESLICE',
    'NODE_CONTROL_MASK',
    'rel_vel_healpix_lookup',
]

# The MapLogLayout string stored in the file header under that key. Earlier files
# carry no such key at all; "b" is the first version, introduced when the
# multiplicity word's top byte was claimed for rel_vel_healpix. A reader that took
# the whole uint32 as the multiplicity would silently be wrong on epoch nodes.
LAYOUT = 'mapnode_24b'

# Node modality codes (low 2 bits of the `control` word). Mirrors the
# #defines in maplog.cc.
NODE_FORMATION = 0
NODE_MERGER = 1
NODE_LIGHTCONE = 2
NODE_TIMESLICE = 3
NODE_CONTROL_MASK = 0x3

# String names accepted by `select_maplog` / `read_asdf(select=...)`.
NODE_TYPES = {
    'formation': NODE_FORMATION,
    'merger': NODE_MERGER,
    'lightcone': NODE_LIGHTCONE,
    'timeslice': NODE_TIMESLICE,
}

# `timestep` is control >> 2, where control is the low 16 bits of word 4.
_TIMESTEP_MAX = (1 << 14) - 1

# Rows handled at a time when selecting and unpacking. Bounds the size of the
# per-field temporaries (e.g. the uint64 packed positions) independent of N.
_BLOCK_ROWS = 1 << 20

# Position encoding: 21 bits per axis, centered at 2^20, 1/128 kpc/h quantum.
_POS_OFFSET = 1 << 20  # 1048576
_POS_SCALE = 128.0  # quanta per Mpc/h
_POS_MASK = (1 << 21) - 1  # 0x1FFFFF

# data[:, 3] is the multiplicity word: an 18-bit count in the low 24 bits, with the
# epoch nodes' rel_vel_healpix pixel in the top byte. The split is that way round
# on purpose -- blosc SHUFFLE byte-transposes these 24-byte records before
# compressing, so keeping the count low leaves its byte 2 near-constant and confines
# the near-random direction to a byte lane of its own.
_MULT_MASK = 0x00FFFFFF
_VRELDIR_SHIFT = 24


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
    'rel_vel',
    'rel_vel_healpix',
    'lc_label',
)


def _field_shape(name, N):
    if name in ('pos', 'vel', 'pid'):
        return (N, 3)
    return (N,)


def _default_dtype(name, float_dtype):
    if name in ('pos', 'vel', 'density', 'vel_disp', 'rel_vel'):
        return float_dtype
    # Math-able fields are promoted to int32 for user safety.
    # Bitflag/enum/identifier fields (`control`, `node_type`, `lc_label`, `pid`,
    # `rel_vel_healpix`) keep their natural unsigned dtype.
    return {
        'mult': np.int32,
        'control': np.uint16,
        'node_type': np.uint8,
        'timestep': np.int32,
        'mult_sec': np.int32,
        'pid': np.uint16,
        'length': np.int32,
        'rel_vel_healpix': np.uint8,
        'lc_label': np.uint8,
    }[name]


def _unpack_block(data, result, requested_set):
    """Unpack one (n, 6) uint32 block into the arrays (views) in ``result``."""
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
        'rel_vel',
        'rel_vel_healpix',
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
        'rel_vel',
        'rel_vel_healpix',
        'lc_label',
    }:
        node_type = control & NODE_CONTROL_MASK
    if requested_set & {'pid', 'density', 'vel_disp'}:
        flex1 = (data[:, 4] >> 16) & 0xFFFF
    if requested_set & {'mult_sec', 'pid', 'length', 'rel_vel', 'lc_label'}:
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
        # Low 24 bits only; the top byte is rel_vel_healpix on epoch nodes. Masked
        # unconditionally so every modality decodes the same way.
        mult_vals = data[:, 3] & _MULT_MASK
        _check_int_range(mult_vals, result['mult'].dtype, field='mult')
        result['mult'][...] = mult_vals

    if 'control' in requested_set:
        result['control'][...] = control

    if 'node_type' in requested_set:
        result['node_type'][...] = node_type

    if 'timestep' in requested_set:
        result['timestep'][...] = control >> 2

    # Modality-dependent (zeroed where not applicable)
    if 'mult_sec' in requested_set:
        is_merger = node_type == NODE_MERGER
        mult_sec_vals = flex2 * is_merger
        _check_int_range(mult_sec_vals, result['mult_sec'].dtype, field='mult_sec')
        result['mult_sec'][...] = mult_sec_vals

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

    if 'rel_vel' in requested_set:
        is_epoch = (node_type == NODE_LIGHTCONE) | (node_type == NODE_TIMESLICE)
        rel_vel_byte = ((flex2 >> 16) & 0xFF).astype(np.uint8)
        result['rel_vel'][...] = _unpack_ufloat8_35(rel_vel_byte) * is_epoch

    if 'rel_vel_healpix' in requested_set:
        # Top byte of the multiplicity word, not flex2. Zero for FORMATION/MERGER.
        is_epoch = (node_type == NODE_LIGHTCONE) | (node_type == NODE_TIMESLICE)
        result['rel_vel_healpix'][...] = (data[:, 3] >> _VRELDIR_SHIFT) * is_epoch

    if 'lc_label' in requested_set:
        is_lightcone = node_type == NODE_LIGHTCONE
        result['lc_label'][...] = ((flex2 >> 24) & 0xFF) * is_lightcone


def unpack_maplog(data, out=None, fields=None, float_dtype=np.float32, index=None):
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
        ``None`` (the default, when ``out`` is also None) means all 14.

    float_dtype : np.dtype, optional
        Dtype used only for newly-allocated float fields.

    index : ndarray of int or None, optional
        Row indices (e.g. from :func:`select_maplog`) to unpack. Only those rows
        are unpacked and the outputs have ``len(index)`` rows. ``None`` (the
        default) unpacks every row. Work is done in fixed-size blocks, so
        temporary memory does not scale with the number of rows.

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
    if index is None:
        N = data.shape[0]
    else:
        index = np.asarray(index)
        N = len(index)

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

    for s0 in range(0, N, _BLOCK_ROWS):
        s1 = min(s0 + _BLOCK_ROWS, N)
        block = data[s0:s1] if index is None else data[index[s0:s1]]
        views = {name: arr[s0:s1] for name, arr in result.items()}
        _unpack_block(block, views, requested_set)

    return result


def select_maplog(data, timestep=None, node_type=None):
    """
    Find the rows of a packed ``MapNode`` array matching the given criteria.

    Only the ``control`` word is read, so this is far cheaper than unpacking.
    Criteria are combined with AND; ``None`` means no constraint.

    Parameters
    ----------
    data : ndarray
        Raw packed data, as for :func:`unpack_maplog`.
    timestep : int or iterable of int or None
        Keep rows whose ``timestep`` field is one of these.
    node_type : str or iterable of str or None
        Keep rows of these node types. Names (case-insensitive) are
        ``'formation'``, ``'merger'``, ``'lightcone'``, ``'timeslice'``.

    Returns
    -------
    ndarray of int64
        Indices of the matching rows, in file order. Pass as ``index=`` to
        :func:`unpack_maplog`.
    """
    data = np.ascontiguousarray(data).view(np.uint32).reshape(-1, 6)

    types = None
    if node_type is not None:
        names = (
            list(node_type)
            if np.iterable(node_type) and not isinstance(node_type, str)
            else [node_type]
        )
        types = []
        for t in names:
            if not isinstance(t, str):
                raise TypeError(
                    f'node_type must be given as strings {sorted(NODE_TYPES)}; got {t!r}'
                )
            try:
                types.append(NODE_TYPES[t.lower()])
            except KeyError:
                raise ValueError(
                    f'unknown node_type {t!r}; valid: {sorted(NODE_TYPES)}'
                ) from None
        types = np.unique(types).astype(np.uint32)

    steps = None
    if timestep is not None:
        vals = [timestep] if np.isscalar(timestep) else list(timestep)
        for t in vals:
            if isinstance(t, (bool, np.bool_)) or int(t) != t:
                raise TypeError(f'timestep must be integers; got {t!r}')
            if not 0 <= t <= _TIMESTEP_MAX:
                raise ValueError(f'timestep {t} outside [0, {_TIMESTEP_MAX}]')
        steps = np.unique(vals).astype(np.uint32)

    N = len(data)
    if types is None and steps is None:
        return np.arange(N)

    pieces = []
    for s0 in range(0, N, _BLOCK_ROWS):
        control = data[s0 : s0 + _BLOCK_ROWS, 4] & 0xFFFF
        mask = np.ones(len(control), dtype=bool)
        if types is not None:
            mask &= np.isin(control & NODE_CONTROL_MASK, types)
        if steps is not None:
            mask &= np.isin(control >> 2, steps)
        pieces.append(np.flatnonzero(mask) + s0)
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def timesteps_in_redshift_range(step_to_z, zmin, zmax):
    """
    Timesteps whose starting redshift lies in ``[zmin, zmax]`` (inclusive).

    ``step_to_z`` is the ``Step_to_Z`` array from the maplog header, indexed by
    timestep. An epoch node's ``timestep`` is the step in which it was written:
    a TIMESLICE node is output at that step's starting redshift, and a LIGHTCONE
    node crosses the lightcone somewhere between that step's start and the next
    step's start, so a lightcone node can sit slightly below its step's redshift.
    """
    if zmin > zmax:
        raise ValueError(f'redshift range is reversed: ({zmin}, {zmax})')
    z = np.asarray(step_to_z, dtype=float)
    return np.flatnonzero((z >= zmin) & (z <= zmax))
