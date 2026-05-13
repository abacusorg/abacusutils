"""
Unpack the Aurora ``HealStruct`` format (ASDF colname ``lightcone_healpix``).

The ``HealStruct`` format is part of the **Aurora** simulation data model. Each
on-disk element is a single ``uint64`` packed as::

    bits 63 .. 28 : pixel    (36 bits) — HEALPix NEST64 pixel index
    bits 27 .. 19 : dist_bin ( 9 bits) — radial bin in 0.2 Mpc/h quanta
                                          (wraps every 102.4 Mpc/h)
    bits 18 ..  0 : count    (19 bits) — particle count in this voxel

The top 45 bits (``pixel | dist_bin``) form a **voxel_id**: rows with equal
``voxel_id`` lie in the same 3D voxel and can be coadded by summing their
``count`` values.

This module also exposes the bit-layout constants (``PIXEL_BITS``,
``PIXEL_SHIFT``, ``PIXEL_MASK``, etc.) so callers who want to do their own bit
manipulation can do so without redefining magic numbers.

Most users will access this through
:func:`abacusnbody.data.read_abacus.read_asdf` rather than calling
:func:`unpack_healstruct` directly.
"""

import numpy as np

__all__ = [
    'unpack_healstruct',
    'LAYOUT',
    'PIXEL_BITS',
    'DIST_BITS',
    'COUNT_BITS',
    'VOXEL_ID_BITS',
    'PIXEL_SHIFT',
    'DIST_SHIFT',
    'COUNT_SHIFT',
    'VOXEL_ID_SHIFT',
    'PIXEL_MASK',
    'DIST_MASK',
    'COUNT_MASK',
    'VOXEL_ID_MASK',
]

# The HealStructLayout string stored in file headers under that key. This
# module only supports this exact layout; readers should validate before
# unpacking.
LAYOUT = 'pixel36_dist9_count19'

# Bit widths
PIXEL_BITS = 36
DIST_BITS = 9
COUNT_BITS = 19
VOXEL_ID_BITS = PIXEL_BITS + DIST_BITS  # 45 — top 45 bits, coaddition key

# Right-shifts: how far to shift `packed` to get the field at bit 0
COUNT_SHIFT = 0
DIST_SHIFT = COUNT_BITS  # 19
PIXEL_SHIFT = COUNT_BITS + DIST_BITS  # 28
VOXEL_ID_SHIFT = COUNT_BITS  # 19 (same as DIST_SHIFT)

# Masks: for use after right-shifting, e.g. `(packed >> DIST_SHIFT) & DIST_MASK`
COUNT_MASK = (1 << COUNT_BITS) - 1  # 0x7FFFF
DIST_MASK = (1 << DIST_BITS) - 1  # 0x1FF
PIXEL_MASK = (1 << PIXEL_BITS) - 1  # 0xFFFFFFFFF
VOXEL_ID_MASK = (1 << VOXEL_ID_BITS) - 1  # 0x1FFFFFFFFFFFF


_ALL_FIELDS = ('pixel', 'dist_bin', 'count', 'healstruct', 'voxel_id')


def _default_dtype(name):
    return {
        'pixel': np.uint64,
        'dist_bin': np.uint16,
        'count': np.uint32,
        'healstruct': np.uint64,
        'voxel_id': np.uint64,
    }[name]


def unpack_healstruct(data, out=None, fields=None):
    """
    Unpack an Aurora ``HealStruct`` array into per-field numpy arrays.

    Parameters
    ----------
    data : ndarray
        The raw packed data. Accepted shapes: ``(N,)`` ``uint64``, or ``(N, 8)``
        ``uint8`` (the form ASDF hands us). The function reinterprets to
        ``(N,)`` ``uint64`` before unpacking.

    out : dict[str, ndarray] or None, optional
        Pre-allocated output arrays to fill in place, keyed by field name.
        Field selection is determined by ``out.keys()``. The caller is
        responsible for providing arrays of shape ``(N,)``. Mutually exclusive
        with ``fields``.

    fields : tuple of str or None, optional
        Which fields to allocate and return. Used only when ``out`` is None.
        ``None`` (the default, when ``out`` is also None) means all five:
        ``'pixel', 'dist_bin', 'count', 'healstruct', 'voxel_id'``.

    Returns
    -------
    dict[str, ndarray]
        Either ``out`` itself (when given), or newly-allocated arrays. All
        arrays have shape ``(N,)``. Default dtypes:

        - ``pixel``: ``uint64`` (36 bits, needs uint64)
        - ``dist_bin``: ``uint16``
        - ``count``: ``uint32`` (19 bits, does not fit in uint16)
        - ``healstruct``: ``uint64`` (the raw packed value)
        - ``voxel_id``: ``uint64`` (45 bits, the coaddition key)
    """
    if out is not None and fields is not None:
        raise ValueError('pass either `out` or `fields`, not both')

    packed = np.ascontiguousarray(data).view(np.uint64).reshape(-1)
    N = packed.shape[0]

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
        expected_shape = (N,)
        if out is not None:
            arr = out[name]
            if arr.shape != expected_shape:
                raise ValueError(
                    f'out[{name!r}] has shape {arr.shape}, expected {expected_shape}'
                )
            result[name] = arr
        else:
            result[name] = np.empty(expected_shape, dtype=_default_dtype(name))

    if 'pixel' in requested_set:
        result['pixel'][...] = packed >> PIXEL_SHIFT
    if 'dist_bin' in requested_set:
        result['dist_bin'][...] = (packed >> DIST_SHIFT) & DIST_MASK
    if 'count' in requested_set:
        result['count'][...] = packed & COUNT_MASK
    if 'healstruct' in requested_set:
        result['healstruct'][...] = packed
    if 'voxel_id' in requested_set:
        result['voxel_id'][...] = packed >> VOXEL_ID_SHIFT

    return result
