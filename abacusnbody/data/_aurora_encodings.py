"""
Internal: low-level decoders for Aurora bit-packed encodings.

These helpers decode small bit-packed scalar/vector types that recur across
Aurora formats:

- ``_unpack_vect32``: 32-bit vector encoding (sign + 5-bit common exponent +
  three 8-bit mantissas) used for velocity in ``output_particle`` MAPs and in
  ``MapNode``.
- ``_unpack_ufloat8_44``: 8-bit unsigned float with 4-bit exponent, 4-bit
  mantissa. Used for density-like fields.
- ``_unpack_ufloat8_35``: 8-bit unsigned float with 3-bit exponent, 5-bit
  mantissa. Used for velocity-dispersion-like fields.

This module is private. Callers should be other modules within
:mod:`abacusnbody.data` (currently :mod:`output_particle` and :mod:`maplog`).
"""

import numpy as np


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
    """ufloat8_44: 4-bit exponent, 4-bit mantissa.  Decoded value for stored
    byte V:

    - ``V <  16``: subnormal range; decoded value is ``V`` (so 0, 1, ..., 15).
    - ``V >= 16``: normal range; ``(16 + M) << (E - 1)`` with ``E = V >> 4``
      and ``M = V & 0xF``.  Max is ``(16 + 15) << 14 = 507904`` for ``V == 255``.

    Mirrors ``ufloat8_44::as_double()`` in ``abacus/src/include/ufloat8.cc``.
    Computed in integer arithmetic — no subnormal float is materialised — so
    the result is correct even when NumPy is running under an MXCSR with FTZ
    set (which can flush a subnormal float32 read to zero during the
    ``.astype(np.float64)`` cast used by the naive bit-cast implementation).
    """
    v = np.asarray(values, dtype=np.uint32)
    E = v >> 4
    M = v & 0xF
    # In the subnormal branch (v < 16) E is 0, so E - 1 would underflow as
    # uint; we cast to int32 and clamp.  The shifted value is masked out by
    # np.where anyway, so any extra computation there is harmless.
    shifted = (16 + M) << np.maximum(E.astype(np.int32) - 1, 0)
    return np.where(v < 16, v, shifted).astype(np.float32)


def _unpack_ufloat8_35(values):
    """ufloat8_35: 3-bit exponent, 5-bit mantissa.  Decoded value for stored
    byte V:

    - ``V <  32``: subnormal range; decoded value is ``V`` (so 0, 1, ..., 31).
    - ``V >= 32``: normal range; ``(32 + M) << (E - 1)`` with ``E = V >> 5``
      and ``M = V & 0x1F``.  Max is ``(32 + 31) << 6 = 4032`` for ``V == 255``.

    Mirrors ``ufloat8_35::as_double()`` in ``abacus/src/include/ufloat8.cc``.
    See :func:`_unpack_ufloat8_44` for why the implementation avoids any
    subnormal float intermediate.
    """
    v = np.asarray(values, dtype=np.uint32)
    E = v >> 5
    M = v & 0x1F
    shifted = (32 + M) << np.maximum(E.astype(np.int32) - 1, 0)
    return np.where(v < 32, v, shifted).astype(np.float32)
