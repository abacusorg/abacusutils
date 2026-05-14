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
