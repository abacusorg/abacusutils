"""
Tests for the 'bsc2' (Blosc2) ASDF compressor in abacusnbody.data.asdf.

The ASDF block payload must be exactly one Blosc2 cframe: that's what the
Abacus C++ writer emits, and both are read back by the same decompressor.
"""

import io
import struct

import asdf
import blosc2
import numpy as np
import pytest

from abacusnbody.data.asdf import Blosc2Compressor

# flags, compression, allocated_size, used_size, data_size, checksum
BLOCK_HEADER = struct.Struct('>I4sQQQ16s')
BLOCK_MAGIC = b'\xd3BLK'


def iter_asdf_blocks(raw):
    """Yield a dict per binary block in the ASDF file bytes `raw`."""
    off = 0
    while (off := raw.find(BLOCK_MAGIC, off)) >= 0:
        header_size = struct.unpack('>H', raw[off + 4 : off + 6])[0]
        _flags, comp, alloc, used, data_size, checksum = BLOCK_HEADER.unpack(
            raw[off + 6 : off + 6 + BLOCK_HEADER.size]
        )
        start = off + 6 + header_size
        yield {
            'compression': comp,
            'allocated': alloc,
            'used': used,
            'data_size': data_size,
            'checksum': checksum,
            'payload': raw[start : start + used],
        }
        off = start + alloc


def write_bsc2(arrays, **kwargs):
    """Write {name: array} to an in-memory ASDF file with per-array bsc2
    compression, and return the raw bytes."""
    af = asdf.AsdfFile(dict(arrays))
    for arr in arrays.values():
        af.set_array_storage(arr, 'internal')
        af.set_array_compression(arr, 'bsc2', typesize=arr.dtype.itemsize, **kwargs)
    buf = io.BytesIO()
    af.write_to(buf)
    return buf.getvalue()


@pytest.mark.parametrize(
    'arr',
    [
        np.arange(10000, dtype=np.int64),
        np.linspace(0, 1, 10000, dtype=np.float64),
        np.linspace(0, 1, 10000, dtype=np.float32),
        np.arange(10000, dtype=np.int64) % 2 == 0,
        np.array([f'abc{i:03d}' for i in range(1000)], dtype='S7'),
        np.array([f'unicode{i}' for i in range(1000)], dtype='U8'),
        np.zeros(100000, dtype=np.float64),
        np.arange(3, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.arange(1000 * 10 * 7, dtype=np.float32).reshape(1000, 10, 7),
    ],
    ids=[
        'int64',
        'float64',
        'float32',
        'bool',
        'bytes',
        'unicode',
        'zeros',
        'tiny',
        'empty',
        '3d',
    ],
)
def test_roundtrip(arr):
    raw = write_bsc2({'a': arr})
    with asdf.open(io.BytesIO(raw), memmap=False, validate_checksums=True) as af:
        assert np.array_equal(af['a'], arr)
        assert af['a'].dtype == arr.dtype


@pytest.mark.parametrize('typesize', [6, 24, 7])
@pytest.mark.parametrize('chunksize', [1 << 20, (1 << 20) + 1])
def test_zeros_odd_typesize(typesize, chunksize):
    """All-zeros chunks are stored as run-length "special value" chunks, which
    blosc2 can't read back if the chunk isn't a whole number of records."""
    arr = np.zeros(3 * chunksize // typesize + 5, dtype=f'S{typesize}')
    raw = write_bsc2({'a': arr}, chunksize=chunksize)
    with asdf.open(io.BytesIO(raw), memmap=False) as af:
        assert np.array_equal(af['a'], arr)


def test_payload_is_one_cframe():
    arr = np.arange(100000, dtype=np.int64)
    (block,) = iter_asdf_blocks(write_bsc2({'a': arr}))

    assert block['compression'] == b'bsc2'
    assert block['data_size'] == arr.nbytes
    schunk = blosc2.schunk_from_cframe(block['payload'])
    assert schunk.nbytes == arr.nbytes
    # No length prefix, no trailing bytes: the payload is the cframe and
    # nothing else
    with pytest.raises(RuntimeError):
        blosc2.schunk_from_cframe(block['payload'][:-1])


def test_multiple_chunks():
    arr = np.arange(1 << 22, dtype=np.int64)  # 32 MiB
    (block,) = iter_asdf_blocks(write_bsc2({'a': arr}, chunksize=1 << 22))
    assert blosc2.schunk_from_cframe(block['payload']).nchunks == 8


@pytest.mark.parametrize('nsplit', [0, 1, 100])
def test_decompress_split_blocks(nsplit):
    """ASDF may hand the decompressor the payload in several pieces."""
    arr = np.arange(100000, dtype=np.int64)
    (block,) = iter_asdf_blocks(write_bsc2({'a': arr}))
    payload = block['payload']
    blocks = [payload] if not nsplit else [payload[:nsplit], payload[nsplit:]]

    out = np.empty(arr.nbytes, dtype=np.uint8)
    n = Blosc2Compressor().decompress(blocks, out.data)
    assert n == arr.nbytes
    assert np.array_equal(out.view(arr.dtype), arr)


def compress_to_cframe(data, **kwargs):
    (cframe,) = Blosc2Compressor().compress(memoryview(data).cast('B'), **kwargs)
    # copy=True: the schunk outlives `cframe` here
    return blosc2.schunk_from_cframe(cframe, copy=True)


@pytest.mark.parametrize(
    'filt', [blosc2.Filter.SHUFFLE, blosc2.Filter.BITSHUFFLE, blosc2.Filter.NOFILTER]
)
def test_cparams_kwargs(filt):
    """Compression kwargs reach blosc2.CParams."""
    data = np.arange(10000, dtype=np.int64)
    schunk = compress_to_cframe(data, typesize=8, filters=[filt], clevel=4)

    assert schunk.cparams.filters[0] is filt
    assert schunk.cparams.clevel == 4
    assert schunk.decompress_chunk(0) == data.tobytes()


def test_blocksize_edges():
    data = np.zeros(300000, dtype='S24')
    # Below one record, a block still holds a record...
    assert compress_to_cframe(data, typesize=24, blocksize=10).cparams.blocksize == 24
    # ...but zero keeps blosc2's automatic choice
    assert compress_to_cframe(data, typesize=24, blocksize=0).cparams.blocksize > 24


def test_big_typesize_degrades():
    """blosc2 can't store a typesize > 255, so we fall back to no shuffle."""
    data = np.arange(10000, dtype=np.int64)
    schunk = compress_to_cframe(data, typesize=256)
    assert schunk.cparams.typesize == 1
    assert schunk.decompress_chunk(0) == data.tobytes()


def test_indivisible_typesize_warns():
    """A partial record in the last chunk would make an unreadable cframe, so
    we fall back to no shuffle, but noisily: the caller asked for shuffle."""
    data = np.arange(1000, dtype=np.int64)  # 8000 bytes, not a multiple of 3
    with pytest.warns(UserWarning, match='does not divide'):
        schunk = compress_to_cframe(data, typesize=3)
    assert schunk.cparams.typesize == 1
    assert schunk.decompress_chunk(0) == data.tobytes()


@pytest.mark.parametrize(
    'kwargs,exc',
    [
        ({'typesize': 0}, ValueError),  # blosc2 would segfault
        ({'typesize': -1}, ValueError),
        ({'cname': 'zstd'}, TypeError),  # not a CParams field
        ({'shuffle': 'shuffle'}, TypeError),
    ],
)
def test_bad_kwargs(kwargs, exc):
    with pytest.raises(exc):
        compress_to_cframe(np.arange(1000, dtype=np.int64), **kwargs)


def test_all_array_compression_discards_kwargs():
    """ASDF's all_array_compression overrides per-array compression kwargs,
    so callers that need a real typesize must use set_array_compression()."""
    arr = np.arange(100000, dtype=np.int64)

    (block,) = iter_asdf_blocks(write_bsc2({'a': arr}))
    assert blosc2.schunk_from_cframe(block['payload']).cparams.typesize == 8

    af = asdf.AsdfFile({'a': arr})
    af.set_array_compression(arr, 'bsc2', typesize=arr.dtype.itemsize)
    buf = io.BytesIO()
    af.write_to(buf, all_array_compression='bsc2')

    (block,) = iter_asdf_blocks(buf.getvalue())
    assert blosc2.schunk_from_cframe(block['payload']).cparams.typesize == 1
