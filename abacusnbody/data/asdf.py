'''
This module contains the ASDF extensions that allow the asdf Python package
to read Abacus ASDF files that use Blosc compression internally.

There are two classes here: an Extension subclass, and a Compressor subclass.
The Extension is registered with ASDF via a setuptools "entry point" in setup.py.
It contains the reference to the Compressor subclass that knows how to
handle Blosc compression.
'''

import time
import struct

import numpy as np
import blosc
from asdf.extension import Extension, Compressor, Decompressor

class BloscDecompressor(Decompressor):
    '''
    An implementation of Blosc decompression, as used by Abacus.
    '''
    
    def __init__(self, nthreads=1):
        '''
        Construct a BloscDecompressor object.
        '''
        
        self.nthreads = nthreads
        blosc.set_nthreads(nthreads)
        
        # Decompression fields
        self._size = 0
        self._pos = 0
        self._buffer = None
        self._partial_len = b''
        self.decompression_time = 0.
            
    def __del__(self):
        assert self._buffer is None
    
    @property
    def labels(self):
        '''
        The string labels in the binary block headers
        that indicate Blosc compression
        '''
        return ['blsc']
    
    def decompress_into(self, data, out):
        bytesout = 0
        data = memoryview(data).cast('c').toreadonly()  # don't copy on slice
        
        # Blosc code probably assumes contiguous buffer
        assert data.contiguous
        
        while len(data):
            if not self._size:
                # Don't know the (compressed) length of this block yet
                if len(self._partial_len) + len(data) < 4:
                    self._partial_len += data
                    break  # we've exhausted the data
                if self._partial_len:
                    # If we started to fill a len key, finish filling it
                    remaining = 4-len(self._partial_len)
                    if remaining:
                        self._partial_len += data[:remaining]
                        data = data[remaining:]
                    self._size = struct.unpack('!I', self._partial_len)[0]
                    self._partial_len = b''
                else:
                    # Otherwise just read the len key directly
                    self._size = struct.unpack('!I', data[:4])[0]
                    data = data[4:]

            if len(data) < self._size or self._buffer is not None:
                # If we have a partial block, or we're already filling a buffer, use the buffer
                if self._buffer is None:
                    self._buffer = np.empty(self._size, dtype=np.byte)  # use numpy instead of bytearray so we can avoid zero initialization
                    self._pos = 0
                newbytes = min(self._size - self._pos, len(data))  # don't fill past the buffer len!
                self._buffer[self._pos:self._pos+newbytes] = np.frombuffer(data[:newbytes], dtype=np.byte)
                self._pos += newbytes
                data = data[newbytes:]

                if self._pos == self._size:
                    start = time.perf_counter()
                    n_thisout = blosc.decompress_ptr(memoryview(self._buffer), out.ctypes.data + bytesout)
                    self.decompression_time += time.perf_counter() - start
                    bytesout += n_thisout
                    self._buffer = None
                    self._size = 0
            else:
                # We have at least one full block
                start = time.perf_counter()
                n_thisout = blosc.decompress_ptr(memoryview(data[:self._size]), out.ctypes.data + bytesout)
                self.decompression_time += time.perf_counter() - start
                bytesout += n_thisout
                data = data[self._size:]
                self._size = 0

        return bytesout
    

class BloscCompressor(Compressor):
    '''
    An implementation of Blosc compression, as used by Abacus.
    '''
    
    def __init__(self, nthreads=1, asdf_block_size=None,
                 shuffle='shuffle', typesize='auto',
                 cname='zstd', clevel=1, blosc_block_size=512*1024):
        '''
        Construct a BloscCompressor object.
        '''
        
        self.nthreads = nthreads
        blosc.set_nthreads(nthreads)
        
        # Compression fields
        self.blosc_block_size = blosc_block_size
        blosc.set_blocksize(blosc_block_size)
        # Only set this field if we want to override the default
        if asdf_block_size:
            self.asdf_block_size = asdf_block_size
        self.typesize = typesize  # dtype size in bytes, e.g. 8 for int64
        self.clevel = clevel  # compression level, usually only need lowest for zstd
        self.cname = cname  # compressor name, default zstd, good performance/compression tradeoff
        if shuffle == 'shuffle':
            self.shuffle = blosc.SHUFFLE
        elif shuffle == 'bitshuffle':
            self.shuffle = blosc.BITSHUFFLE
        elif shuffle == None:
            self.shuffle = blosc.NOSHUFFLE
        else:
            raise ValueError(shuffle)
    
    @property
    def labels(self):
        '''
        The string labels in the binary block headers
        that indicate Blosc compression
        '''
        return ['blsc']
    
    
    def compress(self, data):
        # Blosc code probably assumes contiguous buffer
        assert data.contiguous
        
        if data.nbytes > 2147483631:  # ~2 GB
            # This should never happen, because we compress in blocks that are 4 MiB
            raise ValueError("data blocks must be smaller than 2147483631 bytes due to internal blosc limitations")
        if self.typesize == 'auto':
            this_typesize = data.itemsize
        else:
            this_typesize = self.typesize
        #assert this_typesize != 1
        compressed = blosc.compress(data, typesize=this_typesize, clevel=self.clevel, shuffle=self.shuffle, cname=self.cname)
        header = struct.pack('!I', len(compressed))
        # TODO: this probably triggers a data copy, feels inefficient. Probably have to add output array arg to blosc to fix
        return header + compressed  # bytes type
       

class AbacusExtension(Extension):
    '''
    An ASDF Extension that deals with Abacus types and formats.
    Currently only implements Blosc compression.
    '''
    
    @property
    def extension_uri(self):
        """
        Get the URI of the extension to the ASDF Standard implemented
        by this class.  Note that this may not uniquely identify the
        class itself.

        Returns
        -------
        str
        """
        return "asdf://abacusnbody.org/extensions/abacus-0.0.1"
    
    @property
    def compressors(self):
        '''
        Return the Compressors implemented in this extension
        '''
        return [BloscCompressor]
    
    @property
    def decompressors(self):
        '''
        Return the Decompressors implemented in this extension
        '''
        return [BloscDecompressor]
    