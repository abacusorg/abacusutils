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
from asdf.extension import Extension, Compressor
    

class BloscCompressor(Compressor):
    '''
    An implementation of Blosc compression, as used by Abacus.
    '''
    
    def __init__(self, compression_kwargs=None,
                 decompression_kwargs=None,
                ):
        '''
        Construct a BloscCompressor object.
        
        Useful compression kwargs:
        nthreads
        compression_block_size
        blosc_block_size
        shuffle
        typesize
        cname
        clevel
        
        Useful decompression kwargs:
        nthreads
        '''
        
        if compression_kwargs is None:
            compression_kwargs = {}
        if decompression_kwargs is None:
            decompression_kwargs = {}
        
        # Compression fields
        compression_kwargs = compression_kwargs.copy()
        self.compression_kwargs = compression_kwargs
        self.nthreads_compress = compression_kwargs.pop('nthreads',1)
        self.compression_block_size = compression_kwargs.pop('compression_block_size',1<<22)
        self.blosc_block_size = compression_kwargs.pop('blosc_block_size', 512*1024)
        shuffle = compression_kwargs.pop('shuffle', 'shuffle')
        if shuffle == 'shuffle':
            self.shuffle = blosc.SHUFFLE
        elif shuffle == 'bitshuffle':
            self.shuffle = blosc.BITSHUFFLE
        elif shuffle == None:
            self.shuffle = blosc.NOSHUFFLE
        else:
            raise ValueError(shuffle)
        self.typesize = compression_kwargs.pop('typesize','auto')  # dtype size in bytes, e.g. 8 for int64
        self.clevel = compression_kwargs.pop('clevel',1)  # compression level, usually only need lowest for zstd
        self.cname = compression_kwargs.pop('cname','zstd')  # compressor name, default zstd, good performance/compression tradeoff
        
        # Decompression fields
        decompression_kwargs = decompression_kwargs.copy()
        self.decompression_kwargs = decompression_kwargs
        self.nthreads_decompress = decompression_kwargs.pop('nthreads',1)
        self._size = 0
        self._pos = 0
        self._buffer = None
        self._partial_len = b''
        self.decompression_time = 0.
        
    
    @property
    def labels(self):
        '''
        The string labels in the binary block headers
        that indicate Blosc compression
        '''
        return ['blsc']
    
    
    def compress(self, data, out=None):
        # Blosc code probably assumes contiguous buffer
        assert data.contiguous
        blosc.set_nthreads(self.nthreads_compress)
        blosc.set_blocksize(self.blosc_block_size)
        
        if self.typesize == 'auto':
            this_typesize = data.itemsize
        else:
            this_typesize = self.typesize
        #assert this_typesize != 1
        
        nelem = self.compression_block_size // data.itemsize
        for i in range(0,len(data),nelem):
            compressed = blosc.compress(data[i:i+nelem], typesize=this_typesize, clevel=self.clevel, shuffle=self.shuffle, cname=self.cname,
                                    **self.compression_kwargs)
            header = struct.pack('!I', len(compressed))
            # TODO: this probably triggers a data copy, feels inefficient. Probably have to add output array arg to blosc to fix
            yield header + compressed
    
    
    def decompress(self, data, out=None):
        if out is None:
            raise NotImplementedError
            
        bytesout = 0
        data = memoryview(data).cast('c').toreadonly()  # don't copy on slice
        
        # Blosc code probably assumes contiguous buffer
        if not data.contiguous:
            raise ValueError(data.contiguous)
        if not out.contiguous:
            raise ValuerError(out.contiguous)
        
        # get the out address
        out = np.frombuffer(out, dtype=np.uint8).ctypes.data
        
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
                    n_thisout = blosc.decompress_ptr(memoryview(self._buffer), out + bytesout)
                    self.decompression_time += time.perf_counter() - start
                    bytesout += n_thisout
                    self._buffer = None
                    self._size = 0
            else:
                # We have at least one full block
                start = time.perf_counter()
                n_thisout = blosc.decompress_ptr(memoryview(data[:self._size]), out + bytesout)
                self.decompression_time += time.perf_counter() - start
                bytesout += n_thisout
                data = data[self._size:]
                self._size = 0

        return bytesout
       

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
