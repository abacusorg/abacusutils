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

import asdf
def _monkey_patch(*args,**kwargs):
    raise Exception("Please use abacusnbody.data.asdf.set_nthreads(nthreads)")
    
asdf.compression.set_decompression_options = _monkey_patch

def set_nthreads(nthreads):
    blosc.set_nthreads(nthreads)
    

class BloscCompressor(Compressor):
    '''
    An implementation of Blosc compression, as used by Abacus.
    '''
        
    @property
    def label(self):
        '''
        The string labels in the binary block headers
        that indicate Blosc compression
        '''
        return b'blsc'
    
    
    def compress(self, data, **kwargs):
        '''Useful compression kwargs:
        nthreads
        compression_block_size
        blosc_block_size
        shuffle
        typesize
        cname
        clevel
        '''
        # Blosc code probably assumes contiguous buffer
        assert data.contiguous
        
        nthreads = kwargs.pop('nthreads', 1)
        compression_block_size = kwargs.pop('compression_block_size',1<<22)
        blosc_block_size = kwargs.pop('blosc_block_size', 512*1024)
        typesize = kwargs.pop('typesize','auto')  # dtype size in bytes, e.g. 8 for int64
        clevel = kwargs.pop('clevel',1)  # compression level, usually only need lowest for zstd
        cname = kwargs.pop('cname','zstd')  # compressor name, default zstd, good performance/compression tradeoff
        
        shuffle = kwargs.pop('shuffle', 'shuffle')
        if shuffle == 'shuffle':
            shuffle = blosc.SHUFFLE
        elif shuffle == 'bitshuffle':
            shuffle = blosc.BITSHUFFLE
        elif shuffle == None:
            shuffle = blosc.NOSHUFFLE
        else:
            raise ValueError(shuffle)
        
        blosc.set_nthreads(nthreads)
        blosc.set_blocksize(blosc_block_size)
        
        if typesize == 'auto':
            this_typesize = data.itemsize
        else:
            this_typesize = typesize
        #assert this_typesize != 1
        
        nelem = compression_block_size // data.itemsize
        for i in range(0,len(data),nelem):
            compressed = blosc.compress(data[i:i+nelem], typesize=this_typesize, clevel=clevel, shuffle=shuffle, cname=cname,
                                    **kwargs)
            header = struct.pack('!I', len(compressed))
            # TODO: this probably triggers a data copy, feels inefficient. Probably have to add output array arg to blosc to fix
            yield header + compressed
    
    
    def decompress(self, blocks, out, **kwargs):
        '''Useful decompression kwargs:
        nthreads
        '''
        # TODO: controlled globally for now
        #nthreads = kwargs.pop('nthreads',1)
        #blosc.set_nthreads(nthreads)
        
        _size = 0
        _pos = 0
        _buffer = None
        _partial_len = b''
        
        decompression_time = 0.
        bytesout = 0
        
        # Blosc code probably assumes contiguous buffer
        if not out.contiguous:
            raise ValueError(out.contiguous)
        
        # get the out address
        out = np.frombuffer(out, dtype=np.uint8).ctypes.data
        
        for block in blocks:
            block = memoryview(block).cast('c')
            try:
                block = block.toreadonly()  # python>=3.8 only
            except AttributeError:
                pass
            
            if not block.contiguous:
                raise ValueError(block.contiguous)
                
            while len(block):
                if not _size:
                    # Don't know the (compressed) length of this block yet
                    if len(_partial_len) + len(block) < 4:
                        _partial_len += block
                        break  # we've exhausted the data
                    if _partial_len:
                        # If we started to fill a len key, finish filling it
                        remaining = 4-len(_partial_len)
                        if remaining:
                            _partial_len += block[:remaining]
                            block = block[remaining:]
                        _size = struct.unpack('!I', _partial_len)[0]
                        _partial_len = b''
                    else:
                        # Otherwise just read the len key directly
                        _size = struct.unpack('!I', block[:4])[0]
                        block = block[4:]

                if len(block) < _size or _buffer is not None:
                    # If we have a partial block, or we're already filling a buffer, use the buffer
                    if _buffer is None:
                        _buffer = np.empty(_size, dtype=np.byte)  # use numpy instead of bytearray so we can avoid zero initialization
                        _pos = 0
                    newbytes = min(_size - _pos, len(block))  # don't fill past the buffer len!
                    _buffer[_pos:_pos+newbytes] = np.frombuffer(block[:newbytes], dtype=np.byte)
                    _pos += newbytes
                    block = block[newbytes:]

                    if _pos == _size:
                        start = time.perf_counter()
                        n_thisout = blosc.decompress_ptr(memoryview(_buffer), out + bytesout, **kwargs)
                        decompression_time += time.perf_counter() - start
                        bytesout += n_thisout
                        _buffer = None
                        _size = 0
                else:
                    # We have at least one full block
                    start = time.perf_counter()
                    n_thisout = blosc.decompress_ptr(memoryview(block[:_size]), out + bytesout, **kwargs)
                    decompression_time += time.perf_counter() - start
                    bytesout += n_thisout
                    block = block[_size:]
                    _size = 0

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
        return [BloscCompressor()]
