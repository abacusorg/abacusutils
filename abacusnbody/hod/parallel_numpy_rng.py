'''
File: parallel_numpy_rng.py
Author: Lehman Garrison (https://github.com/lgarrison)
Website: https://github.com/lgarrison/parallel-numpy-rng
License: Apache-2.0
'''

import os

import numba
from numba import njit
import numpy as np

__all__ = ['default_rng', 'MTGenerator']

def default_rng(seed):
    '''A convenience function to create a ``MTGenerator`` with the default
    Numpy bit generator'''
    return MTGenerator(np.random.PCG64(seed))


class MTGenerator:
    '''
    A Multi-Threaded random number generator. Generates the same random number stream
    independent of the number of threads. Implements a subset of the
    ``numpy.random.Generator`` API.
    
    Usage
    -----
    ```python
    p = np.random.PCG64(123)  # or PCG64DXSM
    mtg = MTGenerator(p)
    r1 = mtg.random(size=16, nthread=2, dtype=np.float32)
    r2 = mtg.standard_normal(size=16, nthread=2, dtype=np.float32)
    ```
    
    Details
    -------
    This generator lets multiple threads sample from the same logical RNG stream by
    utilizing the fast-forward feature of the underling RNG (probably PCG). This
    requires that generating each random float calls the RNG a constant number of
    times---not true for the default Numpy algorithms, which use rejection sampling!
    Therefore, the algorithms here may be slower than their Numpy counterparts, maybe
    by a lot. But with enough threads, MTGenerator might be faster.
    
    Not all numpy.random.Generator methods are implemented. Some kinds of random
    values, like bounded random ints, are hard to generate without rejection sampling.
    
    Even though we are relying on implementation details of the underlying RNG
    (specifically, how many calls to the RNG were made to generate each output value),
    a key aspect of this is that we don't have to guess if we were right: we can
    query the state of the RNG after generating values to see if it matched our
    guess.
    '''
    def __init__(self, bit_generator, nthread=-1):
        self.bitgen = bit_generator
        if nthread <= 0:
            nthread = len(os.sched_getaffinity(0))
        self.nthread = nthread
        
        # each is a dict, keyed on dtype
        self._next_float = _next_float[bit_generator.state['bit_generator']]['zero']
        self._next_float_nonzero = _next_float[bit_generator.state['bit_generator']]['nonzero']
        
        self._cached_normal = None
        
    def random(self, size=None, nthread=None, out=None, verify_rng=True, dtype=np.float64):
        if size == None:
            size = 1
        if nthread == None:
            nthread = self.nthread
        if nthread > size:
            nthread = size
        
        starts = np.linspace(0, size, num=nthread+1, endpoint=True, dtype=np.int64)
        vals_per_call = 2 if dtype == np.float32 else 1
        bitgens = []
        states = np.empty(nthread, dtype=int)
        for t in range(nthread):
            bitgens += [self._copy_bitgen()]
            self._advance_bitgen(bitgens[-1], starts[t], vals_per_call)
            states[t] = bitgens[-1].ctypes.state_address
        
        if out is None:
            out = np.empty(size, dtype=dtype)
        next_float = self._next_float[dtype]
        _random(states, starts, out, next_float)
            
        if verify_rng:
            # Did we advance each RNG by the right amount?
            for t in range(nthread):
                _b = self._copy_bitgen()
                self._advance_bitgen(_b, starts[t+1], vals_per_call)
                assert bitgens[t].state['state'] == _b.state['state']
            
        # finally, advance the base RNG
        self._advance_bitgen(self.bitgen, size, vals_per_call)
        
        return out
    
    @staticmethod
    def _advance_bitgen(bitgen, vals, vals_per_call):
        '''Advance the underlying generator, possibly "fractionally"
        if the generator produces, e.g., two 32-bit values per 64-bit call
        '''
        if bitgen.state['has_uint32']:
            assert vals_per_call == 2  # only supposed to happen with float32
            bitgen.ctypes.next_uint32(bitgen.ctypes.state)
            vals -= 1
        bitgen.advance(vals//vals_per_call)
        for _ in range(vals % vals_per_call):
            bitgen.ctypes.next_uint32(bitgen.ctypes.state)
    
    
    def standard_normal(self, size=None, nthread=None, out=None, verify_rng=True, dtype=np.float64):
        '''
        
        Parameters
        ----------
        verify_rng: bool
            Check the correctness; specifically, that each thread made the number of RNG
            calls expected. Disabling this may improve performance for latency-bound cases.
            Default: True.
        '''
        if size == None:
            size = 1
        if nthread == None:
            nthread = self.nthread
        if nthread > max(size//2,1):
            nthread = max(size//2,1)
        if out is None:
            out = np.empty(size, dtype=dtype)
        if size == 0:
            return out
        
        first = 0
        if self._cached_normal != None:
            # the base RNG will already be advanced just past the cached value
            out[0] = self._cached_normal
            self._cached_normal = None
            first = 1  # where to start writing non-cached values
        
        # amount to fast-forward each thread's state
        # force even registration per thread
        # the last thread will spill into a cache if it overflows the array
        ff = 2*np.linspace(0, (size+1-first)//2, num=nthread+1, endpoint=True, dtype=np.int64)
        # note that ff[-1] will be out of bounds because here we're just tracking how much each
        # rng gets advanced, not where it's writing
        
        vals_per_call = 2 if dtype == np.float32 else 1
        bitgens = []
        states = np.empty(nthread, dtype=int)
        for t in range(nthread):
            bitgens += [self._copy_bitgen()]
            bitgens[-1].advance(ff[t]//vals_per_call)
            states[t] = bitgens[-1].ctypes.state_address
        
        # now offset the out array
        _out = out[first:]
        next_float_nonzero = self._next_float_nonzero[dtype]
        _cached_normal = _boxmuller(states, ff, _out, next_float_nonzero)
        if not np.isnan(_cached_normal):
            self._cached_normal = _cached_normal
        del _out
            
        # Did we advance each RNG by the right amount?
        if verify_rng:
            for t in range(nthread):
                _b = self._copy_bitgen()
                _b.advance(ff[t+1]//vals_per_call)
                assert bitgens[t].state['state'] == _b.state['state'], t
            
        # finally, advance the base RNG
        self.bitgen.advance(ff[-1]//vals_per_call)
        
        return out


    @staticmethod
    def _advance_bitgen_boxmuller(bitgen, vals, vals_per_call):
        '''Advance the bitgen for Box-Muller normals
        '''
        bitgen.advance((vals//vals_per_call//2)*2)
        return (vals//vals_per_call) % 2
    
    
    def _copy_bitgen(self):
        '''Return a copy of the base bitgen in its current state'''
        new = self.bitgen.__class__()
        new.state = self.bitgen.state  # this is a deep copy
        return new
    
@njit(fastmath=True, parallel=True)
def _random(states, starts, out, next_double):
    nthread = len(states)
    numba.set_num_threads(nthread)

    for t in numba.prange(nthread):
        a = starts[t]
        b = starts[t+1]
        s = states[t]
        for i in range(a,b):
            out[i] = next_double(s)
            

@njit(fastmath=True,parallel=True)
def _boxmuller(states, starts, out, next_double):
    nthread = len(states)
    numba.set_num_threads(nthread)
    dtype = out.dtype.type

    cache = np.full(1, np.nan, dtype=dtype)
    for t in numba.prange(nthread):
        a = starts[t]
        b = min(starts[t+1],len(out))
        s = states[t]
        for i in range(a,b,2):
            u1 = next_double(s)
            u2 = next_double(s)
            amp = np.sqrt(dtype(-2)*np.log(u1))
            ang = dtype(2*np.pi)*u2
            z0 = amp*np.cos(ang)
            z1 = amp*np.sin(ang)

            out[i] = z0
            if i+1 < b:
                out[i+1] = z1
            elif t == nthread-1:
                cache[0] = z1

    return cache[0]


# TODO: there are now enough of these that they should live in their own module
# TODO: for very low latency cases, there may be benefit to inlining these
def _generate_int_to_float(bitgen):
    # initialize the numba functions to convert ints to floats
    _next_uint32_pcg64 = bitgen().ctypes.next_uint32
    _next_uint64_pcg64 = bitgen().ctypes.next_uint64

    @njit(fastmath=True)
    def _next_float_pcg64(state):
        '''Random float in the semi-open interval [0,1)'''
        return np.float32(np.int32(_next_uint32_pcg64(state) >> np.uint32(8)) * (np.float32(1.) / np.float32(16777216.)))

    @njit(fastmath=True)
    def _next_float_pcg64_nonzero(state):
        '''Random float in the semi-open interval (0,1]'''
        return np.float32((np.int32(_next_uint32_pcg64(state) >> np.uint32(8)) + np.int32(1))  * (np.float32(1.) / np.float32(16777216.)))

    @njit(fastmath=True)
    def _next_double_pcg64(state):
        '''Random double in the semi-open interval [0,1)'''
        return np.float64(np.int64(_next_uint64_pcg64(state) >> np.uint64(11)) * (np.float64(1.) / np.float64(9007199254740992.)))

    @njit(fastmath=True)
    def _next_double_pcg64_nonzero(state):
        '''Random double in the semi-open interval (0,1]'''
        return np.float64((np.int64(_next_uint64_pcg64(state) >> np.uint64(11)) + np.int64(1))  * (np.float64(1.) / np.float64(9007199254740992.)))
    
    _next_float = {'zero': {np.float32:_next_float_pcg64, np.float64:_next_double_pcg64},
                   'nonzero': {np.float32:_next_float_pcg64_nonzero, np.float64:_next_double_pcg64_nonzero},
                  }
    return _next_float

_next_float = {}
_next_float['PCG64'] = _generate_int_to_float(np.random.PCG64)

if hasattr(np.random, 'PCG64DXSM'):
    # Numpy >= 1.21
    _next_float['PCG64XDSM'] = _generate_int_to_float(np.random.PCG64DXSM)
