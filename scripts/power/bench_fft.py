#!/usr/bin/env python3

import os
import timeit

import click
import numpy as np

import scipy.fft  # noqa: F401
# import mkl_fft._scipy_fft_backend as be
# Set mkl_fft to be used as backend of SciPy's FFT functions.
# scipy.fft.set_global_backend(be)

import pyfftw

NTHREAD = len(os.sched_getaffinity(0))

@click.command
@click.argument('n1d', default=256)
@click.argument('dtype', default='f4')
@click.option('-t', '--nthreads', 'nworker', default=NTHREAD)
@click.option('-b', '--backend', default='scipy')
def main(n1d: int = 256,
         dtype: str = 'f4',
         nworker: int = NTHREAD,
         backend: str = 'scipy',
         ):
    config = dict(n1d=n1d, dtype=dtype, nworker=nworker, backend=backend)
    rng = np.random.default_rng(300)

    if backend == 'scipy':
        # data = pyfftw.empty_aligned((n1d,n1d,n1d), dtype=dtype)
        # out = pyfftw.empty_aligned((n1d,n1d,n1d//2+1), dtype=np.complex64)
        data = np.empty((n1d,n1d,n1d), dtype=dtype)
        rng.random((n1d,n1d,n1d), dtype=dtype, out=data)

        cmd = 'scipy.fft.rfftn(data, overwrite_x=True, workers=nworker)'
    elif backend == 'pyfftw':
        data = pyfftw.empty_aligned((n1d,n1d,n1d), dtype=dtype)
        out = pyfftw.empty_aligned((n1d,n1d,n1d//2+1), dtype=np.complex64)

        rng.random((n1d,n1d,n1d), dtype=dtype, out=data)

        fftw_obj = pyfftw.FFTW(
            data,
            out,
            axes=(0,1,2),
            flags=('FFTW_MEASURE','FFTW_DESTROY_INPUT'),
            threads=nworker,
            )
        t = -timeit.default_timer()
        fftw_obj()
        t += timeit.default_timer()
        print(f'Plan time: {t*1e3:.2f} ms')
        cmd = 'fftw_obj()'
    elif backend == 'pyfftw-inplace':
        raw = pyfftw.empty_aligned((n1d,n1d,n1d+2), dtype=dtype)
        data = raw[:,:,:n1d]
        out = raw.view(dtype=np.complex64).reshape(n1d,n1d,n1d//2+1)

        data[:] = rng.random(data.shape, dtype=dtype)

        fftw_obj = pyfftw.FFTW(
            data,
            out,
            axes=(0,1,2),
            flags=('FFTW_MEASURE','FFTW_DESTROY_INPUT'),
            threads=nworker,
            )
        t = -timeit.default_timer()
        fftw_obj()
        t += timeit.default_timer()
        print(f'Plan time: {t*1e3:.2f} ms')
        cmd = 'fftw_obj()'

    number = int(12*nworker**0.5*(256/n1d))
    t = timeit.repeat(
        cmd,
        repeat=5,
        number=number,
        globals=globals() | locals(),
    )
    t = np.array(t) / number
    print(f'Time: {t.min() * 1e3:.2f} ms')
    print(f'Config: {config}')

if __name__ == '__main__':
    main()
