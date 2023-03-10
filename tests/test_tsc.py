'''
Tests of `abacusnbody.analysis.tsc`.
'''

from pathlib import Path

import asdf
import numpy as np
import pytest

testdir = Path(__file__).parent
refdir = testdir / 'ref_tsc'

@pytest.mark.parametrize('ngrid', [10,256])
@pytest.mark.parametrize('dtype', ['f4','f8'])
@pytest.mark.parametrize('nthread', [1,-1], ids=['serial','parallel'])
@pytest.mark.filterwarnings("ignore:.*dtype")
@pytest.mark.filterwarnings("ignore:.*npartition")
class TestTSC:
    box = 123.
    coord = 0

    def test_single(self, ngrid, dtype, nthread):
        from abacusnbody.analysis.tsc import tsc_parallel

        box = self.box
        coord = self.coord

        # single particle test
        cen = np.array([5,6,7])
        single = (cen/ngrid*box).astype(dtype).reshape(1,-1)
        dens = tsc_parallel(single, ngrid, box, nthread=nthread, coord=coord)
        assert (dens == 0).sum() == ngrid**3 - 27
        assert np.isclose(dens.sum(), 1.)

        cube = dens[cen[0]-1:cen[0]+2, cen[1]-1:cen[1]+2, cen[2]-1:cen[2]+2]

        # corners
        assert np.allclose([
            cube[0,0,0],
            cube[0,0,2],
            cube[0,2,0],
            cube[0,2,2],
            cube[2,0,0],
            cube[2,0,2],
            cube[2,2,0],
            cube[2,2,2],
            ], 0.5**9,
            )

        # edges
        assert np.allclose([
            cube[0,0,1],
            cube[0,1,0],
            cube[1,0,0],
            cube[0,2,1],
            cube[0,1,2],
            cube[1,0,2],
            cube[2,0,1],
            cube[2,1,0],
            cube[1,2,0],
            cube[2,2,1],
            cube[2,1,2],
            cube[1,2,2],
            ], 0.5**6 * 0.75,
            )

        # faces
        assert np.allclose([
            cube[1,1,0],
            cube[1,0,1],
            cube[0,1,1],
            cube[1,1,2],
            cube[1,2,1],
            cube[2,1,1],
            ], 0.5**3 * 0.75**2,
            )

        # center
        assert np.allclose(cube[1,1,1], 0.75**3)


    def test_multi(self, ngrid, dtype, nthread, save_result=False,
        save_nbodykit=False,
        ):
        from abacusnbody.analysis.tsc import _tsc_scatter, tsc_parallel

        # multi-particle tests
        box = self.box
        coord = self.coord
        N = 10000
        seed = 234
        rng = np.random.default_rng(seed)
        pos = rng.random((N,3), dtype='f4').astype(dtype)*box
        weights = rng.random((N,), dtype='f4').astype(dtype)

        dens = tsc_parallel(pos, ngrid, box, nthread=nthread, coord=coord,
            weights=weights,
            )

        assert np.isclose(dens.sum(dtype='f8'), weights.sum(dtype='f8'))

        # compare with the serial, pure-Python version
        pydens = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
        _tsc_scatter.py_func(pos, pydens, box, weights)

        assert np.allclose(dens, pydens)

        # compare with a saved result
        ref_fn = refdir / f'tsc_ngrid{ngrid}.asdf'
        if save_result and nthread == 1 and dtype == 'f8':
            with asdf.AsdfFile(dict(pydens=pydens)) as af:
                af.write_to(ref_fn, all_array_compression='blsc')

        with asdf.open(ref_fn) as af:
            savedens = af['pydens']
            assert np.allclose(dens, savedens, rtol=1e-4, atol=1e-5)

        # compare with nbodykit
        nbodykit_fn = refdir / f'nbodykit_tsc_ngrid{ngrid}.asdf'
        if save_nbodykit and nthread == 1 and dtype == 'f8':
            from nbodykit.source.catalog import ArrayCatalog

            cat = ArrayCatalog({'Position': pos, 'Weight': weights}, BoxSize=box)
            mesh = np.array(
                cat.to_mesh(Nmesh=ngrid, resampler='tsc', compensated=False,
                    interlaced=False, dtype='f4',
                ).compute()
            )
            mesh *= weights.sum(dtype='f8') / ngrid**3

            with asdf.AsdfFile(dict(mesh=mesh)) as af:
                af.write_to(nbodykit_fn, all_array_compression='blsc')
        with asdf.open(nbodykit_fn) as af:
            savedens = af['mesh']
            assert np.allclose(dens, savedens, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('seed', [123,456], ids=['seed1','seed2'])
@pytest.mark.parametrize('dtype', ['f4','f8'])
@pytest.mark.parametrize('npartition', [1,1000], ids=['1p','Np'])
@pytest.mark.parametrize('nthread', [1,-1], ids=['serial','parallel'])
def test_partition(seed, dtype, npartition, nthread):
    from abacusnbody.analysis.tsc import partition_parallel

    rng = np.random.default_rng(seed)
    box = 123.
    N = 10000
    coord = 0
    pos = rng.random((N,3), dtype=dtype)*box
    weights = rng.random((N,), dtype=dtype)

    ppart, starts, wpart = partition_parallel(pos, npartition, box,
        weights=weights, coord=coord, nthread=nthread,
        )

    # Partition with Numpy
    keys = (pos[:,coord] * (npartition/box)).astype(np.int32)
    iord = keys.argsort()
    pos = pos[iord]
    weights = weights[iord]
    np_counts = np.bincount(keys, minlength=npartition)
    np_starts = np.empty(npartition+1, dtype=np.int64)
    np_starts[0] = 0
    np_starts[1:] = np_counts.cumsum()
    assert np.all(np_starts == starts)

    for i in range(npartition):
        assert np.all(
            np.isin(
                ppart[starts[i]:starts[i+1]],
                pos[np_starts[i]:np_starts[i+1]],
            )
        )
        assert np.all(
            np.isin(
                wpart[starts[i]:starts[i+1]],
                weights[np_starts[i]:np_starts[i+1]],
            )
        )
