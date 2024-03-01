Changelog
=========

2.0.1 (2024-03-01)
------------------
This is a bugfix release primarily to add support for ASDF 3.1.0.
Several other bugs in the ZCV and HOD modules are also fixed.

Fixes
~~~~~
- Fix ASDF error in ZCV module, add ASDF 3.1.0 support [#130]
- bug fix keys() [#126]
- Sandydev: fixed reseeding bug [#127]
- fix reseed bug [#128]
- backward compatible fix for velocity bias [#121]

2.0.0 (2023-11-15)
------------------

abacusutils 2.0 introduces a power spectrum module based on fast, parallelized TSC
and FFT, including grid interlacing and window compensation, that can output bandpowers,
Legendre multipoles, (k,mu) wedges, and more.

Furthermore, there's a new Zel'dovich Control Variates (ZCV) module, and the HOD module
has many additions, performance improvements, and bug fixes.

The set of default installed dependencies has also been reduced to avoid trouble with
source-only distributions. Use ``pip install abacusutils[all]`` if you need functionality
provided by a non-default dependency.

This is a relatively large release, so the version number has been bumped to 2.0.0, since
there may be associated backwards incompatibilities.

Supported Python versions are 3.8-3.11. Python 3.7 continues to work, although we'll
drop support if/when this is no longer the case.

New Features
~~~~~~~~~~~~
- HOD now supports a new ELG conformity model
- Add a power spectrum module, and a zeldovich control variates (ZCV) module that uses it [#68]
- New parallel TSC module [#79]

Fixes
~~~~~
- Bump Numba requirement to fixed version and enable parallelism in env calc [#60]
- Many small bug fixes

Enhancements
~~~~~~~~~~~~
- Add power spectrum to ``metadata`` module [#69]
- Upgrade docs and CI [#71]
- Power spectrum optimization and parallelization [#102]
- Compute Xi(rp,pi) from P(k) [#115]
- Update CI test and build infrastructure [#118]

Installation
~~~~~~~~~~~~
- Refactor optional dependencies; Corrfunc now optional. Migrate build to pyproject.toml. [#89]

1.3.0 (2022-06-08)
------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Dependencies for tests and scripts now factorized under ``abacusutils[test]`` and ``abacusutils[extra]`` [#46]
- Python 3.6 (EOL) support has been dropped [#56]

New Features
~~~~~~~~~~~~
- ``abacusnbody.metadata`` added. Supports querying simulation parameters without downloading simulation data. [#56]

Fixes
~~~~~
- Fix periodicity in theory-box HOD, and add halo LC features [#41]
- Fix read lc rv [#37]

Enhancements
~~~~~~~~~~~~
- Some nice numba accelerations for fenv calculation [#45]
- Made clustering_params optional, among some minor quality of life updates. [#39]
- Reduce memory usage in Menv tree queries [#51]
- HOD now supports two new conformity parameters for ELGs, conf_a, conf_c [#54]

1.2.0 (2022-02-02)
------------------

New Features
~~~~~~~~~~~~
- Now supports Python 3.10 [#19]
- HOD module now works with halo light cone catalogs [#28]

1.1.0 (2022-01-21)
------------------

Fixes
~~~~~
- Fixed issues with QSO incompleteness [#15]
- Fix ``cleandir`` and propagate cleaning info in header [#18]

New Features
~~~~~~~~~~~~
- Add ``filter_func`` superslab filtering to ``CompaSOHaloCatalog`` [#16]
- Add pack9 reader [#25]
- Add light cone catalog reading to ``CompaSOHaloCatalog`` [#11]

Enhancements
~~~~~~~~~~~~
- Sped up RNG for reseeding [#24]

Changes
~~~~~~~
- Migrate testing to GitHub CI; start some linting [#17]
- Automatic versioning and releasing [#27]

1.0.4 (2021-07-15)
------------------

Fixes
~~~~~
- Fix IC parameter in config file and ELG HOD generation

1.0.3 (2021-06-16)
------------------

Fixes
~~~~~
- Fix HOD ``prepare_sim`` error when ``want_AB = False`` [#14]

Changes
~~~~~~~
- Start testing Python 3.9 [#13]

1.0.2 (2021-06-04)
------------------

Changes
~~~~~~~
- Relax numba version requirement for DESI Conda compatibility. Warning: ``numba<0.52`` not fully tested with ``abacusnbody.hod`` package.


1.0.1 (2021-06-03)
------------------

Changes
~~~~~~~
- Use updated directory structure for cleaned catalogs.

1.0.0 (2021-06-02)
------------------

Fixes
~~~~~
- Fixed issue where satellite galaxy halo ID was incorrect.

New Features
~~~~~~~~~~~~
- ``CompaSOHaloCatalog`` can read "cleaned" halo catalogs with ``cleaned=True`` (the default) [#6]

Breaking Changes
~~~~~~~~~~~~~~~~
- Can no longer load field particles or L0 halo particles through ``CompaSOHaloCatalog``; use
  ``abacusnbody.data.read_abacus.read_asdf()`` to read the particle files directly instead. [#6]

Enhancements
~~~~~~~~~~~~
- AbacusHOD now supports cleaned catalogs and uses them by default [#6]

- Printing a ``CompaSOHaloCatalog`` now shows the memory usage (also available with ``CompaSOHaloCatalog.nbytes()``) [#6]

- Our custom fork of ASDF is no longer required [#10]

Deprecations
~~~~~~~~~~~~
- Passing a string to the ``load_subsamples`` argument of ``CompaSOHaloCatalog`` is deprecated;
  use a dict instead, like: ``load_subsamples=dict(A=True, rv=True)``. [#6]

- ``cleaned_halos`` renamed to ``cleaned``

0.4.0 (2021-02-03)
------------------

New Features
~~~~~~~~~~~~
- Add ``AbacusHOD`` module for fast HOD generation using AbacusSummit simulations [#4]

- ``CompaSOHaloCatalog`` constructor now takes field names in the ``unpack_bits`` field

Enhancements
~~~~~~~~~~~~
- Bump minimum Blosc version to support zero-copy decompression in our ASDF fork

0.3.0 (2020-08-11)
------------------

Enhancements
~~~~~~~~~~~~
- Use 4 Blosc threads for decompression by default

Fixes
~~~~~
- Specify minimum Astropy version to avoid
  ``AttributeError: 'numpy.ndarray' object has no attribute 'info'``

0.2.0 (2020-07-08)
------------------

New Features
~~~~~~~~~~~~
- Add pipe_asdf.py script as an example of using Python to deal with file container
  so that C/Fortran/etc don't have to know about ASDF or blosc

0.1.0 (2020-06-24)
------------------

New Features
~~~~~~~~~~~~
- CompaSOHaloCatalog accepts ``fields`` keyword to limit the IO and unpacking to
  the requsted halo catalog columns

0.0.5 (2020-05-26)
------------------

- First stable release
