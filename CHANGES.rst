Changelog
=========
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
