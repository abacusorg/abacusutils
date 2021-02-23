Changelog
=========

1.0.0 (upcoming)
----------------

New Features
~~~~~~~~~~~~
- ``CompaSOHaloCatalog`` can read "cleaned" halo catalogs with ``cleaned_halos=True`` (the default) [#6]

Breaking Changes
~~~~~~~~~~~~~~~~
- Can no longer load field particles or L0 halo particles through ``CompaSOHaloCatalog``; use
  ``abacusnbody.data.read_abacus.read_asdf()`` to read the particle files directly instead. [#6]

<<<<<<< HEAD
=======
Enhancements
~~~~~~~~~~~~
- AbacusHOD now supports cleaned catalogs and uses them by default [#6]

- Printing a ``CompaSOHaloCatalog`` now shows the memory usage (also available with ``CompaSOHaloCatalog.nbytes()``) [#6]

>>>>>>> 79e9a2361f1bbf389cc53ca266b4c287a5c5af9c
Deprecations
~~~~~~~~~~~~
- Passing a string to the ``load_subsamples`` argument of ``CompaSOHaloCatalog`` is deprecated;
  use a dict instead, like: ``load_subsamples=dict(A=True, rv=True)``. [#6]

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
