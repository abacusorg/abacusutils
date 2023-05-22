CompaSO Halo Catalogs
=====================

The ``compaso_halo_catalog`` module loads halo catalogs from CompaSO, Abacus's
on-the-fly halo finder.  The module defines one class, ``CompaSOHaloCatalog``,
whose constructor takes the path to a halo catalog as an argument.
Users should use this class as the primary interface to load
and manipulate halo catalogs.

The halo catalogs and particle subsamples are stored on disk in
ASDF files and are loaded into memory as Astropy tables.  Each
column of an Astropy table is essentially a Numpy array and can
be accessed with familiar Numpy-like syntax.  More on Astropy
tables here: http://docs.astropy.org/en/stable/table/

Beyond just loading the halo catalog files into memory, this
module performs a few other manipulations.  Many of the halo
catalog columns are stored in bit-packed formats (e.g.
floats are scaled to a ratio from 0 to 1 then packed in 16-bit
ints), so these columns are unpacked as they are loaded.

Furthermore, the halo catalogs for big simulations are divided
across a few dozen files.  These files are transparently loaded
into one monolithic Astropy table if one passes a directory
to ``CompaSOHaloCatalog``; to save memory by loading only one file,
pass just that file as the argument to ``CompaSOHaloCatalog``.

Importantly, because ASDF and Astropy tables are both column-
oriented, it can be much faster to load only the subset of
halo catalog columns that one needs, rather than all 60-odd
columns.  Use the ``fields`` argument to the ``CompaSOHaloCatalog``
constructor to specify a subset of fields to load.  Similarly, the
particles can be quite large, and one can use the ``subsamples``
argument to restrict the particles to the subset one needs.

Some brief examples and technical details about the halo catalog
layout are presented below, followed by the full module API.
Examples of using this module to work with AbacusSummit data can
be found on the AbacusSummit website here:
https://abacussummit.readthedocs.io


Short Example
-------------
>>> from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
>>> # Load the RVs and PIDs for particle subsample A
>>> cat = CompaSOHaloCatalog('/storage/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.100', subsamples=dict(A=True,pos=True))
>>> print(cat.halos[:5])  # cat.halos is an Astropy Table, print the first 5 rows
   id    npstartA npstartB ... sigmavrad_L2com sigmavtan_L2com rvcirc_max_L2com
-------- -------- -------- ... --------------- --------------- ----------------
25000000        0        2 ...       0.9473971      0.96568024      0.042019103
25000001       11       12 ...      0.86480814       0.8435805      0.046611086
48000000       18       15 ...      0.66734606      0.68342227      0.033434115
58000000       31       18 ...      0.52170926       0.5387341      0.042292822
58001000       38       23 ...       0.4689916      0.40759262      0.034498636
>>> print(cat.halos['N','x_com'][:5])  # print the position and mass of the first 5 halos
  N         x_com [3]
--- ------------------------
278 -998.88525 .. -972.95404
 45  -998.9751 .. -972.88416
101   -999.7485 .. -947.8377
 82    -998.904 .. -937.6313
 43   -999.3252 .. -937.5813
>>> # Examine the particle subsamples associated with the 5th halo
>>> h5 = cat.halos[4]
>>> print(cat.subsamples['pos'][h5['npstartA']:h5['npstartA'] + h5['npoutA']])
        pos [3]
------------------------
  -999.3019 .. -937.5229
 -999.33435 .. -937.5515
-999.38965 .. -937.58777
>>> # At a glance, the pos fields match that of the 5th halo above, so it appears we have indexed correctly!


Catalog Structure
-----------------
The catalogs are stored in a directory structure that looks like:

.. code-block:: none

    - SimulationName/
        - halos/
            - z0.100/
                - halo_info/
                    halo_info_000.asdf
                    halo_info_001.asdf
                    ...
                - halo_rv_A/
                    halo_rv_A_000.asdf
                    halo_rv_A_001.asdf
                    ...
                - <field & halo, rv & PID, subsample A & B directories>
            - <other redshift directories, some with particle subsamples, others without>

The file numbering roughly corresponds to a planar chunk of the simulation
(all y and z for some range of x).  The matching of the halo_info file numbering
to the particle file numbering is important; halo_info files index into the
corresponding particle files.

The halo catalogs are stored on disk in ASDF files (https://asdf.readthedocs.io/).
The ASDF files start with a human-readable header that describes
the data present in the file and metadata about the simulation
from which it came (redshift, cosmology, etc).  The rest of
the file is binary blobs, each representing a column (i.e.
a halo property).

Internally, the ASDF binary portions are usually compressed.  This should
be transparent to users, although you may be prompted to install the
blosc package if it is not present.  Decompression should be fast,
at least 500 MB/s per core.


Particle Subsamples
-------------------
We define two disjoint sets of "subsample" particles, called "subsample A" and
"subsample B".  Subsample A is a few percent of all particles, with subsample B
a few times larger than that.  Particles membership in each group is a function
of PID and is thus consistent across redshift.

At most redshifts, we only output halo catalogs and halo subsample particle PIDs.
This aids with construction of merger trees.  At a few redshifts, we provide
subsample particle positions as well as PIDs, for both halo particles and
non-halo particles, called "field" particles.  Only halo particles (specifically,
:doc:`L1 particles<summit:compaso>`) may be loaded through this module; field
particles and L0 halo particles can be loaded by reading the particle files
directly with :ref:`abacusnbody.data:read_abacus module`.

Use the ``subsamples`` argument to the constructor to specify loading
subsample A and/or B, and which fields---pos, vel, pid---to load.  Note that
if only one of pos & vel is specified, the IO amount is the same, because
the pos & vel are packed together in :ref:`RVint format<compaso:Bit-packed Formats>`.
But the memory usage and time to unpack will be lower.


Halo File Types
---------------
Each file type (for halos, particles, etc) is grouped into a subdirectory.
These subdirectories are:

- ``halo_info/``
  The primary halo catalog files.  Contains stats like
  CoM positions and velocities and moments of the particles.
  Also indicates the index and count of subsampled particles in the
  ``halo_pid_A/B`` and ``halo_rv_A/B`` files.

- ``halo_pid_A/`` and ``halo_pid_B/``
  The 64-bit particle IDs of particle subsamples A and B.  The PIDs
  contain information about the Lagrangian position of the particles,
  whether they are tagged, and their local density.

The following subdirectories are only present for the redshifts for which
we output particle subsamples and not just halo catalogs:

- ``halo_rv_A/`` and ``halo_rv_B/``
  The positions and velocities of the halo subsample particles, in "RVint"
  format. The halo associations are recoverable with the indices in the
  ``halo_info`` files.

- ``field_rv_A/`` and ``field_rv_B/``
  Same as ``halo_rv_<A|B>/``, but only for the field (non-halo) particles.

- ``field_pid_A/`` and ``field_pid_B/``
  Same as ``halo_pid_<A|B>/``, but only for the field (non-halo) particles.


Bit-packed Formats
------------------
The "RVint" format packs six fields (x,y,z, and vx,vy,vz) into three ints (12 bytes).
Positions are stored to 20 bits (global), and velocities 12 bits (max 6000 km/s).

The PIDs are 8 bytes and encode a local density estimate, tag bits for merger trees,
and a unique particle id, the last of which encodes the Lagrangian particle coordinate.

These are described in more detail on the :doc:`AbacusSummit Data Model page <summit:data-products>`.

Use the ``unpack_bits`` argument of the ``CompaSOHaloCatalog`` constructor to specify
which PID bit fields you want unpacked.  Be aware that some of them might use a lot of
memory; e.g. the Lagrangian positions are three 4-byte floats per subsample particle.
Also be aware that some of the returned arrays use narrow int dtypes to save memory,
such as the ``lagr_idx`` field using ``int16``.  It is easy to silently overflow such
narrow int types; make sure your operations stay within the type width and cast
if necessary.


Field Subset Loading
--------------------
Because the ASDF files are column-oriented, it is possible to load just one or a few
columns (halo catalog fields) rather than the whole file.  This can save huge amounts
of IO, memory, and CPU time (the latter due to the decompression).  Use the ``fields`` argument
to the ``CompaSOHaloCatalog`` constructor to specify the list of columns you want.

In detail, some columns are stored as ratios to other columns.  For example, ``r90``
is stored as a ratio relative to ``r100``.  So to properly unpack
``r90``, the ``r100`` column must also be read.  ``CompaSOHaloCatalog`` knows about
these dependencies and will load the minimum set necessary to return the requested
columns to the user.  However, this may result in more IO than expected.  The ``verbose``
constructor flag or the ``dependency_info`` field of the ``CompaSOHaloCatalog``
object may be useful for diagnosing exactly what data is being loaded.

Despite the potential extra IO and CPU time, the extra memory usage is granular
at the level of individual files.  In other words, when loading multiple files,
the concatenated array will never be constructed for columns that only exist for
dependency purposes.


Superslab (Chunk) Processing
----------------------------
The halo catalogs are divided across multiple files, called "superslabs", which are
typically planar chunks of the simulation volume (all y,z for some range of x, with
a bit of overlap at the boundaries).  Applications that can process the volume
superslab-by-superslab can save a substantial amount of memory compared to loading
the full volume.  To load a single superslab, pass the corresponding ``halo_info_XXX.asdf``
file as the ``path`` argument:

.. code-block:: python

    cat = CompaSOHaloCatalog('AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_000.asdf')

If your application needs one slab of padding, you can pass a list of files and proceed in a
rolling fashion:

.. code-block:: python

    cat = CompaSOHaloCatalog(['AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_033.asdf',
                              'AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_000.asdf',
                              'AbacusSummit_base_c000_ph000/halos/z0.100/halo_info/halo_info_001.asdf'])


Superslab Filtering
-------------------
Another way to save memory is to use the ``filter_func`` argument.  This function
will be called for each superslab, and must return a mask representing the rows
to keep.  For example, to drop all halos with less than 100 particles, use:

.. code-block:: python

    cat = CompaSOHaloCatalog(..., filter_func=lambda h: h['N'] >= 100)

Because this mask is applied on each superslab immediately after loading,
the full, unfiltered table is never constructed, thus saving memory.

The filtering adds some CPU time, but in many cases loading catalogs is
IO limited, so this won't add much overhead.


Multi-threaded Decompression
----------------------------
The Blosc compression we use inside the ASDF files supports multi-threaded
decompression.  We have packed AbacusSummit files with 4 Blosc blocks (each ~few MB)
per ASDF block, so 4 Blosc threads is probably the optimal value.  This is the
default value, unless fewer cores are available (as determined by the process
affinity mask).

.. note::

    Loading a CompaSOHaloCatalog will use 4 decompression threads by default.

You can control the number of decompression threads with:

.. code-block:: python

    import abacusnbody.data.asdf
    abacusnbody.data.asdf.set_nthreads(N)


API
---

.. automodule:: abacusnbody.data.compaso_halo_catalog
   :members:
   :undoc-members:
   :show-inheritance:
