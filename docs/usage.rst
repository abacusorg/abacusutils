Usage
=====

Python
------
abacusutils contains a Python namespace package called ``abacusnbody``.
This is the name to import (not ``abacusutils``, which is just the name of the PyPI package).
For example, to import the ``compaso_halo_catalog`` module, use
::
    
    import abacusnbody.data.compaso_halo_catalog

The :doc:`compaso` page has the documentation and API for this module.


The API for the other, less commonly used modules in the ``abacusnbody`` namespace is
located here: :doc:`modules`.

Specific examples of how to use abacusutils to work with AbacusSummit data will soon be
given at the AbacusSummit website: https://abacussummit.readthedocs.io

C/C++
-----
Coming soon!  For simple data access patterns, the :ref:`usage:Unix Pipes` approach may suffice.

Unix Pipes
----------
The ``pipe_asdf`` Python script reads columns from ASDF files and pipes them to
``stdout``.  Programs can then read the raw binary from ``stdin`` without having
to worry about the details of file formats or compression.  For example, to pipe
two columns from ``halo_info_000.asdf`` to the ``client`` analysis program, use:

.. code-block:: bash

    $ pipe_asdf halo_info_000.asdf -f N -f x_com | ./client

The ``pipe_asdf`` script is installed when installing abacusutils via pip.
Alternatively, it is available directly in the ``abacusutils/pipe_asdf/`` directory.
An example client program is available in the same directory.

See the documentation here: :doc:`pipes`.
