pipe_asdf
=========

Using Unix pipes, we can unpack Abacus ASDF files in Python and pipe the binary output
over stdout so that client C/Fortran/etc programs can just read the raw binary over
stdin.  This directory contains a Python script called ``pipe_asdf.py`` that does
the reading and pipes the data to stdout, and an example analysis program called
``client.c`` that reads the data over stdin.

The ``pipe_asdf.py`` script is normally installed as ``pipe_asdf`` via pip.
However, one can also invoke the ``.py`` file directly from this directory.

For full documentation, see the `abacusutils pipe_asdf documentation <https://abacusutils.readthedocs.io/en/latest/pipes.html>`_.

Example C Client Program
------------------------
The ``client.c`` file is a stand-in for an "analysis" code that reads the raw
binary data for two columns, ``N`` and ``x_com``, and does something with the
data (in this case, just prints some values).

One can build this program with:

.. code-block:: console
    
    $ make

and run it with:

.. code-block:: console

    $ ./pipe_asdf.py halo_info_000.asdf -f N -f x_com | ./client

You can use the example ``halo_info_000.asdf`` file symlinked from this directory to test this.
