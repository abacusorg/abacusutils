Unix Pipes: pipe_asdf
=====================

.. automodule:: abacusnbody.data.pipe_asdf
   :members:
   :undoc-members:
   :show-inheritance:

   Python API
   ==========

Example C Client Code
---------------------
An example C program called ``client.c`` that receives data over a pipe
is given in the `abacusutils/pipe_asdf directory <https://github.com/abacusorg/abacusutils/tree/master/pipe_asdf>`_.

From this directory, one can build the ``client`` program by running 

.. code-block:: bash
    
    $ make

and run it with:

.. code-block:: bash

    $ ./pipe_asdf.py halo_info_000.asdf -f N -f x_com | ./client

You can use the example ``halo_info_000.asdf`` file symlinked in the ``pipe_asdf`` directory to test this.

This program is a stand-in for an analysis code. In this case, it just reads the raw
binary data for two columns, ``N`` and ``x_com``, and prints the values.
