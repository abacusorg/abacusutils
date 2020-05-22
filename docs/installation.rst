Installation
============

.. highlight:: console

Python Installation
-------------------
For access to the Python functionality of abacusutils, the recommended installation method is via pip:
::
    
    $ pip install git+https://github.com/lgarrison/asdf.git abacusutils

This command installs both our fork of ASDF and the abacusutils package.

Python Dependencies
^^^^^^^^^^^^^^^^^^^
The Python dependencies are numpy, asdf (our fork), blosc, astropy, and numba.
The only "unusual" dependency is asdf, because we require our fork of the project
to be installed (located at: https://github.com/lgarrison/asdf/).  Our fork supports
`blosc compression <https://blosc.org/pages/blosc-in-depth/>`_.


Direct Download
---------------
To use the C/C++ code, the recommended method is to clone the repository directly:
::
    
    $ git clone https://github.com/abacusorg/abacusutils.git


To install the Python package directly from the downloaded repo, one can invoke pip on the cloned repo:
::
    
    $ git clone https://github.com/abacusorg/abacusutils.git
    $ cd abacusutils/
    $ pip install -r requirements.txt -e .  # << note the dot

The ``-e`` flag ("editable") is optional but recommended so that the installed copy is just a
link to the cloned repo (and thus the Python code will behave a little more like the
C code in terms of being able to see the immediate effect of edits).  The requirements file
brings in the ASDF fork.
