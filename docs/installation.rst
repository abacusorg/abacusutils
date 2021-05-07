Installation
============

.. highlight:: console

Pip Installation
----------------
For access to the Python functionality of abacusutils, you can either install via pip
or clone from GitHub.  The pip installation is recommended if you don't need to modify
the source:
::
    
    $ pip install git+https://github.com/lgarrison/asdf.git abacusutils

This command installs both our fork of ASDF and the abacusutils package.
It also gives access to the :doc:`pipes` functionality.

Python Dependencies
^^^^^^^^^^^^^^^^^^^
The Python dependencies are numpy, asdf (our fork), blosc, astropy, and numba.
The only "unusual" dependency is asdf, because we require our fork of the project
to be installed (located at: https://github.com/lgarrison/asdf/).  Our fork supports
`blosc compression <https://blosc.org/pages/blosc-in-depth/>`_.


Installing from Cloned Repository
---------------------------------
If you want to hack on the abacusutils source code (or use the C/C++ code),
then the recommendation is to clone the repo and install the package in
pip "editable mode":
::
    $ git clone https://github.com/abacusorg/abacusutils.git
    $ cd abacusutils
    $ pip install -r requirements.txt -e ./  # install from current dir in editable mode
    
The ``-e`` flag ("editable") is optional but recommended so that the installed copy is just a
link to the cloned repo (and thus modifications to the Python code will be seen by code that
imports abacusutils).  The requirements file brings in the ASDF fork.
    
.. warning::
    If you download via pip and then later clone the repo, don't forget to
    run ``pip install -e ./`` in the repo.  Otherwise, you will have two
    copies of abacusutils: one cloned, and one downloaded via pip.
