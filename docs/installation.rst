Installation
============

.. highlight:: console

Pip Installation
----------------
For access to the Python functionality of abacusutils, you can either install via pip
or clone from GitHub.  The pip installation is recommended if you don't need to modify
the source:
::
    
    $ pip install abacusutils

This command will also give access to the command-line :doc:`pipes` functionality.

.. note::
    Previously, a custom fork of ASDF was required.  As of abacusutils 1.0.0,
    this is no longer required, instead using the extension mechanism of ASDF 2.8.

All the pip-installed functionality is pure-Python, using numba for any performance-intensive
routines.

Installing from Cloned Repository
---------------------------------
If you want to hack on the abacusutils source code (or use the C/C++ code),
then the recommendation is to clone the repo and install the package in
pip "editable mode":

::
    $ git clone https://github.com/abacusorg/abacusutils.git
    $ cd abacusutils
    $ pip install -e ./  # install from current dir in editable mode
    
The ``-e`` flag ("editable") is optional but recommended so that the installed copy is just a
link to the cloned repo (and thus modifications to the Python code will be seen by code that
imports abacusutils).
    
.. warning::
    If you download via pip and then later clone the repo, don't forget to
    run ``pip install -e ./`` in the repo.  Otherwise, you will have two
    copies of abacusutils: one cloned, and one downloaded via pip.
