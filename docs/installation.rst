Installation
============

.. highlight:: console

Requirements
------------
abacusutils should work with Python 3.7-3.10 on all mainstream Linux distributions.
MacOS support should be possible, but is presently not working and help would be
needed to finish adding Mac support (see
`this PR <https://github.com/abacusorg/abacusutils/pull/59>`_).

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
    this is no longer required, instead using the `extension mechanism
    <https://asdf.readthedocs.io/en/stable/asdf/extending/extensions.html>`_
    of ASDF 2.8.

All the pip-installed functionality is pure-Python, using numba for any performance-intensive
routines.

Installing from Cloned Repository
---------------------------------
If you want to hack on the abacusutils source code, we recommend that you clone
the repo and install the package in pip "editable mode":

::
    
    $ git clone https://github.com/abacusorg/abacusutils.git
    $ cd abacusutils
    $ pip install -e .[extra]  # install from current dir in editable mode, including extras
    
The ``-e`` flag ("editable") is optional but recommended so that the installed copy is just a
link to the cloned repo (and thus modifications to the Python code will be seen by code that
imports abacusutils).

The ``.[extra]`` syntax says to install from the current directory (``.``), including the
set of "optional dependencies" called ``extra``.  This includes Python packages needed
to run things in the ``scripts`` directory.
    
.. warning::
    If you first install via pip and then later clone the repo, don't forget to
    run ``pip install -e .[extra]`` in the repo.  Otherwise, you will have two
    copies of abacusutils: one cloned, and one installed via pip.
