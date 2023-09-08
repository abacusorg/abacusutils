Installation
============

.. highlight:: console

Requirements
------------
abacusutils should work with Python 3.8-3.11 (and possibly earlier versions, too)
on all mainstream Linux distributions.
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

This will install most dependencies, with the exception of some "hard to install"
dependencies like Corrfunc or classy. (Corrfunc, for example, is considered hard to
install because it has non-Python dependencies that need to be available at install
time).  To install everything, use:
::

    $ pip install abacusutils[all]

All the pip-installed functionality is pure-Python, using numba for any performance-intensive
routines.  The command-line :doc:`pipes` functionality also becomes available after a
pip install.

Developers may wish to use:
    * ``abacusutils[test]``: packages required to run the tests
    * ``abacusutils[docs]``: to build the docs

.. note::
    Previously, a custom fork of ASDF was required.  As of abacusutils 1.0.0,
    this is no longer required, instead using the `extension mechanism
    <https://asdf.readthedocs.io/en/stable/asdf/extending/extensions.html>`_
    of ASDF 2.8.

Note that installing abacusutils should allow you to read any Abacus ASDF file,
even if you don't use abacusutils and go directly through
the ASDF package.  abacusutils installs a setuptools "entry point" that provides
decompression hooks to ASDF so it can read our custom compression.

For Developers
--------------

Installing from Cloned Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to hack on the abacusutils source code, we recommend that you clone
the repo and install the package in pip "editable mode":

::

    $ git clone https://github.com/abacusorg/abacusutils.git
    $ cd abacusutils
    $ pip install -e .[all]  # install all deps from current dir in editable mode

The ``-e`` flag ("editable") is optional but recommended so that the installed copy is just a
link to the cloned repo (and thus modifications to the Python code will be seen by code that
imports abacusutils).

The ``.[all]`` syntax says to install from the current directory (``.``), including the
set of "optional dependencies" called ``all``.

.. warning::
    If you first install via pip and then later clone the repo, don't forget to
    run ``pip install -e .[all]`` in the repo.  Otherwise, you will have two
    copies of abacusutils: one cloned, and one installed via pip.

pre-commit
~~~~~~~~~~
abacusutils uses `pre-commit <https://pre-commit.com/>`_ for linting.
pre-commit will automatically commit fixes for small code style issues when
a PR is opened, or you can install it locally to your repository and have it
apply the fixes locally:

::

    $ pip install pre-commit
    $ cd abacusutils
    $ pre-commit install

The pre-commit checks will run for each commit and modify files to fix
any linting issues.

pytest
~~~~~~
The testing uses `pytest <https://pytest.org/>`_.  This runs as a check in
GitHub Actions, so it's a good idea to run the tests locally before you push.
Install the test dependencies with:
::

    $ pip install abacusutils[test]

And run the tests with

::

    $ pytest

New tests should be added in the ``tests/`` directory, following the example
of the tests that are already in there.  Test discovery is automatic (see
the pytest docs).
