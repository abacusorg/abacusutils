# abacusutils

[![Documentation Status](https://readthedocs.org/projects/abacusutils/badge/?version=latest)](https://abacusutils.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/abacusutils)](https://pypi.org/project/abacusutils/) [![Build Status](https://travis-ci.com/abacusorg/abacusutils.svg?branch=master)](https://travis-ci.com/abacusorg/abacusutils)

abacusutils is a package for reading and manipulating data products from the Abacus N-body project.
In particular, these utilities are intended for use with the [AbacusSummit](https://abacussummit.readthedocs.io)
suite of simulations.  We provide multiple interfaces: primarily Python 3, but also C/C++ [coming soon!] and
language-agnostic interfaces like Unix pipes.

These interfaces are documented here: <https://abacusutils.readthedocs.io>

Press the GitHub "Watch" button in the top right and select "Releases Only" to be notified about bug fixes
and new features!  This package is still in early stages, and bugs are likely to be identified and squashed,
and new performance opportunities identified.

## Installation
The Python abacusutils package is hosted on PyPI and can be installed
by installing "abacusutils" and our fork of the ASDF library with the following command:
```
pip install git+https://github.com/lgarrison/asdf.git abacusutils
```

The Unix pipe interface (`pipe_asdf`) is also installed as part of the pip install.

The C/C++ code (coming soon!) can be downloaded directly by cloning
this repository:
```
git clone https://github.com/abacusorg/abacusutils.git
```
or by downloading a zip archive of the repository:
```
wget https://github.com/abacusorg/abacusutils/archive/master.zip
```

### Python Dependencies
The Python dependencies are numpy, asdf (our fork), blosc, astropy, and numba.
The only "unusual" dependency is asdf, because we require our fork of the project
to be installed (located at: https://github.com/lgarrison/asdf/).  Our fork supports
[blosc compression](https://blosc.org/pages/blosc-in-depth/).

## Usage
abacusutil has multiple interfaces, summarized here and at <https://abacusutils.readthedocs.io/en/latest/usage.html>.

Specific examples of how to use abacusutils to work with AbacusSummit data will soon
be given at the AbacusSummit website: <https://abacussummit.readthedocs.io>

### Python
The abacusutils PyPI package contains a Python package called `abacusnbody`.
This is the name to import (not `abacusutils`, which is just the name of the PyPI package).
For example, to import the `compaso_halo_catalog` module, use
```python
import abacusnbody.data.compaso_halo_catalog
```

### Unix Pipes
The ``pipe_asdf`` Python script reads columns from ASDF files and pipes them to
``stdout``.  For example:

```bash
    $ pipe_asdf halo_info_000.asdf -f N -f x_com | ./client
```
