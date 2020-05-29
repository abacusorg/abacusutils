# abacusutils

[![Documentation Status](https://readthedocs.org/projects/abacusutils/badge/?version=latest)](https://abacusutils.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/abacusutils)](https://pypi.org/project/abacusutils/) [![Travis (.org)](https://img.shields.io/travis/abacusorg/abacusutils)](https://travis-ci.com/github/abacusorg/abacusutils)

abacusutils is a package for reading and manipulating data products from the Abacus N-body project.
In particular, these utilities are intended for use with the [AbacusSummit](https://abacussummit.readthedocs.io)
suite of simulations.  Most of the code is in Python 3, but we also provide some examples of how to
interface with C/C++.

Full API documentation: <https://abacusutils.readthedocs.io>

## Installation
The Python abacusutils package is hosted on PyPI and can be installed
by installing "abacusutils" and our fork of the ASDF library with the following command:
```
pip install git+https://github.com/lgarrison/asdf.git abacusutils
```

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
The abacusutils PyPI package contains a Python package called `abacusnbody`.
This is the name to import (not `abacusutils`, which is just the name of the PyPI package).
For example, to import the `compaso_halo_catalog` module, use
```python
import abacusnbody.data.compaso_halo_catalog
```

See the full documentation at <https://abacusutils.readthedocs.io>

Specific examples of how to use abacusutils to work with AbacusSummit data are given
at the AbacusSummit website: <https://abacussummit.readthedocs.io>
