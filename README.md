# abacusutils

abacusutils is a package for reading and manipulating data products from the Abacus N-body project.
In particular, these utilities are intended for use with the [AbacusSummit](https://abacussummit.readthedocs.io)
suite of simulations.  Most of the code is in Python 3, but we also provide some examples of how to
interface with C/C++.

Full API documentation: <https://abacusutils.readthedocs.io>

## Installation
The Python portions of abacusutils are hosted on PyPI and can be installed with:
```
pip install abacusutils
```

The C/C++ code (e.g. the `pack9/` directory) can be downloaded directly by cloning
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
