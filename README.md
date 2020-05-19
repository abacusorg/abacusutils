# abacusutils

abacusutils is the primary repository for distribution of code to read and manipulate
data products from the Abacus N-body project.  In particular, these utilities are intended for use
with the [AbacusSummit](https://abacussummit.readthedocs.io) suite of simulations.  We provide
C/C++ and Python 3 code.

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
See the documentation at <https://abacusutils.readthedocs.io>
