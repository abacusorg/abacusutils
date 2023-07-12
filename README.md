# abacusutils

<p align="center">
<img src="docs/images/icon_red.png" width="175px" alt="Abacus Logo">
</p>

[![Documentation Status](https://readthedocs.org/projects/abacusutils/badge/?version=latest)](https://abacusutils.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/abacusutils)](https://pypi.org/project/abacusutils/) [![Tests](https://github.com/abacusorg/abacusutils/actions/workflows/tests.yml/badge.svg)](https://github.com/abacusorg/abacusutils/actions/workflows/tests.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/abacusorg/abacusutils/master.svg)](https://results.pre-commit.ci/latest/github/abacusorg/abacusutils/master)

abacusutils is a package for reading and manipulating data products from the Abacus *N*-body project.
In particular, these utilities are intended for use with the [AbacusSummit](https://abacussummit.readthedocs.io>)
suite of simulations.  The package focuses on the Python 3 API, but there is also a language-agnostic Unix pipe
interface to some of the functionality.

These interfaces are documented here: <https://abacusutils.readthedocs.io>

Press the GitHub "Watch" button in the top right and select "Custom->Releases" to be notified about bug fixes
and new features!

## Installation
The Python abacusutils package is hosted on PyPI and can be installed
by installing "abacusutils":
```
pip install abacusutils
```
or
```
pip install abacusutils[all]
```

For more information, see <https://abacusutils.readthedocs.io/en/latest/installation.html>.

## Usage
abacusutils has multiple interfaces, summarized here and at <https://abacusutils.readthedocs.io/en/latest/usage.html>.

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
