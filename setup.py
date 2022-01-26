import os

from setuptools import setup, find_namespace_packages

install_requires = ['numpy>=1.16','blosc>=1.9.2','astropy>=4.0.0','scipy','numba>=0.50','asdf>=2.8','h5py','emcee','schwimmbad','getdist','dynesty','sklearn']

# If we're on ReadTheDocs, can't install packages with C dependencies, like Corrfunc
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    # This list must be kept synchronized with `autodoc_mock_imports` in docs/conf.py
    install_requires += ['Corrfunc>=2']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="abacusutils",
    author="Daniel Eisenstein, Philip Pinto, Lehman Garrison, Nina Maksimova, Sownak Bose, Boryana Hadzhiyska, Sihan (Sandy) Yuan",
    author_email="lgarrison@flatironinstitute.org",
    description="Python and C/C++ code to read halo catalogs and other Abacus N-body data products",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abacusorg/abacusutils",
    packages=find_namespace_packages(include=['abacusnbody.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.6',
    install_requires = install_requires,
    entry_points={'console_scripts':['pipe_asdf = abacusnbody.data.pipe_asdf:main'],
                  'asdf.extensions':['abacusutils = abacusnbody.data.asdf:AbacusExtension']}
)
