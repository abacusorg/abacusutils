import os

from setuptools import setup, find_namespace_packages

install_requires = ['numpy>=1.16','blosc>=1.9.2','astropy>=4.0.0','scipy>=1.5.0','numba>=0.50','asdf>=2.8','h5py','pyyaml']

# enable "pip install abacusutils[test]" and "abacusutils[extra]"
# "extra" will be everything used by scripts but not the importable code
extras_require = dict(test=['pytest'],
                      extra=['emcee','schwimmbad','getdist','dynesty','dill', 'click'],
                    )

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
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires='>=3.7',
    install_requires = install_requires,
    extras_require = extras_require,
    entry_points={'console_scripts':['pipe_asdf = abacusnbody.data.pipe_asdf:main'],
                  'asdf.extensions':['abacusutils = abacusnbody.data.asdf:AbacusExtension']}
)
