#!/usr/bin/env python
from distutils.core import setup
setup(name='kvsstcp', version='0',
    description="Key value storage server",
    long_description=open("Readme.md").read(),
    packages=['kvsstcp'],
    package_dir={'kvsstcp':'.'},
    package_data={'kvsstcp':['wskvspage.html']})
