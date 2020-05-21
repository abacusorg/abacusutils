Installation
============

.. highlight:: console

Python Installation
-------------------
For access to the Python functionality of abacusutils, the recommended installation method is via pip:
::
    
    $ pip install abacusutils


Direct Download
---------------
To use the C/C++ code, the recommended method is to clone the repository directly:
::
    
    $ git clone https://github.com/abacusorg/abacusutils.git


To install the Python package directly from the downloaded repo, one can invoke pip on the cloned repo:
::
    
    $ git clone https://github.com/abacusorg/abacusutils.git
    $ cd abacusutils/
    $ pip install -e .  # << note the dot

The ``-e`` flag ("editable") is optional but recommended so that the installed copy is just a
symlink to the cloned repo (and thus the Python code will behave a little more like the
C code in terms of being able to see the immediate effect of edits).
