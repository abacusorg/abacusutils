.. abacusutils documentation master file, created by
   sphinx-quickstart on Tue May 19 12:00:18 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
.. image:: images/icon_red.png
  :width: 175px
  :alt: Abacus logo
  :align: center

abacusutils
===========

abacusutils is a package for reading and manipulating data products from the Abacus *N*-body project.
In particular, these utilities are intended for use with the `AbacusSummit <https://abacussummit.readthedocs.io>`_
suite of simulations.  We provide multiple interfaces: primarily Python 3, but also C/C++ [coming soon!] and
language-agnostic interfaces like Unix pipes.

abacusutils is hosted in the `abacusorg/abacusutils <https://github.com/abacusorg/abacusutils>`_ GitHub repository.  Please report
bugs and ask questions by opening an issue in that repository.

While you're there, press the GitHub "Watch" button in the top right and select "Custom->Releases" to be notified about bug fixes
and new features!  This package is still in early stages, and bugs are likely to be identified and squashed,
and new performance opportunities identified.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: Main Modules

   compaso
   hod
   pipes

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules
   CHANGES


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
