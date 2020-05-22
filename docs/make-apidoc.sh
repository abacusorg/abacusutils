#!/usr/bin/env bash

# Runs sphinx-apidoc to generate autodoc templates from the Python source

sphinx-apidoc -o . ../abacusnbody --implicit-namespaces $@
