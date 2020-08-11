Release Procedure
=================

- Update the version in ``setup.py``
- Update ``CHANGES.rst``
- Check that Travis is passing
- Check that docs are building
- Make a new release/git tag on Github: https://github.com/abacusorg/abacusutils/releases
- Build the PyPI distributions:

    ::
    
      rm -rf dist/
      python setup.py sdist bdist_wheel
      twine upload dist/*
      
