Release Procedure
=================

- Update the version in ``setup.py``
- Update ``CHANGES.rst`` (check ``git diff vOLD``)
- Make & push a new commit: ``git commit -m "Preparing for release vX.Y.Z"``
- Check that GitHub CI is passing
- Check that docs are building
- Make a new release/git tag on GitHub: https://github.com/abacusorg/abacusutils/releases
- Build and upload the PyPI distributions:

    ::
    
      rm -rf dist/
      python setup.py sdist bdist_wheel
      twine upload dist/*
