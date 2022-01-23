Release Procedure
=================

- Update ``CHANGES.rst`` (check ``git diff vOLD``)
- Check that GitHub CI is passing
- Check that docs are building
- Make a new release/git tag on GitHub: https://github.com/abacusorg/abacusutils/releases
- Check that GitHub Actions pushed a PyPI release from the GitHub release
