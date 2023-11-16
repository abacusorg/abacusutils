# Release Procedure

- Update `CHANGES.rst` (check `git diff vOLD`)
- Check that GitHub CI is passing
- Check that docs are building
- Make a new git tag:
```console
$ git tag -a vX.Y.Z -m 'Version X.Y.Z'
$ git push origin vX.Y.Z
```
- Check that GitHub Actions built, tested, and uploaded the release to PyPI
- Make a GitHub release of tag vX.Y.Z: https://github.com/abacusorg/abacusutils/releases
