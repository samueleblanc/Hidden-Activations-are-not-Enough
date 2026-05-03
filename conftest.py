"""Root conftest.py — prevent unit_test/km_feature_viz/ from shadowing the real package.

pytest adds the *parent* of a package-based test directory (i.e. unit_test/) to
sys.path, which makes unit_test/km_feature_viz/__init__.py shadow the real
km_feature_viz/ package.  We pre-import the real package here (before pytest
touches sys.path) so the correct entry is already cached in sys.modules and the
shadow __init__.py is never loaded in its place.
"""
import sys
from pathlib import Path

_root = str(Path(__file__).parent)

# Keep the project root first so any subsequent sys.path additions can't win.
if _root not in sys.path:
    sys.path.insert(0, _root)
elif sys.path[0] != _root:
    sys.path.remove(_root)
    sys.path.insert(0, _root)

# Pre-import the real km_feature_viz package into sys.modules so pytest's
# later insertion of unit_test/ onto sys.path cannot shadow it.
import km_feature_viz  # noqa: E402  (project root is now first on sys.path)
import km_feature_viz.compute_kms  # noqa: E402
