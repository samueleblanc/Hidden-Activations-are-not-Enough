"""Root conftest.py — ensure project root sits first on sys.path for tests."""
import sys
from pathlib import Path

_root = str(Path(__file__).parent)

if _root not in sys.path:
    sys.path.insert(0, _root)
elif sys.path[0] != _root:
    sys.path.remove(_root)
    sys.path.insert(0, _root)
