"""Backward-compatibility shim for one_euro_filter."""
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.filters.one_euro_filter import *
