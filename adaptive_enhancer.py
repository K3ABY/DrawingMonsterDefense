"""Backward-compatibility shim for adaptive_enhancer."""
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.vision.adaptive_enhancer import *
