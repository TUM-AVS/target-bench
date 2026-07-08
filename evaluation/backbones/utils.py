"""Small utilities for loading pipeline files without package-name collisions."""

import importlib.util
import os
import sys


def add_path(path):
    if path and os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
