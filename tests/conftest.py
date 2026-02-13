import os
import sys


def _add_project_root_to_path():
    # Ensure imports like `import automl_gui` work regardless of pytest CWD.
    tests_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(tests_dir, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_add_project_root_to_path()
