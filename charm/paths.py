"""Locations supplied by the caller, independent of the author's computer."""
import os
from pathlib import Path

def run_root():
    return Path(os.environ.get('CHARM_WORKSPACE', 'runs/reproduction')).resolve()

def data_directory():
    return Path(os.environ.get('CHARM_DATA_DIR', 'data/boston')).resolve()
