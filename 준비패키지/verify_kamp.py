"""Compatibility command for the current isolated integration suite.
The previous October script is preserved in legacy/verify_kamp_20261026.py.
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1]/'tests/test_system.py'),run_name='__main__')
