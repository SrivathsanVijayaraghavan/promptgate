"""
Root conftest.py — ensures both promptgate and injectionbench packages
are importable during pytest runs in this repository.
"""
import sys
from pathlib import Path

# Add project root to sys.path so injectionbench is importable
# alongside the editable-installed promptgate package.
sys.path.insert(0, str(Path(__file__).resolve().parent))