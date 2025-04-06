"""Modifies the Python path for tests, allowing it to find module."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))