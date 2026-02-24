"""SolBench v2 support module for Solar analysis.

This module provides utilities to parse and process kernels from
sol-bench/data/benchmark for Solar analysis.
"""

from .converter import convert_solbenchv2_file
from .parser import parse_solbenchv2_file

__all__ = ["convert_solbenchv2_file", "parse_solbenchv2_file"]
