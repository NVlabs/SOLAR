"""SolBench V1 converter module.

Converts benchmark files (with reference_backward/ReferenceModel format)
to Solar-compatible format for analysis.
"""

from .converter import SolBenchV1Converter

__all__ = ['SolBenchV1Converter']
