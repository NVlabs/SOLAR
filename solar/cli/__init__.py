"""Solar CLI package.

Keep this module import-light: users often run subcommands via
`python -m solar.cli.<subcommand>`, and eager importing of sibling modules here
can cause `runpy` warnings and subtle import-order issues.
"""

__all__ = []