# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load .py modules from their CURRENT source bytes.

The default import machinery consults ``__pycache__``: a timestamp-valid
stale ``.pyc`` (equal source byte length, preserved mtime) executes OLD
bytecode while the new source bytes get hashed — round-9 P1: the meta-path
finder in ``solar.translate._runner`` never sees a direct
``spec_from_file_location(...).exec_module()`` load, so the MAIN adapter
could still execute stale bytecode.  Every direct ``.py`` load in the
frontends therefore goes through :func:`load_source_module`, which compiles
the bytes it reads from disk and never consults a bytecode cache.

During a verification run the runner installs a record here
(:func:`record_executed_sources`) so the sha256 of the exact bytes handed to
``compile()`` is captured as execution evidence; the runner refuses strict
qualification when the adapter's executed bytes cannot be tied to the
hashed source bytes.
"""

import contextlib
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path

# Records installed by the verification runner while a reference is being
# prepared: {resolved_path: sha256-of-the-bytes-handed-to-compile()}.
_ACTIVE_RECORDS: list = []


@contextlib.contextmanager
def record_executed_sources(record: dict):
    """Capture the executed-bytes hash of every module loaded through
    :func:`load_source_module` while the context is active."""
    _ACTIVE_RECORDS.append(record)
    try:
        yield record
    finally:
        _ACTIVE_RECORDS.remove(record)


class _SourceBytesLoader(importlib.machinery.SourceFileLoader):
    """Compile the module from its current source bytes, never a .pyc."""

    def get_code(self, fullname):
        data = self.get_data(self.path)
        digest = hashlib.sha256(data).hexdigest()
        resolved = str(Path(self.path).resolve())
        for record in _ACTIVE_RECORDS:
            record[resolved] = digest
        return self.source_to_code(data, self.path)


def load_source_module(file_path, module_name: str, *, register: bool = False):
    """Import *file_path* as *module_name* from its current source bytes.

    Bytecode caches are bypassed entirely.  With ``register=True`` the
    module is placed in ``sys.modules`` BEFORE execution (the JAX frontend's
    original behavior, so self-referential imports resolve); otherwise it is
    not registered at all.
    """
    import sys

    path = Path(file_path).resolve()
    loader = _SourceBytesLoader(module_name, str(path))
    spec = importlib.util.spec_from_file_location(
        module_name, str(path), loader=loader)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[module_name] = module
    loader.exec_module(module)
    return module
