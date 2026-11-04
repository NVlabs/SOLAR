# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Console logging for SOLAR's command-line entry points.

Library modules report progress and diagnostics through ``logging`` (logger
names under ``solar``).  A CLI calls :func:`configure` once after parsing its
arguments so those records reach the console: ``INFO`` and above by default,
``DEBUG`` with ``--debug`` / ``--verbose``.  Records go to stdout, where the
same messages were printed before they became log records, so scripted users
see no change.  Importing ``solar`` as a library configures nothing.
"""
from __future__ import annotations

import logging
import sys

_MARK = "_solar_cli_handler"


def configure(debug: bool = False, stream=None) -> logging.Logger:
    """Attach (once) a plain-message handler to the ``solar`` logger and set its level."""
    log = logging.getLogger("solar")
    level = logging.DEBUG if debug else logging.INFO
    log.setLevel(level)
    handler = next((h for h in log.handlers if getattr(h, _MARK, False)), None)
    if handler is None:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _MARK, True)
        log.addHandler(handler)
    handler.setLevel(level)
    return log
