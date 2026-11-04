# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-binding provenance of the imported ``solar`` engine (119-TRACE-PROVENANCE-1).

Path-invoked scripts under ``solar/scripts`` insert their repository root first
on ``sys.path`` so they import the checkout they live in rather than whatever
editable install is on the path.  The record returned by :func:`engine_source`
-- the resolved ``solar.__file__``, the package's git commit, the blob id of
``__init__.py`` and a dirty flag -- is written by exactly these writers:

* ``solar.translate.receipts.emit_receipt`` (top-level ``engine_source`` in
  every receipt it writes; ``scripts/verify_saved_ir.py`` reaches it through
  ``finalize_verify`` -> ``emit_workload_receipt``);
* ``scripts/trace_saved_ir.py`` (``metadata.engine_source`` in the trace YAML);
* ``scripts/to_einsum_lib.py`` (top-level ``engine_source`` in the saved
  ``*_einsum_lib.yaml`` metadata, injected at the writer boundary;
  :func:`solar.einsum_lib.metadata.build_metadata` also accepts it as an
  optional keyword).

Other scripts call :func:`probe_hook` only (diagnostic emission on request);
artifacts they write do not carry the record.  Comparison drivers that live
outside this repository and write their own JSON must call
:func:`engine_source` themselves -- nothing here binds them.  The record
identifies ``__init__.py`` (SHA-256 and blob) and the package directory's
git state; it is not a whole-repository attestation and it does not hash every
loaded module.

``probe_hook`` is the test/diagnostic entry: when ``SOLAR_ENGINE_SOURCE_PROBE``
is set it prints the record as one ``engine_source {json}`` line to stderr; the
value ``exit`` additionally terminates the process (status 0) right after, so
a script can be probed without running its workload.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _git_run(args, cwd: Path):
    """(ok, stdout) for a git command; ok is False when git is missing or fails."""
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=20)
    except Exception:
        return False, ""
    return out.returncode == 0, out.stdout.strip()


def _git(args, cwd: Path) -> Optional[str]:
    ok, text = _git_run(args, cwd)
    return text if ok and text else None


def engine_source() -> Dict[str, Any]:
    """Resolved origin of the imported ``solar`` package (never raises)."""
    import solar  # the package that THIS interpreter resolved

    init = Path(solar.__file__).resolve()
    pkg = init.parent
    rec: Dict[str, Any] = {
        "solar_file": str(init),
        "package_dir": str(pkg),
        "version": getattr(solar, "__version__", None),
        "python": sys.executable,
        "init_sha256": None,
        "git_toplevel": None,
        "git_commit": None,
        "git_blob_init": None,
        "git_dirty": None,
    }
    try:
        rec["init_sha256"] = hashlib.sha256(init.read_bytes()).hexdigest()
    except Exception:
        pass
    top = _git(["rev-parse", "--show-toplevel"], pkg)
    if top:
        rec["git_toplevel"] = top
        rec["git_commit"] = _git(["rev-parse", "HEAD"], pkg)
        prefix = _git(["rev-parse", "--show-prefix"], pkg) or ""
        rec["git_blob_init"] = _git(["rev-parse", f"HEAD:{prefix}__init__.py"], pkg)
        ok, status = _git_run(["status", "--porcelain", "--", "."], pkg)
        rec["git_dirty"] = bool(status) if ok else None   # False == clean, None == git status failed
    return rec


def emit_engine_source(stream=None) -> Dict[str, Any]:
    """Print ``engine_source {json}`` (one line) to *stream* (default stderr)."""
    rec = engine_source()
    print("engine_source " + json.dumps(rec, sort_keys=True), file=stream or sys.stderr, flush=True)
    return rec


def probe_hook() -> None:
    """Env-gated probe used by path-invoked scripts right after their sys.path fix."""
    mode = os.environ.get("SOLAR_ENGINE_SOURCE_PROBE", "")
    if not mode:
        return
    emit_engine_source()
    if mode.strip().lower() == "exit":
        sys.exit(0)
