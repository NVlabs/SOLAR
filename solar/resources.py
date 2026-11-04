# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate SOLAR's non-Python workflow resources in any supported layout.

SOLAR ships resources that live *next to* the ``solar`` package in a source
checkout: hardware configs (``configs/arch/*.yaml``), the DSL prompt
references the translation agent loads (``docs/einsum_lib_prompt*.md``), the
coding-agent skills (``skills/``), the deterministic verification and tracing
scripts (``scripts/verify_saved_ir.py``, ``scripts/trace_saved_ir.py``,
``scripts/diff_ir_stages.py``), the torchview patch and the example models.
A wheel install has no checkout, so the build (``setup.py``) copies the
trees listed in :data:`RESOURCE_TREES` into the package as ``solar/_data/``.

Lookup order, identical for every caller:

1. ``$SOLAR_RESOURCE_ROOT`` when set (a checkout or an unpacked sdist);
2. the source checkout that contains this package (editable or in-tree use);
3. the packaged copy ``solar/_data`` (wheel install).

Nothing falls back silently: :func:`resource_path` raises
:class:`ResourceNotFound` naming every location that was searched, and
:func:`find_resource` returns ``None`` for callers that keep their own
documented fallback.  This module imports only the standard library so
``setup.py`` can load it without importing ``solar``.

Command line::

    python -m solar.resources                      # show the roots in use
    python -m solar.resources configs/arch/B200.yaml   # print one resolved path
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

ENV_ROOT = "SOLAR_RESOURCE_ROOT"

#: Paths (relative to the checkout root) that a distribution must carry so the
#: documented workflows run outside a checkout.  ``setup.py`` copies each one
#: into ``solar/_data/<same relative path>`` when building a wheel.
RESOURCE_TREES: tuple[str, ...] = (
    "AGENTS.md",
    "configs",
    "docs",
    "examples",
    "examples_jax",
    "patches",
    "scripts/diff_ir_stages.py",
    "scripts/gen_solexecbench_adapter.py",
    "scripts/trace_saved_ir.py",
    "scripts/verify_saved_ir.py",
    "skills",
)

#: A root is a checkout only when both of these exist beneath it.
_CHECKOUT_MARKERS = ("pyproject.toml", "configs/arch")


class ResourceNotFound(FileNotFoundError):
    """A workflow resource is missing from every searched location."""


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def candidate_roots() -> list[tuple[str, Path]]:
    """Ordered ``(kind, path)`` pairs that may hold resources; existence not checked."""
    roots: list[tuple[str, Path]] = []
    env = os.environ.get(ENV_ROOT)
    if env:
        roots.append(("env:" + ENV_ROOT, Path(env).expanduser()))
    roots.append(("checkout", _package_dir().parent))
    roots.append(("packaged", _package_dir() / "_data"))
    return roots


def _root_is_usable(kind: str, root: Path) -> bool:
    if kind == "checkout":
        return all((root / m).exists() for m in _CHECKOUT_MARKERS)
    return root.is_dir()


def resource_roots() -> list[tuple[str, Path]]:
    """The usable roots, in lookup order."""
    return [(k, r) for k, r in candidate_roots() if _root_is_usable(k, r)]


def find_resource(*parts: str) -> Optional[Path]:
    """Return the first existing ``root/parts`` or ``None``."""
    for _, root in resource_roots():
        p = root.joinpath(*parts)
        if p.exists():
            return p
    return None


def resource_path(*parts: str) -> Path:
    """Return the first existing ``root/parts``; raise :class:`ResourceNotFound` otherwise."""
    found = find_resource(*parts)
    if found is not None:
        return found
    rel = "/".join(parts)
    searched = [f"{k} -> {r}" for k, r in candidate_roots()]
    raise ResourceNotFound(
        f"SOLAR resource {rel!r} not found. Searched (in order): "
        + "; ".join(searched)
        + f". Set {ENV_ROOT} to a SOLAR checkout or unpacked sdist, or install "
        "a wheel built from the sdist (it carries solar/_data)."
    )


def arch_config_path(name_or_path: str) -> Path:
    """Resolve ``--arch-config``: an existing file path, else ``configs/arch/<name>.yaml``."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    return resource_path("configs", "arch", f"{name_or_path}.yaml")


def list_arch_configs() -> list[str]:
    """Names of the shipped hardware configs (without ``.yaml``)."""
    d = find_resource("configs", "arch")
    if d is None:
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def describe() -> dict:
    """Diagnostics: every candidate root and whether it is usable."""
    usable = {r for _, r in resource_roots()}
    return {
        "env": ENV_ROOT,
        "roots": [{"kind": k, "path": str(r), "usable": r in usable} for k, r in candidate_roots()],
        "arch_configs": list_arch_configs(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.split("Command line::")[0].strip() if argv else json.dumps(describe(), indent=2))
        if argv:
            print("\nusage: python -m solar.resources [RELATIVE/PATH]")
        return 0
    try:
        print(resource_path(*argv[0].split("/")))
    except ResourceNotFound as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
