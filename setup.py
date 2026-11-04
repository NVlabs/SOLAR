#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build hook for SOLAR; all package metadata lives in pyproject.toml.

The only thing this file adds is a ``build_py`` step that copies the workflow
resources listed in ``solar/resources.py::RESOURCE_TREES`` (hardware configs,
prompt references, skills, verification/tracing scripts, torchview patch,
examples) into ``solar/_data/`` of the built wheel, so an installed SOLAR
resolves them through ``solar.resources`` exactly as a checkout does.  The
sdist carries the same trees at the top level (MANIFEST.in).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import setuptools
from setuptools.command.build_py import build_py as _build_py

HERE = Path(__file__).resolve().parent
_IGNORE = shutil.ignore_patterns("__pycache__", "*.py[cod]", "output", "*_output", ".DS_Store")


def _resource_trees() -> tuple[str, ...]:
    # Load solar/resources.py standalone: importing ``solar`` would pull torch
    # into the build environment.
    spec = importlib.util.spec_from_file_location("_solar_resources", HERE / "solar" / "resources.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return tuple(mod.RESOURCE_TREES)


class build_py(_build_py):
    """Standard build, then copy RESOURCE_TREES into <build_lib>/solar/_data."""

    def run(self) -> None:
        super().run()
        if getattr(self, "editable_mode", False):
            return  # an editable install resolves resources from the checkout
        data_root = Path(self.build_lib) / "solar" / "_data"
        if data_root.exists():
            shutil.rmtree(data_root)
        missing = []
        for rel in _resource_trees():
            src = HERE / rel
            if not src.exists():
                missing.append(rel)
                continue
            dst = data_root / rel
            if src.is_dir():
                shutil.copytree(src, dst, ignore=_IGNORE, symlinks=False)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        if missing:
            raise SystemExit(f"setup.py: resource trees missing from the source tree: {missing}")
        files = sorted(p for p in data_root.rglob("*") if p.is_file())
        manifest = {
            "resource_trees": list(_resource_trees()),
            "files": {str(p.relative_to(data_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        }
        (data_root / "RESOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


setuptools.setup(cmdclass={"build_py": build_py})
